"""Train the shared DRNet implementation on the fixed Area_1/Area_2 split."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .datasets.isprs import ISPRSCloudStore, make_isprs_dataloader
from .isprs_runtime import load_config, confusion_matrix, metric_report
from .losses import combined_loss
from .models import DRNet

PROFILE_FIELDS = ('batch_fetch','cpu_preparation','host_to_device','forward','loss','backward','optimizer_step','total')

def parse_args(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data-root',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--output-dir',required=True); p.add_argument('--resume'); p.add_argument('--epochs',type=int); p.add_argument('--profile-steps',type=int,default=0,metavar='N'); args=p.parse_args(argv)
    if args.profile_steps < 0: p.error('--profile-steps must be non-negative')
    return args

def profile_batches(loader, count):
    """Yield at most count batches together with their DataLoader wait time."""
    iterator=iter(loader)
    for _ in range(count):
        started=time.perf_counter()
        try: batch=next(iterator)
        except StopIteration: return
        yield batch,time.perf_counter()-started

def _sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)

def _summary(samples):
    print('ISPRS step profile summary (seconds):')
    for name in PROFILE_FIELDS:
        values=np.asarray([sample[name] for sample in samples])
        print(f'  {name:16s} mean={values.mean():.6f} min={values.min():.6f} max={values.max():.6f}')

def paper_learning_rate(base_learning_rate, epoch):
    """Return the DR-Net paper learning rate for an absolute epoch number."""
    return base_learning_rate * (0.95 ** epoch)

def main(argv=None):
    args=parse_args(argv)
    cfg=load_config(args.config); device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    store=ISPRSCloudStore(args.data_root,cfg['sub_grid_size'],cfg['labeled_point'],cfg['seed']); _,_,w=store.class_statistics()
    model=DRNet(cfg['input_feature_dim'],9,cfg['d_out'],True,cfg['num_points']).to(device); optimizer=torch.optim.Adam(model.parameters(),cfg['learning_rate']); start=0; best=-1.
    if args.resume:
        state=torch.load(args.resume,map_location=device); model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); start=state['epoch']+1; best=state.get('best_miou',best)
    weights=torch.tensor(w,dtype=torch.float32,device=device); epochs=args.epochs or cfg['epochs']
    for epoch in range(start,epochs):
        learning_rate=paper_learning_rate(cfg['learning_rate'],epoch)
        for group in optimizer.param_groups: group['lr']=learning_rate
        model.train(); losses=[]
        loader=make_isprs_dataloader(store,'training',cfg['batch_size'],cfg['num_points'],cfg['train_steps']*cfg['batch_size'],cfg['noise_init'],cfg['seed']+epoch,cfg['feature_mode'])
        if args.profile_steps:
            samples=[]
            for step,(b,fetch_time) in enumerate(profile_batches(loader,args.profile_steps),1):
                total_start=time.perf_counter()-fetch_time
                started=time.perf_counter(); h_cpu=[{k:v for k,v in x.items()} for x in b['hierarchy']]; features_cpu=b['features']; xyz_cpu=b['xyz_with_anno']; labels_cpu=b['labels_with_anno']; cpu_time=time.perf_counter()-started
                if step == 1: print(f"ISPRS step profile: batch_size={features_cpu.shape[0]} num_points={cfg['num_points']} labeled_ratio={cfg['labeled_point']} xyz_with_anno_shape={tuple(xyz_cpu.shape)} device={device} cpu_threads={torch.get_num_threads()}")
                _sync(device); started=time.perf_counter(); h=[{k:v.to(device) for k,v in x.items()} for x in h_cpu]; features=features_cpu.to(device); xyz=xyz_cpu.to(device); labels=labels_cpu.to(device); _sync(device); transfer_time=time.perf_counter()-started
                started=time.perf_counter(); logits=model(features,xyz,h,{"option":2}); _sync(device); forward_time=time.perf_counter()-started
                started=time.perf_counter(); target=labels.repeat(2,1); loss,_=combined_loss(logits,target,weights); _sync(device); loss_time=time.perf_counter()-started
                optimizer.zero_grad(); _sync(device); started=time.perf_counter(); loss.backward(); _sync(device); backward_time=time.perf_counter()-started
                started=time.perf_counter(); optimizer.step(); _sync(device); optimizer_time=time.perf_counter()-started; losses.append(float(loss))
                sample=dict(zip(PROFILE_FIELDS,(fetch_time,cpu_time,transfer_time,forward_time,loss_time,backward_time,optimizer_time,time.perf_counter()-total_start))); samples.append(sample)
                print(f"profile step {step}/{args.profile_steps}: "+' '.join(f'{k}={v:.6f}s' for k,v in sample.items()))
            _summary(samples)
            return
        else:
            # Keep the original non-profiled loop byte-for-byte equivalent.
            for b in loader:
                h=[{k:v.to(device) for k,v in x.items()} for x in b['hierarchy']]; logits=model(b['features'].to(device),b['xyz_with_anno'].to(device),h,{"option":2}); target=b['labels_with_anno'].to(device).repeat(2,1); loss,_=combined_loss(logits,target,weights); optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss))
        model.eval(); cm=np.zeros((9,9),np.int64)
        val=make_isprs_dataloader(store,'validation',cfg['val_batch_size'],cfg['num_points'],cfg['val_steps']*cfg['val_batch_size'],cfg['noise_init'],cfg['seed'],cfg['feature_mode'])
        with torch.no_grad():
            for b in val:
                h=[{k:v.to(device) for k,v in x.items()} for x in b['hierarchy']]; pred=model(b['features'].to(device),b['xyz_with_anno'].to(device),h).argmax(-1).cpu().numpy(); cm+=confusion_matrix(b['labels_with_anno'].numpy(),pred)
        report=metric_report(cm); serial={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in report.items()}; print(json.dumps({'epoch':epoch,'learning_rate':learning_rate,'loss':np.mean(losses),**serial}))
        state={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best_miou':max(best,report['miou']),'config':cfg}; torch.save(state,out/'checkpoint_last.pt')
        if report['miou']>best: best=report['miou']; torch.save(state,out/'checkpoint_best.pt')
if __name__=='__main__': main()
