"""Train the shared DRNet implementation on the fixed Area_1/Area_2 split."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .datasets.isprs import ISPRSCloudStore, make_isprs_dataloader
from .isprs_runtime import load_config, confusion_matrix, metric_report
from .losses import combined_loss
from .models import DRNet

def paper_learning_rate(base_learning_rate, epoch):
    """Return the DR-Net paper learning rate for an absolute epoch number."""
    return base_learning_rate * (0.95 ** epoch)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data-root',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--output-dir',required=True); p.add_argument('--resume'); p.add_argument('--epochs',type=int); args=p.parse_args(argv)
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
