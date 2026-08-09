"""Strict inspection and one-batch validation of existing ISPRS assets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .datasets.isprs import ISPRSCloudStore, make_isprs_dataloader, PLY_FEATURE_NAMES
from .isprs_runtime import load_config
from .losses import combined_loss
from .models import DRNet

DEFAULT=Path(__file__).with_name("configs")/"isprs.yaml"
def parser():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--config',default=str(DEFAULT)); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--data-only',action='store_true'); p.add_argument('--num-points',type=int); return p
def run(args):
    cfg=load_config(args.config); n=args.num_points or cfg['num_points']
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    store=ISPRSCloudStore(args.data_root,cfg['sub_grid_size'],cfg['labeled_point'],cfg['seed'])
    counts,freq,weights=store.class_statistics()
    print('class counts:',counts.tolist()); print('class frequencies:',freq.tolist()); print('WCE weights:',weights.tolist())
    for split,area in (("training","Area_1"),("validation","Area_2")):
        pts=np.asarray(store.input_trees[split][0].data); aux=store.input_features[split][0]; labels=store.input_labels[split][0]
        print(f'{area}: point count={len(pts)} XYZ range={pts.min(0).tolist()}..{pts.max(0).tolist()}')
        print(f'auxiliary {PLY_FEATURE_NAMES} range={aux.min(0).tolist()}..{aux.max(0).tolist()} histogram={np.bincount(labels,minlength=9).tolist()} KDTree point count={len(pts)} coarse KDTree point count={len(store.coarse_trees[split][0].data)}')
    proj=store.val_proj[0]; print(f'projection shape/range={proj.shape}/{(int(proj.min()),int(proj.max())) if proj.size else "empty"}')
    loader=make_isprs_dataloader(store,'training',1,n,1,cfg['noise_init'],cfg['seed'],cfg['feature_mode']); batch=next(iter(loader)); ds=loader.dataset
    print('sample shape:',tuple(batch['features'].shape),'annotated shape:',tuple(batch['xyz_with_anno'].shape),'label range:',(int(batch['raw_labels'].min()),int(batch['raw_labels'].max())))
    print('hierarchy shapes:',[tuple(x['xyz'].shape) for x in batch['hierarchy']]); print('possibility update:',ds.last_query_stats)
    if args.data_only: return
    device=torch.device(args.device); move=lambda x:x.to(device)
    model=DRNet(input_dim=cfg['input_feature_dim'],num_classes=9,d_out=cfg['d_out'],num_points=n).to(device).train(); opt=torch.optim.Adam(model.parameters(),cfg['learning_rate'])
    hierarchy=[{k:move(v) for k,v in level.items()} for level in batch['hierarchy']]; logits=model(move(batch['features']),move(batch['xyz_with_anno']),hierarchy,{"option":2}); targets=move(batch['labels_with_anno']).repeat(2,1)
    if logits.shape[-1]!=9: raise AssertionError('classifier must emit nine logits')
    loss,parts=combined_loss(logits,targets,torch.tensor(weights,dtype=torch.float32,device=device)); opt.zero_grad(); loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads): raise AssertionError('gradient check failed')
    opt.step(); model.eval()
    with torch.no_grad(): evaluation=model(move(batch['features']),move(batch['xyz_with_anno']),hierarchy)
    print('logits:',tuple(logits.shape),'WCE:',float(parts['weighted_ce']),'Lovasz:',float(parts['lovasz']),'total:',float(loss),'eval:',tuple(evaluation.shape),'PASS')
def main(argv=None): run(parser().parse_args(argv))
if __name__=='__main__': main()
