"""Infer Area_2 and apply the existing legacy projection indices."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .datasets.isprs import ISPRSCloudStore, make_isprs_dataloader
from .isprs_runtime import load_config, confusion_matrix, metric_report
from .models import DRNet

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data-root',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--output-dir',required=True); args=p.parse_args(argv)
    cfg=load_config(args.config); device=torch.device(args.device); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    store=ISPRSCloudStore(args.data_root,cfg['sub_grid_size'],cfg['labeled_point'],cfg['seed']); model=DRNet(cfg['input_feature_dim'],9,cfg['d_out'],True,cfg['num_points']).to(device)
    model.load_state_dict(torch.load(args.checkpoint,map_location=device)['model']); model.eval(); n=len(store.input_labels['validation'][0]); scores=np.zeros((n,9),np.float64); votes=np.zeros(n,np.int64)
    loader=make_isprs_dataloader(store,'validation',1,cfg['num_points'],cfg['val_steps'],cfg['noise_init'],cfg['seed'],cfg['feature_mode'])
    with torch.no_grad():
        for b in loader:
            h=[{k:v.to(device) for k,v in x.items()} for x in b['hierarchy']]; prob=model(b['features'].to(device),b['xyz_with_anno'].to(device),h).softmax(-1)[0].cpu().numpy(); idx=b['point_indices'][0].numpy(); np.add.at(scores,idx,prob); np.add.at(votes,idx,1)
    if (votes==0).any(): print(f'WARNING: {(votes==0).sum()} subsampled points received no vote')
    pred=scores.argmax(1).astype(np.int32); projected=pred[store.val_proj[0]]; np.save(out/'Area_2_subsampled_predictions.npy',pred); np.save(out/'Area_2_projected_predictions.npy',projected)
    if store.val_labels[0].size:
        report=metric_report(confusion_matrix(store.val_labels[0],projected)); print(json.dumps({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in report.items()}))
    else: print('Area_2 original labels unavailable; metrics were not computed.')
if __name__=='__main__': main()
