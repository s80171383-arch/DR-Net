import pickle
import numpy as np
import torch
from sklearn.neighbors import KDTree
from helper_ply import write_ply
from pytorch.datasets.isprs import ISPRSCloudStore, make_isprs_dataloader, map_isprs_labels
from pytorch.isprs_runtime import confusion_matrix, metric_report
from pytorch.losses import combined_loss
from pytorch.models import DRNet

def fixture(tmp_path):
    root=tmp_path; folder=root/'input_0.450'; folder.mkdir()
    rng=np.random.default_rng(4)
    for area,n in [('Area_1',128),('Area_2',96)]:
        xyz=rng.normal(size=(n,3)).astype('f4'); aux=np.column_stack([rng.integers(0,256,n),rng.integers(1,4,n),rng.integers(1,5,n)]).astype('f4'); labels=(np.arange(n)%9).astype('i4')
        assert write_ply(str(folder/f'{area}.ply'),[xyz,aux,labels],['x','y','z','red','green','blue','class'])
        with open(folder/f'{area}_KDTree.pkl','wb') as f: pickle.dump(KDTree(xyz),f)
        with open(folder/f'{area}_coarse_KDTree.pkl','wb') as f: pickle.dump(KDTree(xyz[::4]),f)
    proj=np.arange(171)%96; labels=np.arange(171)%9
    with open(folder/'Area_2_proj.pkl','wb') as f: pickle.dump((proj,labels),f)
    return root

def test_binary_ply_features_trees_projection_and_class_zero(tmp_path):
    store=ISPRSCloudStore(fixture(tmp_path)); assert store.input_labels['training'][0][0]==0
    assert store.input_features['training'][0].shape==(128,3); assert len(store.coarse_trees['training'][0].data)==32
    assert store.val_proj[0].shape==(171,); assert map_isprs_labels(range(9)).tolist()==list(range(9))
    counts,freq,weights=store.class_statistics(); assert counts.sum()==128 and np.isclose(freq.sum(),1) and np.isfinite(weights).all()

def test_hierarchy_nine_logits_loss_backward_and_metrics(tmp_path):
    store=ISPRSCloudStore(fixture(tmp_path)); batch=next(iter(make_isprs_dataloader(store,num_points=64,samples_per_epoch=1)))
    assert len(batch['hierarchy'])==5; model=DRNet(input_dim=3,num_classes=9,num_points=64).train()
    logits=model(batch['features'],batch['xyz_with_anno'],batch['hierarchy'],{'option':2}); assert logits.shape==(2,64,9)
    loss,_=combined_loss(logits,batch['labels_with_anno'].repeat(2,1),torch.ones(9)); loss.backward(); assert torch.isfinite(loss)
    report=metric_report(confusion_matrix(np.arange(9),np.arange(9))); assert report['miou']==1 and report['overall_accuracy']==1
