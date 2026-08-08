import torch
from pytorch.models import DRNet
from pytorch.utils.knn import build_hierarchy
from pytorch.datasets.dales import map_dales_labels
from pytorch.losses import combined_loss

def test_synthetic_end_to_end():
    torch.manual_seed(4); b,n,m=1,1024,9
    x=torch.randn(b,n,3); anno=x[:,:m].clone(); h=build_hierarchy(x,[4,4,4,4,2])
    model=DRNet(); model.train(); logits=model(x,anno,h,{"option":2})
    assert logits.shape==(2*b,m,8) and torch.isfinite(logits).all()
    target=map_dales_labels(torch.tensor([[0,1,2,3,4,5,6,7,8]])).repeat(2,1)
    loss,_=combined_loss(logits,target,torch.arange(1,9).float()); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.parameters() if p.requires_grad)
