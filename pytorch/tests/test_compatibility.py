import torch
from torch import nn
from pytorch.models.blocks import LEAKY_RELU_SLOPE, BuildingBlock, ConvBNAct
from pytorch.models.drnet import DRNet
from pytorch.utils.knn import build_hierarchy
from pytorch.utils.interpolation import semantic_query

def test_k12_is_unused():
    torch.manual_seed(1); b = BuildingBlock(4,16).eval()
    xyz=torch.randn(1,20,3); feature=torch.randn(1,4,20)
    idx16=torch.randint(20,(1,20,16)); idx8=torch.randint(20,(1,20,8))
    with torch.no_grad():
        a=b(xyz,feature,idx16,idx8,torch.zeros(1,20,12,dtype=torch.long))
        c=b(xyz,feature,idx16,idx8,torch.randint(20,(1,20,12)))
    torch.testing.assert_close(a,c)

def test_prefix_sampling():
    xyz=torch.arange(96.).view(1,32,3)
    hierarchy=build_hierarchy(xyz,[4])
    torch.testing.assert_close(hierarchy[0]["xyz"][:,:8],xyz[:,:8])
    assert hierarchy[0]["sub_idx"].shape[1] == 8

def test_semantic_query_is_unweighted_mean():
    f=torch.tensor([[[1.,2.,6.]]]); idx=torch.tensor([[[0,1,2]]])
    assert semantic_query(f,idx,torch.tensor([[[1.,2.,99.]]])).item() == 3
    assert semantic_query(f,idx,torch.tensor([[[99.,2.,1.]]])).item() == 3

def test_all_leaky_relu_slopes_are_tensorflow_value():
    model=DRNet(num_points=32)
    activations=[m for m in model.modules() if isinstance(m,nn.LeakyReLU)]
    assert activations
    assert all(m.negative_slope == LEAKY_RELU_SLOPE == 0.2 for m in activations)

def test_tensorflow_batch_norm_update_semantics():
    # TF: moving = .99 * moving + .01 * batch. PyTorch uses the batch weight.
    block=ConvBNAct(2,3)
    bn=next(m for m in block if isinstance(m,nn.BatchNorm1d))
    assert bn.momentum == 0.01 and bn.eps == 1e-6
