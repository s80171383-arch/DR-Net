import torch
from models.blocks import BuildingBlock
from utils.knn import build_hierarchy
from utils.interpolation import semantic_query

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
