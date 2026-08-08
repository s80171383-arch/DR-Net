import torch
import torch.nn.functional as F
from pytorch.losses import weighted_cross_entropy, combined_loss
from pytorch.losses.lovasz import lovasz_softmax, lovasz_softmax_flat

def test_wce_arithmetic_mean_not_weight_sum():
    logits=torch.tensor([[[2.,0.],[0.,2.]]]); target=torch.tensor([[0,1]]); weights=torch.tensor([2.,4.])
    ce=F.cross_entropy(logits.reshape(-1,2),target.reshape(-1),reduction="none")
    got=weighted_cross_entropy(logits,target,weights)
    torch.testing.assert_close(got,(ce*weights[target.reshape(-1)]).sum()/2)
    assert not torch.isclose(got,(ce*weights[target.reshape(-1)]).sum()/weights[target].sum())

def test_lovasz_flattens_whole_valid_batch_once():
    logits=torch.randn(2,3,8); target=torch.tensor([[0,1,-1],[1,2,3]])
    valid=target.reshape(-1)!=-1
    expected=lovasz_softmax_flat(logits.softmax(-1).reshape(-1,8)[valid],target.reshape(-1)[valid],"present")
    torch.testing.assert_close(lovasz_softmax(logits,target),expected)
    total,parts=combined_loss(logits,target,torch.ones(8))
    torch.testing.assert_close(total,parts["weighted_ce"]+parts["lovasz"])
