import torch
from datasets.dales import NUM_CLASSES, map_dales_labels

def test_dales_mapping():
    assert NUM_CLASSES == 8
    assert map_dales_labels(torch.tensor([0,1,2,8])).tolist()==[-1,0,1,7]
