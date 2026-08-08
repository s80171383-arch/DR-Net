import torch.nn.functional as F

def weighted_cross_entropy(logits, target, class_weights, ignore_index=-1):
    valid = target != ignore_index
    losses = F.cross_entropy(logits[valid], target[valid], reduction="none")
    # TF reduce_mean: arithmetic point mean, never normalized by sum(weights).
    return (losses * class_weights.to(logits)[target[valid]]).mean()
