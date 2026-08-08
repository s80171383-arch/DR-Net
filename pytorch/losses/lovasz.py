import torch
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    return torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1])) if p > 1 else jaccard

def lovasz_softmax_flat(probs, labels, classes="present"):
    losses = []
    for c in range(probs.shape[1]):
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else probs.sum() * 0

def lovasz_softmax(logits, target, ignore_index=-1, classes="present"):
    # One flatten over the entire valid batch, matching Lovasz_losses_tf.py.
    valid = target.reshape(-1) != ignore_index
    probs = F.softmax(logits.reshape(-1, logits.shape[-1])[valid], -1)
    return lovasz_softmax_flat(probs, target.reshape(-1)[valid], classes)
