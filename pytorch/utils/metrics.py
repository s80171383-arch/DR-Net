import torch

def accuracy(logits, target, ignore_index=-1):
    valid = target != ignore_index
    return (logits.argmax(-1)[valid] == target[valid]).float().mean()
