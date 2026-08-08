"""Synthetic smoke runner only; real DALES training is intentionally out of scope."""
import torch
from models import DRNet
from utils.knn import build_hierarchy
from datasets.dales import map_dales_labels
from losses import combined_loss

def synthetic_step(device="cpu"):
    torch.manual_seed(7)
    b, n, m = 2, 2048, 24
    features = torch.randn(b,n,3,device=device)
    hierarchy = build_hierarchy(features[...,:3], [4,4,4,4,2])
    anno = features[:,:m,:3].clone()
    target = map_dales_labels(torch.randint(0,9,(b,m),device=device)).repeat(2,1)
    model = DRNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    logits = model(features, anno, hierarchy, {"option": 2})
    loss, parts = combined_loss(logits, target, torch.ones(8,device=device))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): inference = model(features, anno, hierarchy)
    assert torch.isfinite(logits).all() and torch.isfinite(loss) and all(p.grad is not None for p in model.parameters() if p.requires_grad)
    return {"train_logits": tuple(logits.shape), "inference_logits": tuple(inference.shape), "loss": float(loss), **{k:float(v) for k,v in parts.items()}}

if __name__ == "__main__": print(synthetic_step())
