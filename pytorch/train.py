"""Executable CPU smoke test; real DALES training is intentionally out of scope."""
import torch
from .models import DRNet
from .utils.knn import build_hierarchy
from .datasets.dales import map_dales_labels
from .losses import combined_loss

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
    if not torch.isfinite(logits).all():
        raise AssertionError("training logits contain NaN or Inf")
    if not torch.isfinite(loss) or not all(torch.isfinite(value) for value in parts.values()):
        raise AssertionError("loss contains NaN or Inf")
    optimizer.zero_grad()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None for gradient in gradients):
        raise AssertionError("a trainable parameter has no gradient")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("a gradient contains NaN or Inf")
    optimizer.step()
    model.eval()
    with torch.no_grad(): inference = model(features, anno, hierarchy)
    if inference.shape != (b, m, 8) or not torch.isfinite(inference).all():
        raise AssertionError("eval inference failed or doubled its batch")
    return {
        "device": str(device),
        "input_shape": tuple(features.shape),
        "train_logits": tuple(logits.shape),
        "inference_logits": tuple(inference.shape),
        "loss": float(loss),
        **{key: float(value) for key, value in parts.items()},
        "forward": "PASS",
        "backward": "PASS",
        "optimizer_step": "PASS",
        "nan": "NO",
        "inf": "NO",
        "gradients": "PASS",
    }

if __name__ == "__main__": print(synthetic_step())
