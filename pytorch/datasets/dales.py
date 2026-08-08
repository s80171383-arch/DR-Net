import torch

NUM_CLASSES = 8

def map_dales_labels(raw):
    """Map raw 1..8 to 0..7 and raw unknown 0 to ignore index -1."""
    raw = torch.as_tensor(raw).long()
    if ((raw < 0) | (raw > 8)).any():
        raise ValueError("DALES raw labels must be in [0, 8]")
    return torch.where(raw == 0, -torch.ones_like(raw), raw - 1)
