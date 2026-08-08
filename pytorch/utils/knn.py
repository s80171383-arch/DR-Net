import torch


def knn_search(support, query, k):
    return torch.cdist(query, support).topk(k, largest=False).indices


def build_hierarchy(xyz, ratios, ks=(16, 8, 12)):
    """TF-compatible prefix downsampling; this is explicitly not FPS."""
    levels = []
    current = xyz
    for ratio in ratios:
        indices = [knn_search(current, current, min(k, current.shape[1])) for k in ks]
        count = current.shape[1] // ratio
        levels.append({"xyz": current, "neigh_idx": indices[0], "neigh_idx_1": indices[1], "neigh_idx_2": indices[2], "sub_idx": indices[0][:, :count]})
        current = current[:, :count]
    return levels
