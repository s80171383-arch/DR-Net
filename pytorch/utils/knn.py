"""Exact KNN implementations used by the DR-Net hierarchy."""

import numpy as np
import torch
from scipy.spatial import cKDTree


DEFAULT_DISTANCE_BYTES = 256 * 1024 * 1024
KNN_BACKENDS = ("torch", "ckdtree")


def _validate_knn_inputs(support, query, k):
    if support.ndim != 3 or query.ndim != 3:
        raise ValueError("support and query must have shape [B, N, C]")
    if support.shape[0] != query.shape[0] or support.shape[2] != query.shape[2]:
        raise ValueError("support and query batch/coordinate dimensions must match")
    if support.device != query.device or support.dtype != query.dtype:
        raise ValueError("support and query must have the same device and dtype")
    if not 0 < k <= support.shape[1]:
        raise ValueError("k must be positive and no larger than the support size")


def _query_chunk_size(support, query, distance_byte_budget):
    """Choose a query chunk whose materialized distance tensor fits the budget."""
    distance_bytes = torch.empty((), dtype=query.dtype).element_size()
    per_query_bytes = support.shape[0] * support.shape[1] * distance_bytes
    return max(1, min(query.shape[1], distance_byte_budget // per_query_bytes))


def _torch_knn(support, query, k, query_chunk_size, distance_byte_budget):
    """Reference implementation: chunked ``torch.cdist`` followed by ``topk``."""
    if distance_byte_budget <= 0:
        raise ValueError("distance_byte_budget must be positive")
    if query_chunk_size is None:
        query_chunk_size = _query_chunk_size(support, query, distance_byte_budget)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    chunks = []
    for start in range(0, query.shape[1], query_chunk_size):
        query_chunk = query[:, start:start + query_chunk_size]
        chunks.append(torch.cdist(query_chunk, support).topk(k, largest=False).indices)
    return torch.cat(chunks, dim=1)


def _ckdtree_knn(support, query, k, workers):
    """Exact CPU KNN using SciPy's highly optimized C++ kd-tree.

    cKDTree returns neighbors in increasing-distance order.  ``torch.topk`` is
    also distance-sorted, but either implementation may choose a different
    index at an exactly tied boundary.  No secondary index sort is applied.
    """
    if support.device.type != "cpu":
        raise ValueError("the ckdtree backend supports CPU tensors only")
    if not isinstance(workers, int) or workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer")

    # Tensor.numpy() is zero-copy for the normal contiguous hierarchy tensors.
    # cKDTree itself converts coordinates to float64 internally.
    support_np = support.detach().numpy()
    query_np = query.detach().numpy()
    batches = []
    for batch in range(support.shape[0]):
        tree = cKDTree(support_np[batch])
        _, indices = tree.query(query_np[batch], k=k, workers=workers)
        if k == 1:
            indices = indices[:, None]
        batches.append(np.asarray(indices, dtype=np.int64))
    return torch.from_numpy(np.stack(batches)).to(device=query.device)


def knn_search(support, query, k, *, backend="torch", query_chunk_size=None,
               distance_byte_budget=DEFAULT_DISTANCE_BYTES, workers=-1):
    """Return exact Euclidean KNN indices for batched ``[B, N, C]`` tensors.

    ``backend='torch'`` retains the reference cdist/topk algorithm and works on
    CPU or CUDA. ``backend='ckdtree'`` is an exact, CPU-only optimized backend.
    Support and query roles are never interchanged.
    """
    _validate_knn_inputs(support, query, k)
    if backend == "torch":
        return _torch_knn(support, query, k, query_chunk_size, distance_byte_budget)
    if backend == "ckdtree":
        return _ckdtree_knn(support, query, k, workers)
    raise ValueError(f"backend must be one of {KNN_BACKENDS}, got {backend!r}")


def build_hierarchy(xyz, ratios, ks=(16, 8, 12), *, backend="torch", workers=-1):
    """Build exact neighborhoods with TF-compatible prefix downsampling.

    Prefix sampling and the nested max-K optimization are identical for both
    backends; this is explicitly not farthest-point sampling.
    """
    levels = []
    current = xyz
    for ratio in ratios:
        capped_ks = [min(k, current.shape[1]) for k in ks]
        max_indices = knn_search(current, current, max(capped_ks),
                                 backend=backend, workers=workers)
        indices = [max_indices[:, :, :k] for k in capped_ks]
        count = current.shape[1] // ratio
        levels.append({"xyz": current, "neigh_idx": indices[0],
                       "neigh_idx_1": indices[1], "neigh_idx_2": indices[2],
                       "sub_idx": indices[0][:, :count]})
        current = current[:, :count]
    return levels
