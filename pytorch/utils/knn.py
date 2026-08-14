import torch


DEFAULT_DISTANCE_BYTES = 256 * 1024 * 1024


def _query_chunk_size(support, query, distance_byte_budget):
    """Choose a query chunk whose materialized distance tensor fits the budget."""
    distance_bytes = torch.empty((), dtype=query.dtype).element_size()
    per_query_bytes = support.shape[0] * support.shape[1] * distance_bytes
    return max(1, min(query.shape[1], distance_byte_budget // per_query_bytes))


def knn_search(support, query, k, *, query_chunk_size=None,
               distance_byte_budget=DEFAULT_DISTANCE_BYTES):
    """Return exact Euclidean KNN indices without materializing a full Q x S matrix.

    ``support`` and ``query`` are batched ``[B, N, C]`` tensors.  Chunking only
    partitions the query dimension: every query is still compared with every
    support point, so this has the same mathematical meaning as a single
    ``torch.cdist(query, support).topk(...)`` call.
    """
    if support.ndim != 3 or query.ndim != 3:
        raise ValueError("support and query must have shape [B, N, C]")
    if support.shape[0] != query.shape[0] or support.shape[2] != query.shape[2]:
        raise ValueError("support and query batch/coordinate dimensions must match")
    if support.device != query.device or support.dtype != query.dtype:
        raise ValueError("support and query must have the same device and dtype")
    if not 0 < k <= support.shape[1]:
        raise ValueError("k must be positive and no larger than the support size")
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


def build_hierarchy(xyz, ratios, ks=(16, 8, 12)):
    """TF-compatible prefix downsampling; this is explicitly not FPS."""
    levels = []
    current = xyz
    for ratio in ratios:
        # One exact max-K search supplies the nested hierarchy neighborhoods;
        # avoid repeating the same all-pairs distance calculation for K16/8/12.
        capped_ks = [min(k, current.shape[1]) for k in ks]
        max_indices = knn_search(current, current, max(capped_ks))
        indices = [max_indices[:, :, :k] for k in capped_ks]
        count = current.shape[1] // ratio
        levels.append({"xyz": current, "neigh_idx": indices[0], "neigh_idx_1": indices[1], "neigh_idx_2": indices[2], "sub_idx": indices[0][:, :count]})
        current = current[:, :count]
    return levels
