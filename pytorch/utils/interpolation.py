from models.blocks import gather_neighbour


def semantic_query(features, neighbour_idx, distances=None):
    """Original three_interpolate call with fixed 1/3 weights; distances unused."""
    del distances
    if neighbour_idx.shape[-1] != 3:
        raise ValueError("Semantic Query requires exactly 3 neighbours")
    return gather_neighbour(features.transpose(1, 2), neighbour_idx).mean(dim=2).transpose(1, 2)
