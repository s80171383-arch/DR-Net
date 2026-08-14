import pytest
import torch

from pytorch.utils.knn import build_hierarchy, knn_search


@pytest.mark.parametrize("k", [3, 8, 12, 16])
@pytest.mark.parametrize("same_points", [False, True])
def test_chunked_knn_matches_full_cdist_cpu(k, same_points):
    generator = torch.Generator().manual_seed(100 + k)
    support = torch.randn(2, 37, 3, generator=generator)
    query = support if same_points else torch.randn(2, 23, 3, generator=generator)
    expected = torch.cdist(query, support).topk(k, largest=False).indices

    actual = knn_search(support, query, k, query_chunk_size=7)

    assert torch.equal(actual, expected)
    assert actual.shape == (2, query.shape[1], k)
    assert actual.dtype == torch.int64
    assert actual.device == query.device
    assert actual.min() >= 0
    assert actual.max() < support.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("k", [3, 8, 12, 16])
def test_chunked_knn_matches_full_cdist_cuda(k):
    support = torch.randn(2, 37, 3, device="cuda")
    query = torch.randn(2, 23, 3, device="cuda")
    expected = torch.cdist(query, support).topk(k, largest=False).indices
    actual = knn_search(support, query, k, query_chunk_size=7)
    assert torch.equal(actual, expected)
    assert actual.device.type == "cuda"


def test_distance_budget_prevents_full_square_cdist(monkeypatch):
    original_cdist = torch.cdist
    calls = []

    def recording_cdist(query, support):
        calls.append((query.shape, support.shape))
        return original_cdist(query, support)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    points = torch.randn(2, 257, 3)
    result = knn_search(points, points, 16, distance_byte_budget=2 * 257 * 4 * 31)

    assert result.shape == (2, 257, 16)
    assert len(calls) == 9
    assert max(shape[0][1] for shape in calls) <= 31
    assert all(query_shape[1] < points.shape[1] for query_shape, _ in calls)


def test_hierarchy_reuses_single_max_k_search_per_level(monkeypatch):
    calls = []
    original_cdist = torch.cdist

    def recording_cdist(query, support):
        calls.append((query.shape, support.shape))
        return original_cdist(query, support)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    hierarchy = build_hierarchy(torch.randn(1, 64, 3), (4, 4), ks=(16, 8, 12))
    assert len(calls) == 2
    assert hierarchy[0]["neigh_idx"].shape == (1, 64, 16)
    assert hierarchy[0]["neigh_idx_1"].shape == (1, 64, 8)
    assert hierarchy[0]["neigh_idx_2"].shape == (1, 64, 12)
