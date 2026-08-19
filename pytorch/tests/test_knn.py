import pytest
import torch

from pytorch.utils.knn import build_hierarchy, knn_search


@pytest.mark.parametrize("k", [8, 12, 16])
@pytest.mark.parametrize("sizes", [(1, 64, 41), (2, 257, 83)])
def test_ckdtree_matches_torch_neighbor_sets_without_ties(k, sizes):
    batch, support_count, query_count = sizes
    generator = torch.Generator().manual_seed(6000 + k + support_count)
    support = torch.randn(batch, support_count, 3, generator=generator,
                          dtype=torch.float64)
    query = torch.randn(batch, query_count, 3, generator=generator,
                        dtype=torch.float64)

    reference = knn_search(support, query, k, backend="torch", query_chunk_size=17)
    actual = knn_search(support, query, k, backend="ckdtree", workers=1)

    assert actual.shape == reference.shape == (batch, query_count, k)
    assert actual.dtype == reference.dtype == torch.int64
    assert torch.equal(actual, reference), _parity_mismatch(reference, actual)


def _parity_mismatch(reference, actual):
    mismatches = torch.nonzero(reference != actual)
    return f"KNN parity mismatch at {mismatches[:20].tolist()} ({len(mismatches)} entries)"


def test_ckdtree_tied_boundary_is_mathematically_equivalent():
    support = torch.tensor([[[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.],
                             [0., 1., 0.], [0., 0., 2.]]], dtype=torch.float64)
    query = torch.zeros(1, 1, 3, dtype=torch.float64)
    reference = knn_search(support, query, 2, backend="torch")
    actual = knn_search(support, query, 2, backend="ckdtree", workers=1)

    # Indices at the tied K boundary may differ, but both selected multisets of
    # Euclidean distances must equal the mathematically minimal distances.
    all_distances = torch.cdist(query, support)
    expected_distances = all_distances.sort(dim=-1).values[..., :2]
    reference_distances = all_distances.gather(-1, reference).sort(dim=-1).values
    actual_distances = all_distances.gather(-1, actual).sort(dim=-1).values
    assert torch.equal(reference_distances, expected_distances), _parity_mismatch(reference, actual)
    assert torch.equal(actual_distances, expected_distances), _parity_mismatch(reference, actual)


def test_hierarchy_ckdtree_matches_reference_at_every_level():
    points = torch.randn(2, 1024, 3, generator=torch.Generator().manual_seed(712),
                         dtype=torch.float64)
    reference = build_hierarchy(points, (4, 4, 4), backend="torch")
    actual = build_hierarchy(points, (4, 4, 4), backend="ckdtree", workers=1)

    assert len(actual) == len(reference)
    for level_number, (expected_level, actual_level) in enumerate(zip(reference, actual)):
        assert expected_level.keys() == actual_level.keys()
        for name in expected_level:
            assert actual_level[name].shape == expected_level[name].shape
            assert torch.equal(actual_level[name], expected_level[name]), (
                f"hierarchy level {level_number} field {name}: "
                + _parity_mismatch(expected_level[name], actual_level[name]))


def test_ckdtree_rejects_cuda_and_unknown_backend():
    points = torch.randn(1, 20, 3)
    with pytest.raises(ValueError, match="backend must be"):
        knn_search(points, points, 3, backend="approximate")


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
