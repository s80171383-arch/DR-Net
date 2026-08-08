import pickle
from pathlib import Path

import numpy as np
import pytest

from helper_ply import write_ply
from pytorch.datasets.dales import (DALESCloudStore, DALESSpatiallyRegularDataset,
                            make_dales_dataloader)
from pytorch.datasets.verify_dales import main as verify_main


class FixtureKDTree:
    """Minimal sklearn KDTree-compatible object for the on-disk test fixture."""
    def __init__(self, data):
        self.data = np.asarray(data)

    def query(self, centers, k):
        distances = ((self.data[None] - centers[:, None]) ** 2).sum(axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        return np.take_along_axis(np.sqrt(distances), indices, axis=1), indices


def _cloud(root: Path, split: str, name: str, count: int):
    original = root / "original_ply" / split
    sampled = root / "input_0.320" / split
    original.mkdir(parents=True, exist_ok=True)
    sampled.mkdir(parents=True, exist_ok=True)
    points = np.column_stack((np.arange(count), np.arange(count) % 3, np.zeros(count))).astype("f4")
    labels = (np.arange(count) % 8 + 1).astype("i4")
    write_ply(str(original / name), [points, labels], ["x", "y", "z", "class"])
    write_ply(str(sampled / name), [points, labels], ["x", "y", "z", "class"])
    with (sampled / f"{name}_KDTree.pkl").open("wb") as stream:
        pickle.dump(FixtureKDTree(points), stream)
    if split == "test":
        with (sampled / f"{name}_proj.pkl").open("wb") as stream:
            pickle.dump((np.arange(count - 1, -1, -1), labels.copy()), stream)
    return points, labels


@pytest.fixture
def dales_root(tmp_path):
    train = _cloud(tmp_path, "train", "train_cloud", 11)
    test = _cloud(tmp_path, "test", "test_cloud", 80)
    return tmp_path, train, test


def test_store_split_loaders_projection_and_sparse_percent(dales_root):
    root, _, test = dales_root
    store = DALESCloudStore(root, labeled_point="10%", seed=3)
    assert store.input_names == {"training": ["train_cloud"], "validation": ["test_cloud"]}
    assert len(store.input_trees["training"][0].data) == 11
    assert np.count_nonzero(store.input_labels["training"][0]) == 1
    assert np.array_equal(store.val_proj[0], np.arange(79, -1, -1))
    assert np.array_equal(store.val_labels[0], test[1])


def test_fixed_per_class_sparse_annotation(dales_root):
    store = DALESCloudStore(dales_root[0], labeled_point="1", seed=2)
    labels = store.input_labels["training"][0]
    assert all(np.count_nonzero(labels == value) <= 1 for value in range(1, 9))


def test_spatial_query_update_shuffle_replacement_and_hierarchy(dales_root):
    store = DALESCloudStore(dales_root[0], labeled_point="100%", seed=1)
    dataset = DALESSpatiallyRegularDataset(store, "training", num_points=64,
                                           samples_per_epoch=1, noise_init=0, seed=9)
    sample = next(iter(dataset))
    assert sample["xyz"].shape == (64, 3) and sample["point_indices"].shape == (64,)
    assert len(np.unique(sample["point_indices"])) == 11  # replacement upsampling
    assert not np.array_equal(sample["point_indices"][:11].numpy(), np.arange(11))
    assert dataset.min_possibility[0] > 0
    assert sample["xyz_with_anno"].shape == (64, 3)
    assert sample["labels_with_anno"].min() == 0 and sample["labels_with_anno"].max() == 7
    batch = next(iter(make_dales_dataloader(store, "validation", batch_size=1,
                                            num_points=1024, samples_per_epoch=1,
                                            noise_init=0, seed=4)))
    assert batch["xyz"].shape == (1, 1024, 3)
    assert batch["raw_labels"].shape == (1, 1024)
    assert batch["xyz_with_anno"].shape == (1, 1024, 3)
    assert len(batch["hierarchy"]) == 5
    assert [tuple(level["neigh_idx"].shape) for level in batch["hierarchy"]] == [
        (1, 1024, 16), (1, 256, 16), (1, 64, 16), (1, 16, 16), (1, 4, 4)]
    assert all("neigh_idx_2" in level for level in batch["hierarchy"])


def test_validation_cli_and_clear_missing_file(dales_root, capsys):
    verify_main(["--data-root", str(dales_root[0])])
    assert "OK validation projections: 1" in capsys.readouterr().out
    (dales_root[0] / "input_0.320/test/test_cloud_proj.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="projection is missing"):
        DALESCloudStore(dales_root[0])


def test_projection_range_is_validated(dales_root):
    path = dales_root[0] / "input_0.320/test/test_cloud_proj.pkl"
    with path.open("wb") as stream:
        pickle.dump((np.array([80]), np.array([1])), stream)
    with pytest.raises(ValueError, match="projection index"):
        DALESCloudStore(dales_root[0])
