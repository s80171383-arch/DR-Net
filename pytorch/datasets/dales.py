"""Faithful, stateful DALES input pipeline used by the TensorFlow release."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from helper_ply import read_ply

if __package__ == "datasets":
    from utils.knn import build_hierarchy
else:
    from pytorch.utils.knn import build_hierarchy

NUM_CLASSES = 8
SPLITS = {"training": "train", "validation": "test"}


def map_dales_labels(raw):
    """Map raw 1..8 to network targets 0..7 and unknown 0 to ignore -1."""
    raw = torch.as_tensor(raw).long()
    if ((raw < 0) | (raw > 8)).any():
        raise ValueError("DALES raw labels must be in [0, 8]")
    return torch.where(raw == 0, -torch.ones_like(raw), raw - 1)


class DALESCloudStore:
    """Load the author's subsampled PLY, KDTree and test projection layout."""

    def __init__(self, data_root, sub_grid_size: float = 0.32, labeled_point="100%",
                 seed: int = 0, retrain: bool = False):
        if retrain:
            raise NotImplementedError(
                "pseudo-label retraining is deferred: the TensorFlow reference uses an "
                "undefined `cloud_name` in this path (legacy-reference issue)")
        self.root = Path(data_root)
        self.subsampled_root = self.root / f"input_{sub_grid_size:.3f}"
        self.input_trees: Dict[str, List] = {s: [] for s in SPLITS}
        self.input_labels: Dict[str, List[np.ndarray]] = {s: [] for s in SPLITS}
        self.input_names: Dict[str, List[str]] = {s: [] for s in SPLITS}
        self.val_proj: List[np.ndarray] = []
        self.val_labels: List[np.ndarray] = []
        self.rng = np.random.default_rng(seed)
        self.labeled_point = str(labeled_point)
        self._load()

    def _load(self):
        for split, disk_split in SPLITS.items():
            originals = sorted((self.root / "original_ply" / disk_split).glob("*.ply"))
            if not originals:
                raise FileNotFoundError(f"no PLY files in {self.root / 'original_ply' / disk_split}")
            folder = self.subsampled_root / disk_split
            for original in originals:
                name = original.stem
                ply_path, tree_path = folder / f"{name}.ply", folder / f"{name}_KDTree.pkl"
                for path in (ply_path, tree_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"required DALES file is missing: {path}")
                data = read_ply(str(ply_path))
                required = {"x", "y", "z", "class"}
                if not required.issubset(data.dtype.names or ()):
                    raise ValueError(f"{ply_path}: required PLY fields are {sorted(required)}")
                labels = np.asarray(data["class"], dtype=np.int32).copy()
                if labels.size == 0 or labels.min() < 0 or labels.max() > 8:
                    raise ValueError(f"{ply_path}: class labels must be in [0, 8]")
                with tree_path.open("rb") as stream:
                    tree = pickle.load(stream)
                points = np.asarray(tree.data)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(labels):
                    raise ValueError(f"{tree_path}: tree point count/shape does not match PLY labels")
                if split == "training":
                    labels = self._sparsify(labels)
                self.input_trees[split].append(tree)
                self.input_labels[split].append(labels)
                self.input_names[split].append(name)
                if split == "validation":
                    proj_path = folder / f"{name}_proj.pkl"
                    if not proj_path.is_file():
                        raise FileNotFoundError(f"required DALES projection is missing: {proj_path}")
                    with proj_path.open("rb") as stream:
                        proj, raw_labels = pickle.load(stream)
                    proj, raw_labels = np.asarray(proj), np.asarray(raw_labels)
                    if proj.size and (proj.min() < 0 or proj.max() >= len(points)):
                        raise ValueError(f"{proj_path}: projection index outside [0, {len(points)})")
                    if raw_labels.size and (raw_labels.min() < 0 or raw_labels.max() > 8):
                        raise ValueError(f"{proj_path}: labels must be in [0, 8]")
                    self.val_proj.append(proj)
                    self.val_labels.append(raw_labels)

    def _sparsify(self, labels):
        result = labels.copy()
        if "%" in self.labeled_point:
            ratio = float(self.labeled_point.removesuffix("%")) / 100
            if not 0 <= ratio <= 1:
                raise ValueError("labeled_point percentage must be in [0%, 100%]")
            keep = max(int(len(result) * ratio), 1)
            drop = self.rng.choice(len(result), len(result) - keep, replace=False)
            result[drop] = 0
        else:
            count = int(self.labeled_point)
            if count < 1:
                raise ValueError("fixed labeled_point count must be positive")
            for raw_class in range(1, 9):
                indices = np.flatnonzero(result == raw_class)
                if len(indices) > count:
                    result[self.rng.choice(indices, len(indices) - count, replace=False)] = 0
        return result


class DALESSpatiallyRegularDataset(IterableDataset):
    """The stateful minimum-possibility generator from ``get_batch_gen``."""

    def __init__(self, store: DALESCloudStore, split="training", num_points=65536,
                 samples_per_epoch=500, noise_init=3.5, seed=0):
        super().__init__()
        if split not in SPLITS:
            raise ValueError("split must be 'training' or 'validation'")
        self.store, self.split = store, split
        self.num_points, self.samples_per_epoch = num_points, samples_per_epoch
        self.noise_init, self.seed = noise_init, seed
        if "%" in store.labeled_point:
            ratio = float(store.labeled_point.removesuffix("%")) / 100
            self.num_with_anno_per_batch = max(int(num_points * ratio), 1)
        else:
            # Reference behavior: fixed-per-class mode retains num_classes points/query.
            self.num_with_anno_per_batch = NUM_CLASSES
        self.possibility, self.min_possibility = [], []

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError("stateful DALES possibility sampling requires num_workers=0")
        rng = np.random.default_rng(self.seed)
        self.possibility = [rng.random(len(x)) * 1e-3 for x in self.store.input_labels[self.split]]
        self.min_possibility = [float(x.min()) for x in self.possibility]
        yielded = 0
        while yielded < self.samples_per_epoch:
            cloud_idx = int(np.argmin(self.min_possibility))
            point_idx = int(np.argmin(self.possibility[cloud_idx]))
            tree = self.store.input_trees[self.split][cloud_idx]
            points = np.asarray(tree.data)
            center = points[point_idx].reshape(1, -1)
            pick = center + rng.normal(scale=self.noise_init / 10, size=center.shape).astype(center.dtype)
            query_count = min(len(points), self.num_points)
            queried_idx = np.asarray(tree.query(pick, k=query_count)[1][0], dtype=np.int64)
            queried_idx = rng.permutation(queried_idx)
            xyz = points[queried_idx] - pick
            labels = self.store.input_labels[self.split][cloud_idx][queried_idx]
            dists = np.sum(np.square((points[queried_idx] - pick).astype(np.float32)), axis=1)
            maximum = np.max(dists)
            delta = np.square(1 - dists / maximum) if maximum > 0 else np.ones_like(dists)
            np.add.at(self.possibility[cloud_idx], queried_idx, delta)
            self.min_possibility[cloud_idx] = float(self.possibility[cloud_idx].min())
            if query_count < self.num_points:
                duplicates = rng.choice(query_count, self.num_points - query_count, replace=True)
                xyz = np.concatenate((xyz, xyz[duplicates]))
                labels = np.concatenate((labels, labels[duplicates]))
                queried_idx = np.concatenate((queried_idx, queried_idx[duplicates]))
            annotated = np.flatnonzero(labels != 0)
            if self.split == "training":
                # The reference skips crops containing no usable supervised classes.
                if len(np.unique(labels)) <= 1 or len(annotated) == 0:
                    continue
                n = self.num_with_anno_per_batch
                annotated = rng.choice(annotated, n, replace=len(annotated) < n)
            else:
                annotated = np.arange(len(labels))
            yielded += 1
            yield {"xyz": torch.from_numpy(xyz.astype(np.float32)),
                   "raw_labels": torch.from_numpy(labels.astype(np.int64)),
                   "point_indices": torch.from_numpy(queried_idx.astype(np.int64)),
                   "cloud_index": torch.tensor(cloud_idx, dtype=torch.long),
                   "xyz_with_anno": torch.from_numpy(xyz[annotated].astype(np.float32)),
                   "labels_with_anno": map_dales_labels(labels[annotated]),
                   "cloud_name": self.store.input_names[self.split][cloud_idx]}


def dales_collate(samples, ratios=(4, 4, 4, 4, 2)):
    """Stack fixed-size samples and attach the unchanged K16/K8/K12 hierarchy."""
    batch = {key: ([s[key] for s in samples] if key == "cloud_name" else
                   torch.stack([s[key] for s in samples])) for key in samples[0]}
    batch["hierarchy"] = build_hierarchy(batch["xyz"], ratios, ks=(16, 8, 12))
    return batch


def make_dales_dataloader(store, split="training", batch_size=1, num_points=65536,
                           samples_per_epoch=500, noise_init=3.5, seed=0):
    dataset = DALESSpatiallyRegularDataset(store, split, num_points, samples_per_epoch,
                                           noise_init, seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=dales_collate)
