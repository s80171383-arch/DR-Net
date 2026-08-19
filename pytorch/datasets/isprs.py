"""Reader and faithful spatial sampler for the preprocessed ISPRS release layout.

The legacy preprocessing writes intensity, return number and number of returns
under the generic PLY names ``red``, ``green`` and ``blue`` respectively.  They
are deliberately never called RGB in this module.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from helper_ply import read_ply
from ..utils.knn import build_hierarchy

NUM_CLASSES = 9
SPLITS = {"training": "Area_1", "validation": "Area_2"}
PLY_FEATURE_NAMES = ("intensity", "return_number", "number_of_returns")


def map_isprs_labels(raw):
    labels = torch.as_tensor(raw).long()
    if labels.numel() and ((labels < 0) | (labels >= NUM_CLASSES)).any():
        raise ValueError("ISPRS labels must be integers in [0, 8]")
    return labels


class ISPRSCloudStore:
    """Strict loader for existing files; it never creates preprocessing assets."""
    def __init__(self, data_root, sub_grid_size=.45, labeled_point="100%", seed=0):
        self.root = Path(data_root)
        self.folder = self.root / f"input_{sub_grid_size:.3f}"
        self.input_trees: Dict[str, List] = {x: [] for x in SPLITS}
        self.coarse_trees: Dict[str, List] = {x: [] for x in SPLITS}
        self.input_labels: Dict[str, List[np.ndarray]] = {x: [] for x in SPLITS}
        self.input_features: Dict[str, List[np.ndarray]] = {x: [] for x in SPLITS}
        self.input_names: Dict[str, List[str]] = {x: [] for x in SPLITS}
        self.annotation_masks: Dict[str, List[np.ndarray]] = {x: [] for x in SPLITS}
        self.val_proj, self.val_labels = [], []
        self.num_per_class = np.zeros(NUM_CLASSES, np.int64)
        self.labeled_point = str(labeled_point)
        allowed = {"100%": 1.0, "10%": .1, "1%": .01, "0.1%": .001}
        if self.labeled_point not in allowed:
            raise ValueError(f"labeled_point must be one of {sorted(allowed)}, got {self.labeled_point!r}")
        self.labeled_ratio = allowed[self.labeled_point]
        self.seed = int(seed)
        self._load()
        self._initialize_annotation_masks()

    @staticmethod
    def _tree_points(tree, path):
        points = np.asarray(getattr(tree, "data", None))
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{path}: KDTree.data must be finite [N,3]")
        return points

    def _load(self):
        if not self.folder.is_dir():
            raise FileNotFoundError(f"required processed directory is missing: {self.folder}")
        required = {"x", "y", "z", "red", "green", "blue", "class"}
        for split, name in SPLITS.items():
            paths = [self.folder / f"{name}.ply", self.folder / f"{name}_KDTree.pkl",
                     self.folder / f"{name}_coarse_KDTree.pkl"]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"required ISPRS file is missing: {path}")
            data = read_ply(str(paths[0]))
            fields = set(data.dtype.names or ())
            if not required <= fields:
                raise ValueError(f"{paths[0]}: missing PLY fields {sorted(required-fields)}")
            xyz = np.column_stack([data[x] for x in ("x", "y", "z")]).astype(np.float32)
            # Semantic aliases established by the legacy ISPRS preparation path.
            aux = np.column_stack([data[x] for x in ("red", "green", "blue")]).astype(np.float32)
            raw = np.asarray(data["class"])
            if not np.isfinite(xyz).all() or not np.isfinite(aux).all():
                raise ValueError(f"{paths[0]}: coordinates/features contain NaN or Inf")
            if raw.dtype.kind not in "iu" or raw.size == 0 or raw.min() < 0 or raw.max() > 8:
                raise ValueError(f"{paths[0]}: class must be integer [0,8]")
            with paths[1].open("rb") as stream: tree = pickle.load(stream)
            with paths[2].open("rb") as stream: coarse = pickle.load(stream)
            points = self._tree_points(tree, paths[1])
            self._tree_points(coarse, paths[2])
            if len(points) != len(xyz) or not np.allclose(points, xyz):
                raise ValueError(f"{paths[1]}: KDTree.data does not agree with PLY XYZ")
            labels = raw.astype(np.int64, copy=True)
            if split == "training": self.num_per_class += np.bincount(labels, minlength=9)
            self.input_trees[split].append(tree); self.coarse_trees[split].append(coarse)
            self.input_labels[split].append(labels); self.input_features[split].append(aux)
            self.input_names[split].append(name)
        projection = self.folder / "Area_2_proj.pkl"
        if not projection.is_file(): raise FileNotFoundError(f"required ISPRS file is missing: {projection}")
        with projection.open("rb") as stream: obj = pickle.load(stream)
        if not isinstance(obj, (tuple, list)) or len(obj) != 2:
            raise ValueError(f"{projection}: expected (projection_indices, original_labels)")
        proj, labels = map(np.asarray, obj)
        n = len(self.input_labels["validation"][0])
        if proj.ndim != 1 or (proj.size and (proj.min() < 0 or proj.max() >= n)):
            raise ValueError(f"{projection}: invalid projection indices for {n} subsampled points")
        if labels.ndim != 1 or len(labels) not in (0, len(proj)):
            raise ValueError(f"{projection}: original labels must be empty or match projection length")
        if labels.size and (labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() > 8):
            raise ValueError(f"{projection}: original labels must be integer [0,8]")
        self.val_proj.append(proj.astype(np.int64)); self.val_labels.append(labels.astype(np.int64))

    def _initialize_annotation_masks(self):
        """Draw one global, unstratified Area_1 mask; Area_2 always stays dense."""
        sizes = [len(labels) for labels in self.input_labels["training"]]
        total = sum(sizes)
        keep = max(int(total * self.labeled_ratio), 1)
        if keep > total:
            raise ValueError("Area_1 must contain at least one point")
        selected = np.random.default_rng(self.seed).choice(total, keep, replace=False)
        global_mask = np.zeros(total, dtype=bool)
        global_mask[selected] = True
        offsets = np.cumsum([0] + sizes)
        self.annotation_masks["training"] = [global_mask[offsets[i]:offsets[i + 1]].copy()
                                             for i in range(len(sizes))]
        self.annotation_masks["validation"] = [np.ones(len(labels), dtype=bool)
                                               for labels in self.input_labels["validation"]]

    def annotation_statistics(self):
        """Return global training annotation counts without changing the fixed mask."""
        labels = np.concatenate(self.input_labels["training"])
        mask = np.concatenate(self.annotation_masks["training"])
        return {"total": int(labels.size), "requested_ratio": self.labeled_ratio,
                "annotated": int(mask.sum()),
                "actual_ratio": float(mask.mean()),
                "per_class": np.bincount(labels[mask], minlength=NUM_CLASSES)}

    def class_statistics(self):
        total = self.num_per_class.sum()
        freq = self.num_per_class / total
        if total == 0 or (self.num_per_class == 0).any():
            raise ValueError("Area_1 must contain all nine classes for WCE")
        return self.num_per_class.copy(), freq, np.sqrt(total) / np.sqrt(self.num_per_class)


class ISPRSSpatiallyRegularDataset(IterableDataset):
    def __init__(self, store, split="training", num_points=65536, samples_per_epoch=500,
                 noise_init=3.5, seed=0, feature_mode="xyz"):
        if split not in SPLITS: raise ValueError("split must be training or validation")
        if feature_mode not in ("xyz", "xyz_aux"): raise ValueError("feature_mode must be xyz or xyz_aux")
        self.store, self.split, self.num_points = store, split, num_points
        self.samples_per_epoch, self.noise_init, self.seed = samples_per_epoch, noise_init, seed
        self.feature_mode, self.possibility, self.min_possibility = feature_mode, [], []
        self.last_query_stats = None

    def __iter__(self):
        if get_worker_info() is not None: raise RuntimeError("possibility sampling requires num_workers=0")
        rng = np.random.default_rng(self.seed)
        self.possibility = [rng.random(len(x))*1e-3 for x in self.store.input_labels[self.split]]
        self.min_possibility = [float(x.min()) for x in self.possibility]
        emitted = 0
        while emitted < self.samples_per_epoch:
            ci = int(np.argmin(self.min_possibility)); before = self.min_possibility[ci]
            pi = int(np.argmin(self.possibility[ci])); tree = self.store.input_trees[self.split][ci]
            points = np.asarray(tree.data); center = points[pi:pi+1]
            pick = center + rng.normal(scale=self.noise_init/10, size=center.shape).astype(center.dtype)
            count = min(len(points), self.num_points)
            idx = rng.permutation(np.asarray(tree.query(pick, k=count)[1][0], np.int64))
            xyz = points[idx] - pick; labels = self.store.input_labels[self.split][ci][idx]
            dist = np.square(points[idx]-pick).sum(1); maximum = dist.max()
            np.add.at(self.possibility[ci], idx, np.square(1-dist/maximum) if maximum else np.ones_like(dist))
            self.min_possibility[ci] = float(self.possibility[ci].min())
            if count < self.num_points:
                dup = rng.choice(count, self.num_points-count, replace=True)
                xyz=np.concatenate([xyz,xyz[dup]]); labels=np.concatenate([labels,labels[dup]]); idx=np.concatenate([idx,idx[dup]])
            block_annotated = np.flatnonzero(self.store.annotation_masks[self.split][ci][idx])
            target = self.num_points if self.split == "validation" else max(int(self.num_points * self.store.labeled_ratio), 1)
            if block_annotated.size == 0:
                # Advance spatial coverage but do not invent an annotation.
                continue
            if block_annotated.size > target:
                annotated = rng.choice(block_annotated, target, replace=False)
            elif block_annotated.size < target:
                extra = rng.choice(block_annotated, target - block_annotated.size, replace=True)
                annotated = np.concatenate((block_annotated, extra))
                rng.shuffle(annotated)
            else:
                annotated = block_annotated
            aux = self.store.input_features[self.split][ci][idx]
            features = xyz if self.feature_mode == "xyz" else np.concatenate([xyz, aux], 1)
            self.last_query_stats={"min_possibility_before":before,"min_possibility_after":self.min_possibility[ci],
                                   "replacement_upsampling":count<self.num_points,
                                   "available_annotations":int(block_annotated.size), "annotation_target":target}
            emitted += 1
            yield {"xyz":torch.from_numpy(xyz.astype(np.float32)), "features":torch.from_numpy(features.astype(np.float32)),
                   "aux_features":torch.from_numpy(aux.astype(np.float32)), "raw_labels":torch.from_numpy(labels),
                   "point_indices":torch.from_numpy(idx), "cloud_index":torch.tensor(ci),
                   "xyz_with_anno":torch.from_numpy(xyz[annotated].astype(np.float32)),
                   "labels_with_anno":map_isprs_labels(labels[annotated]), "cloud_name":self.store.input_names[self.split][ci]}


def isprs_collate(samples, ratios=(4,4,4,4,2), knn_backend="ckdtree"):
    batch={k:([s[k] for s in samples] if k=="cloud_name" else torch.stack([s[k] for s in samples])) for k in samples[0]}
    batch["hierarchy"]=build_hierarchy(batch["xyz"], ratios, ks=(16,8,12), backend=knn_backend); return batch


def make_isprs_dataloader(store, split="training", batch_size=1, num_points=65536,
                          samples_per_epoch=500, noise_init=3.5, seed=0, feature_mode="xyz",
                          knn_backend="ckdtree"):
    ds=ISPRSSpatiallyRegularDataset(store,split,num_points,samples_per_epoch,noise_init,seed,feature_mode)
    collate = lambda samples: isprs_collate(samples, knn_backend=knn_backend)
    return DataLoader(ds,batch_size=batch_size,num_workers=0,collate_fn=collate)
