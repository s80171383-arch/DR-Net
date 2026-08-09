"""Run one reproducible real-DALES batch through the faithful DR-Net pipeline."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch

from .datasets.dales import DALESCloudStore, make_dales_dataloader
from .losses import combined_loss
from .models import DRNet

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "dales.yaml"


def _load_config(path):
    """Read the deliberately simple Stage 2 configuration without a runtime dependency."""
    result, parents = {}, [(0, result)]
    for raw_line in Path(path).read_text(encoding="utf8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, raw_value = line.strip().split(":", 1)
        while parents[-1][0] > indent:
            parents.pop()
        target = parents[-1][1]
        value = raw_value.strip()
        if not value:
            target[key] = {}
            parents.append((indent + 1, target[key]))
            continue
        normalized = {"true": "True", "false": "False", "null": "None"}.get(value, value)
        try:
            target[key] = ast.literal_eval(normalized)
        except (ValueError, SyntaxError):
            target[key] = value
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="DALES_ROOT (never inferred)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, help="diagnostic override of config batch_size")
    parser.add_argument("--num-points", type=int, help="diagnostic override of config num_points")
    parser.add_argument("--data-only", action="store_true")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _finite(name, value):
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def validate_batch(batch, store, dataset, *, model=None, device="cpu", learning_rate=1e-2,
                   class_weights=None):
    """Validate data invariants and optionally perform train/backward/step/eval."""
    xyz, raw, point_indices = batch["xyz"], batch["raw_labels"], batch["point_indices"]
    anno, targets = batch["xyz_with_anno"], batch["labels_with_anno"]
    if len(batch["hierarchy"]) != 5:
        raise AssertionError("hierarchy must contain exactly 5 levels")
    if raw.numel() == 0 or raw.min() < 0 or raw.max() > 8:
        raise AssertionError("raw DALES labels must be in [0, 8]; raw 0 is the ignore label")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() > 7:
        raise AssertionError("classifier targets must be in [0, 7]")
    for name, value in batch.items():
        if torch.is_tensor(value):
            _finite(name, value)
    for level_number, level in enumerate(batch["hierarchy"]):
        _finite(f"hierarchy[{level_number}].xyz", level["xyz"])
        support_count = level["xyz"].shape[1]
        for key in ("neigh_idx", "neigh_idx_1", "neigh_idx_2", "sub_idx"):
            index = level[key]
            if index.numel() and (index.min() < 0 or index.max() >= support_count):
                raise AssertionError(f"hierarchy[{level_number}].{key} index out of range")
    for row, cloud_idx in zip(point_indices, batch["cloud_index"]):
        count = len(store.input_trees[dataset.split][int(cloud_idx)].data)
        if row.numel() and (row.min() < 0 or row.max() >= count):
            raise AssertionError("point_indices index out of range")

    print(f"split: {dataset.split}")
    print(f"cloud_name: {batch['cloud_name']}")
    counts = [len(store.input_trees[dataset.split][int(i)].data) for i in batch["cloud_index"]]
    print(f"cloud point count: {counts}")
    print(f"xyz shape / dtype: {tuple(xyz.shape)} / {xyz.dtype}")
    print(f"raw_labels shape / range: {tuple(raw.shape)} / [{int(raw.min())}, {int(raw.max())}]")
    print(f"point_indices shape / range: {tuple(point_indices.shape)} / "
          f"[{int(point_indices.min())}, {int(point_indices.max())}]")
    print(f"xyz_with_anno shape: {tuple(anno.shape)}")
    print(f"labels_with_anno shape / range: {tuple(targets.shape)} / "
          f"[{int(targets.min())}, {int(targets.max())}]")
    for number, level in enumerate(batch["hierarchy"]):
        shapes = ", ".join(f"{key}={tuple(level['sub_idx' if key == 'pool_idx' else key].shape)}"
                           for key in ("xyz", "neigh_idx", "neigh_idx_1", "neigh_idx_2", "pool_idx"))
        print(f"hierarchy[{number}]: {shapes}")
    stats = dataset.last_query_stats or {}
    print("min_possibility query change: "
          f"{stats.get('min_possibility_before')} -> {stats.get('min_possibility_after')}")
    print(f"replacement upsampling: {stats.get('replacement_upsampling')}")
    if model is None:
        print("data-only validation: PASS")
        return {"data_only": True}

    device = torch.device(device)
    batch_on_device = lambda value: value.to(device) if torch.is_tensor(value) else value
    xyz, anno, targets = map(batch_on_device, (xyz, anno, targets))
    hierarchy = [{key: batch_on_device(value) for key, value in level.items()}
                 for level in batch["hierarchy"]]
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    weights = (torch.ones(8) if class_weights is None else class_weights).to(device)
    optimizer.zero_grad()
    logits = model(xyz, anno, hierarchy, {"option": 2})
    expected_train = (2 * xyz.shape[0], anno.shape[1], 8)
    if tuple(logits.shape) != expected_train:
        raise AssertionError(f"training logits must have doubled-batch shape {expected_train}")
    doubled_targets = targets.repeat(2, 1)
    loss, parts = combined_loss(logits, doubled_targets, weights)
    forward_finite = bool(torch.isfinite(logits).all())
    loss_finite = bool(torch.isfinite(loss) and all(torch.isfinite(x) for x in parts.values()))
    if not forward_finite or not loss_finite:
        raise AssertionError("forward or loss contains NaN or Inf")
    loss.backward()
    required = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    gradients_exist = bool(required) and all(gradient is not None for gradient in required)
    gradients_finite = gradients_exist and all(torch.isfinite(gradient).all() for gradient in required)
    if not gradients_exist:
        raise AssertionError("not all required gradients exist")
    if not gradients_finite:
        raise AssertionError("gradients contain NaN or Inf")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        inference = model(xyz, anno, hierarchy)
    expected_eval = (xyz.shape[0], anno.shape[1], 8)
    if tuple(inference.shape) != expected_eval:
        raise AssertionError(f"inference logits must have shape {expected_eval}")
    joined = torch.cat((logits.detach().flatten(), inference.flatten()))
    nan_count, inf_count = int(torch.isnan(joined).sum()), int(torch.isinf(joined).sum())
    if nan_count or inf_count:
        raise AssertionError("eval logits contain NaN or Inf")
    report = {"training logits shape": tuple(logits.shape),
              "inference logits shape": tuple(inference.shape),
              "WCE": float(parts["weighted_ce"]), "Lovasz": float(parts["lovasz"]),
              "total loss": float(loss), "forward finite": forward_finite,
              "loss finite": loss_finite, "all required gradients exist": gradients_exist,
              "gradients finite": gradients_finite, "optimizer.step success": True,
              "eval logits finite": True, "NaN count": nan_count, "Inf count": inf_count}
    for key, value in report.items():
        print(f"{key}: {value}")
    return report


def run(args):
    config = _load_config(args.config)
    if config.get("compatibility_mode") is not True:
        raise ValueError("Stage 4 requires compatibility_mode=True")
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    num_points = args.num_points if args.num_points is not None else config["num_points"]
    if batch_size < 1 or num_points < 1:
        raise ValueError("batch_size and num_points must be positive")
    for name, override, value in (("batch-size", args.batch_size, batch_size),
                                  ("num-points", args.num_points, num_points)):
        if override is not None:
            print(f"DIAGNOSTIC OVERRIDE --{name}={value}; not equivalent to formal DALES config")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    store = DALESCloudStore(args.data_root, config["data"]["sub_grid_size"],
                            labeled_point=str(config.get("labeled_point", "100%")), seed=0)
    loader = make_dales_dataloader(store, args.split, batch_size, num_points,
                                   samples_per_epoch=batch_size,
                                   noise_init=config.get("noise_init", 3.5), seed=0)
    batch = next(iter(loader))
    model = None if args.data_only else DRNet(
        num_classes=config["num_classes"], d_out=config["d_out"],
        compatibility_mode=True, num_points=num_points)
    frequencies = torch.as_tensor(store.num_per_class, dtype=torch.float32)
    if not args.data_only and (frequencies <= 0).any():
        raise ValueError("training data must contain every DALES class to compute WCE weights")
    weights = frequencies.sum().sqrt() / frequencies.sqrt() if not args.data_only else None
    try:
        return validate_batch(batch, store, loader.dataset, model=model, device=args.device,
                              learning_rate=config.get("learning_rate", 1e-2), class_weights=weights)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            raise RuntimeError(f"OOM during Stage 4 validation; device={args.device}, "
                               f"batch_size={batch_size}, num_points={num_points}, "
                               "failure location=data/model validation. Use diagnostic overrides only.") from error
        raise


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
