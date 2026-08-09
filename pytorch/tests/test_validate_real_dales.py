import pickle

import numpy as np
import pytest
import torch

from helper_ply import write_ply
from pytorch.datasets.dales import DALESCloudStore, make_dales_dataloader
from pytorch.models import DRNet
from pytorch.tests.test_dales_pipeline import FixtureKDTree
from pytorch.validate_real_dales import parse_args, main, validate_batch


def _fixture(root, count=64):
    for split in ("train", "test"):
        original = root / "original_ply" / split
        sampled = root / "input_0.320" / split
        original.mkdir(parents=True)
        sampled.mkdir(parents=True)
        points = np.random.default_rng(2).normal(size=(count, 3)).astype("f4")
        labels = (np.arange(count) % 8 + 1).astype("i4")
        write_ply(str(original / "cloud.ply"), [points, labels], ["x", "y", "z", "class"])
        write_ply(str(sampled / "cloud.ply"), [points, labels], ["x", "y", "z", "class"])
        with (sampled / "cloud_KDTree.pkl").open("wb") as stream:
            pickle.dump(FixtureKDTree(points), stream)
        if split == "test":
            with (sampled / "cloud_proj.pkl").open("wb") as stream:
                pickle.dump((np.arange(count), labels), stream)


def _config(path):
    path.write_text("""compatibility_mode: true
batch_size: 1
num_points: 64
noise_init: 0
learning_rate: 0.001
labeled_point: "100%"
data:
  sub_grid_size: 0.32
num_classes: 8
d_out: [16, 64, 128, 256, 512]
""")


def test_cli_argument_parsing():
    args = parse_args(["--data-root", "/data", "--device", "cuda", "--batch-size", "2",
                       "--num-points", "128", "--data-only"])
    assert (args.data_root, args.device, args.batch_size, args.num_points, args.data_only) == (
        "/data", "cuda", 2, 128, True)


def test_data_only_path(tmp_path, capsys):
    _fixture(tmp_path)
    config = tmp_path / "config.yaml"
    _config(config)
    result = main(["--data-root", str(tmp_path), "--config", str(config), "--data-only"])
    assert result == {"data_only": True}
    assert "data-only validation: PASS" in capsys.readouterr().out


def test_single_batch_helper_and_invalid_failures(tmp_path):
    _fixture(tmp_path)
    store = DALESCloudStore(tmp_path)
    loader = make_dales_dataloader(store, "training", 1, 64, 1, noise_init=0)
    batch = next(iter(loader))
    assert validate_batch(batch, store, loader.dataset) == {"data_only": True}
    report = validate_batch(batch, store, loader.dataset, model=DRNet(num_points=64),
                            class_weights=torch.ones(8))
    assert report["training logits shape"] == (2, 64, 8)
    assert report["inference logits shape"] == (1, 64, 8)
    assert report["optimizer.step success"] is True

    bad_label = {**batch, "raw_labels": batch["raw_labels"].clone()}
    bad_label["raw_labels"][0, 0] = 9
    with pytest.raises(AssertionError, match="raw DALES labels"):
        validate_batch(bad_label, store, loader.dataset)
    bad_index = {**batch, "point_indices": batch["point_indices"].clone()}
    bad_index["point_indices"][0, 0] = 64
    with pytest.raises(AssertionError, match="point_indices"):
        validate_batch(bad_index, store, loader.dataset)
    bad_float = {**batch, "xyz": batch["xyz"].clone()}
    bad_float["xyz"][0, 0, 0] = torch.nan
    with pytest.raises(AssertionError, match="NaN or Inf"):
        validate_batch(bad_float, store, loader.dataset)
