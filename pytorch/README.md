# Faithful PyTorch DR-Net (DALES only)

This directory implements the original TensorFlow graph's observed behavior. It is
not a redesign: prefix sampling, the unused K=12 branch, fixed-mean Semantic Query,
non-upsampling 1x1 transposed convolutions, geometry-copying augmentation, DALES
label mapping, and original loss reductions are deliberately preserved.

From the repository root, run `python -m pytest -q pytorch/tests` and
`python -m pytorch.train`.

Stage 3 adds the DALES real-data loading, spatially regular sampling,
sparse annotation, hierarchy construction, and dataset verification pipeline.
Full DALES training, TensorFlow checkpoint conversion, and ISPRS support
remain out of scope at this stage.

## Reference mapping

| TensorFlow `DR-Net.py` | PyTorch |
|---|---|
| `relative_pos_encoding`, `gather_neighbour` | `models/blocks.py` |
| `att_pooling`, `building_block` | `AttentionPooling`, `BuildingBlock` |
| `dilated_res_block`, `random_sample` | `DilatedResidualBlock`, `random_sample` |
| `three_interpolate(..., weight=1/3)` | `utils/interpolation.py:semantic_query` |
| decoder and classifier | `models/drnet.py:DRNet` |
| augmentation / graph duplication | `utils/augmentation.py` |
| WCE + flattened Lovasz | `losses/` |

The public layout is `[B,N,C]`; pointwise neural features use `[B,C,N]`. Thus the
reference `[B,N,1,C]` singleton spatial dimension is omitted, except around the
decoder's `ConvTranspose2d`, where it is restored explicitly.

`compatibility_mode: false` is reserved for a future corrected implementation and
currently raises `NotImplementedError`, preventing accidental algorithm changes.
## Stage 3: DALES data pipeline

The real-data loader expects the preprocessing layout produced by the original
TensorFlow project (data itself must remain outside this repository):

```text
DALES_ROOT/
  original_ply/{train,test}/*.ply
  input_0.320/train/{cloud.ply,cloud_KDTree.pkl}
  input_0.320/test/{cloud.ply,cloud_KDTree.pkl,cloud_proj.pkl}
```

Set `data.root` in `configs/dales.yaml`, or validate a location explicitly:

```bash
python -m pytorch.datasets.verify_dales --data-root /path/to/dales
```

`DALESCloudStore` loads both author-defined splits without repartitioning.
`DALESSpatiallyRegularDataset` is an `IterableDataset`, preserving the mutable
minimum-possibility query state. `make_dales_dataloader` returns dictionaries
containing `xyz [B,N,3]`, raw labels and source indices `[B,N]`, cloud index/name,
annotated XYZ `[B,M,3]`, mapped annotated targets `[B,M]`, and the established
five-level prefix-sampled K16/K8/K12 hierarchy. Use `num_workers=0`; separate
worker processes would incorrectly create independent possibility states.

## Stage 4: Real DALES validation

Stage 4 provides a deliberately single-batch, reproducible validation runner. It
reuses `DALESCloudStore`, spatially regular sampling, the five-level hierarchy,
the compatibility-mode DR-Net, augmentation, and WCE + Lovasz loss. It does not
start epoch training or write a checkpoint. Run the data layer first from the
repository root:

```bash
python -m pytorch.validate_real_dales --data-root /path/to/DALES_ROOT --data-only
```

Then validate one train, backward, optimizer, and eval step:

```bash
python -m pytorch.validate_real_dales --data-root /path/to/DALES_ROOT --device cuda
```

`--config` defaults to `pytorch/configs/dales.yaml`; `--split` accepts `training`
or `validation`. `--batch-size` and `--num-points` are diagnostic overrides only,
and the runner labels them as such. They are useful for identifying OOM limits
but are not equivalent to the formal DALES configuration. DataLoader workers
remain fixed at zero because possibility sampling is stateful.
