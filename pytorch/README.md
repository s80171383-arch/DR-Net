# Faithful PyTorch DR-Net (DALES only)

This directory implements the original TensorFlow graph's observed behavior. It is
not a redesign: prefix sampling, the unused K=12 branch, fixed-mean Semantic Query,
non-upsampling 1x1 transposed convolutions, geometry-copying augmentation, DALES
label mapping, and original loss reductions are deliberately preserved.

Run `pytest -q` and `python train.py` from this directory. Real-data loading, full
training, TensorFlow checkpoint conversion, and ISPRS support are out of scope.

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
