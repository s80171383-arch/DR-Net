# Stage 2A TensorFlow/PyTorch compatibility audit

This audit is intentionally limited to `compatibility_mode=True`. It does not
introduce a real-data pipeline, ISPRS support, FPS, distance weighting, or a
"corrected" architecture.

## Initializers

| TF layer | TensorFlow initializer | Previous PyTorch initializer | Aligned? | Faithful-mode treatment / limitation |
|---|---|---|---|---|
| `fc0` (`tf.layers.dense`) | TensorFlow layers default Glorot/Xavier uniform; zero bias | `Conv1d` Kaiming uniform | No | Explicit Xavier uniform and zero bias. A 1x1 convolution has the same affine tensor semantics after layout conversion. Framework RNG streams need not produce identical samples. |
| attention-pooling `tf.layers.dense(..., use_bias=False)` | Glorot/Xavier uniform | `Linear` Kaiming uniform | No | Explicit Xavier uniform, no bias. |
| augmentation `channel_attention` dense | Glorot/Xavier uniform, no bias | Missing | No | Added as an explicit Xavier-uniform, bias-free `Linear(num_points, 1)`. |
| `tf_util.conv2d` | Calls `_variable_with_weight_decay(..., use_xavier=False)`: truncated normal with `std=sqrt(2 / kernel_shape[-1])`, then rounds to 0.001; zero bias | `Conv1d` Kaiming uniform | No | Explicit truncated-normal sampling, two-standard-deviation bounds, 0.001 rounding, and zero bias. TensorFlow and PyTorch truncated-normal algorithms/RNGs are not guaranteed sample-identical. |
| `tf_util.conv2d_transpose` | Same non-Xavier helper; its TF kernel's last dimension is input channels | `ConvTranspose2d` Kaiming uniform | No | Same explicit approximation, using input channels in the standard deviation denominator; zero bias. Same RNG limitation. |
| classifier (`tf_util.conv2d`, BN/activation disabled) | Same rounded truncated normal; zero bias | `Conv1d` Kaiming uniform | No | Uses the same faithful convolution initializer. |

`use_xavier=True` and `use_xavier=False` are therefore not conflated. No DR-Net
`tf_util.conv2d`/`conv2d_transpose` call overrides the helper's `False` default.
Exact initial tensors cannot be promised across frameworks because their random
number generators and truncated-normal implementations differ.

## Behavior checked

* Augmentation retains the original single randomly selected mirror, Y-axis
  rotation (`3.14592653`), or batch-shared jitter operation. Channel attention
  follows those operations and applies only to the generated training half.
* Channel attention transposes `[B,N,C]` to `[B,C,N]`, applies the original
  bias-free dense map over **N**, transposes `[B,C,1]` to `[B,1,C]`, softmaxes
  over **C**, and broadcasts multiplication back over `[B,N,C]`. This is not an
  SE block or conventional channel-attention redesign.
* TensorFlow LeakyReLU defaults/calls in `fc0` and the residual sum, plus the
  forced activations inside `conv2d` and `conv2d_transpose`, all use alpha 0.2.
  Every corresponding PyTorch module/functional call now states 0.2 explicitly.
* TensorFlow batch normalization updates `moving = 0.99 * moving + 0.01 * batch`.
  PyTorch defines `momentum` as the new-batch coefficient, so `momentum=0.01`
  is the matching update; `eps=1e-6` also matches.
* The K=12 neighbor input remains deliberately unused; hierarchy sampling is a
  prefix; semantic-query weights remain fixed at 1/3; decoder kernels/strides
  remain 1x1/1; training doubles features, annotation coordinates, and the
  unchanged geometry graph; `anno_xyz` is not augmented.
* Weighted cross entropy remains a mean of per-element CE multiplied by the
  selected class weight (there is no division by sum of weights). Lovasz uses
  flattened probabilities/labels and `classes='present'`; combined loss remains
  WCE + Lovasz. DALES label 0 remains ignored and labels 1--8 map to 0--7.

## Numerical-equivalence boundary

The audit establishes matching graph operations, layouts, hyperparameters, and
initializer families. It does not claim bitwise numerical equivalence: random
augmentation draws, initializer samples, KNN tie ordering, floating-point
kernels, and BatchNorm reduction details can differ between TensorFlow and
PyTorch even when their documented semantics agree.
