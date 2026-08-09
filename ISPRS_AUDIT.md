# ISPRS Stage 5A audit

## Search result and provenance

The complete repository was searched for `ISPRS`, `Area_1`, `Area_2`,
`input_0.450`, `coarse_KDTree`, `batch_limits`, `neighbors_limits`, and
`proj.pkl`, including Git history. **This checkout contains no original ISPRS
loader, preparation script, configuration, or evaluator.** The TensorFlow
sources present cover DALES, S3DIS, STPLS3D and Toronto3D. Consequently, this
adapter does not falsely attribute unverifiable behavior to a file that is not
present. Values conventionally inherited from the DR-Net input pipeline are
marked as such in `pytorch/configs/isprs.yaml`; they require confirmation
against the user's missing original ISPRS source.

## Established dataset contract

* The supplied processed convention defines Area_1 as training and Area_2 as
  validation/test. There is no random split.
* ISPRS raw labels 0 through 8 map identically to targets 0 through 8. Class 0
  is supervised, so it is never used as the DALES-style unlabeled sentinel.
* The legacy ISPRS preparation convention stores **intensity**, **return
  number**, and **number of returns** in PLY properties `red`, `green`, and
  `blue`, respectively. The adapter exposes semantic aliases and never calls
  them RGB. The faithful model feature construction remains centered XYZ
  (three channels); auxiliary attributes are loaded, checked and reported.
* The normal KDTree supplies exact subsampled points and spatial crop queries.
  The coarse KDTree is a persisted annotation/support index in the legacy
  layout. Stage 5A loads and validates it, but the fully annotated baseline
  must not substitute its coordinates for model points or rebuild it.
* `Area_2_proj.pkl` is a two-item pickle `(projection_indices,
  original_labels)`. Each original Area_2 point has one integer index into the
  subsampled Area_2 cloud; the second item contains public original-resolution
  labels (or is empty when withheld). Inference indexes subsampled predictions
  with the first item and reports metrics only when the second item is present.

Class counts, frequencies, and weights cannot be truthfully hard-coded without
the user's files. They are computed from Area_1 at startup. The weighting rule
is `sqrt(total_count) / sqrt(class_count)` and includes all nine classes.

## Non-changes

No PLY, KDTree, coarse KDTree, or projection is generated. Coordinates are not
changed on disk. DR-Net encoder, semantic query, prefix sampling, K16/K8/K12,
decoder, augmentation, and combined WCE/Lovasz semantics are shared unchanged;
only the classifier/input dimensions are parameters.
