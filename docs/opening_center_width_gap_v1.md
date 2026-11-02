# Opening Center-Width Gap Decomposition v1

Date: 2026-11-02

## Gap Decomposition

Opening F1 is evaluated as five independently owned gaps:

- Presence/center recovery.
- Interval width.
- Door/window family.
- Host-wall identity.
- Localization error.

`opening_head_scopes.py` can freeze the trunk and train only `foreground`,
`center`, `width`, `family`, `width_family`, `center_width`, `gap_fusion`, or
the four output heads. The center-width head represents each opening as a
wall-local center peak and normalized width instead of a connected foreground
component.

## Results

On full val491 with oracle target walls, the center-width head plus its
low-threshold candidate verifier reaches vector F1 `0.380565`, precision
`0.315683`, and recall `0.505308`. This replaces the dense representation's
geometry ceiling of about `4.94` matched openings with `8.63` matched openings
per case.

In the complete predicted-wall consumer, Opening F1 reaches `0.337977` versus
`0.282302` for the previous promoted opening branch. The paired gain is
`+0.055675`, 95% CI `[0.033259, 0.078565]`, rank-biserial `0.237417`, Cliff's
delta `0.162933`, and Wilcoxon `p=3.15e-6`. Room, wall, node, edge,
deterministic GED, validity, watertightness, and outside-footprint are
unchanged on all 491 cases.

## Gap Fusion

The cross-attention gap fusion has four learned tokens and channel gates.
Training only this 65,761-parameter scope for four epochs reduces raw vector F1
from `0.158657` to `0.155614`, so the partial-fusion arm is not promoted.

## Rejections

- Image-space peak decoding drops oracle F1 from `0.380565` to `0.267262`.
- Remapping decoded openings through the explicit-plan wall ordering drops
  consumer F1 from `0.337977` to `0.325020`.
- Lowering the verifier threshold to `0.55` or `0.60` reduces consumer F1.
- Full-consumer batched attachment is not yet equivalent to the verified
  per-case attachment path; keep per-case mode for reported results.
- A verifier trained on 96 predicted-host cases does not improve its 63-case
  holdout (`0.179994 -> 0.175530`, paired CI crossing zero). The remaining host
  gap must be repaired in wall geometry or a typed host graph.

## Remaining Boundary

The oracle-to-consumer gap is `0.042588`, and the consumer remains `0.012023`
below the absolute `0.35` gate. The next owned task is predicted-wall-aware
verification, not detector capacity.
