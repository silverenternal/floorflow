# Wall Tail Trim v1

Date: 2026-11-01

## Motivation

Canonical false-positive decomposition attributes most remaining wall errors to
GT-wall extension. The readout previously thresholded wall probability at `0.5`
and retained all weaker tails.

## Implementation

`StructureReadoutPolicy` now supports an opt-in wall-core gate:

- wall pixels require the normal `wall_threshold`;
- a higher `wall_core_threshold` defines the confident wall core;
- `wall_tail_radius` keeps a fixed tolerance around that core;
- junction support and room seams remain independent evidence.

The default is disabled (`wall_core_threshold=0`), preserving canonical labels.

## Zero-Training Screen

The same structure checkpoint and calibration64 cases were evaluated with the
canonical `wall_gated 0.2` policy.

| Arm | Wall F1 | Room IoU | Count MAE | Node F1 | Edge | GED |
|---|---:|---:|---:|---:|---:|---:|
| canonical | .52113 | .33256 | 2.609 | .7413 | .3698 | 10.375 |
| core .75, radius 1 | .51653 | .32898 | 2.547 | .7523 | .3726 | 9.344 |
| core .60, radius 1 | .52202 | .32871 | 2.531 | .7523 | .3726 | 9.406 |

Both thresholds improve graph/GED signals but regress Wall F1 and Room IoU.
The `0.60` arm is effectively a wall-F1 tie, not a gain. The policy gate fails,
so tail trimming is rejected as a canonical default and no further threshold
sweep is justified.

Evidence:

- `outputs/reports/wall_tail_trim_core060_cal64_v1.json`
- `outputs/reports/wall_tail_trim_core075_cal64_v1.json`

