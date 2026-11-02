# Raster-Conditioned Room Control Screen v1

## Contract

This branch is evaluated as `raster_conditioned_reconstruction`, not
`contour-only` generation.

The input has three channels:

1. walls, doors, windows, and structure;
2. fixed furniture;
3. footprint boundary and outside region.

Room-class maps and room-instance maps are explicitly excluded. The renderer
audit is `src/floorflow/data/raster_control.py` and the invariance test changes
all room identities while requiring the generated input to remain identical.

## Training

The existing ResNet18-FPN room segmenter was initialized from
`room_segmenter_r18_uniform_full40_v2` and fine-tuned on train1024 with dense
room CE, Dice, boundary weighting, and class balancing. The screen used 5
epochs and selected on val191 only.

## Results

The 16-case smoke and full val491 screens both identified a sharp improvement.
The full train5126 model reached the following val491 overlay results with
the clean, non-room structural raster:

| Metric | Contour-only v11 | Raster control | Delta |
|---|---:|---:|---:|
| Room macro IoU | 0.16984 | **0.39717** | +0.22733 |
| Instance geometric IoU | 0.51443 | **0.68447** | +0.17003 |
| Instance F1 | 0.17874 | **0.38629** | +0.20755 |
| Instance type accuracy | 0.45485 | **0.67514** | +0.22029 |
| Room support IoU | 0.75326 | 0.75326 | 0.00000 |
| Validity | 1.0 | 1.0 | 0.00000 |
| Conditional watertightness | 0.90631 | **0.98086** | +0.07455 |

Wall F1 remains `0.43552` and opening F1 remains `0.11275`, showing that the
gain comes from room identity rather than changing the DMD structural decoder.
The full-CAD report is
`outputs/reports/raster_room_control_full_model_val491_fullmetrics_v1.json`.

The native CubiCasa PNG protocol is harder. With the same ResNet18-FPN and
full train5126 split, native-image val491 reaches:

| Metric | Native CubiCasa image |
|---|---:|
| Room macro IoU | 0.27575 |
| Instance geometric IoU | 0.58656 |
| Instance F1 | 0.24451 |
| Instance type accuracy | 0.59390 |
| Validity | 1.00000 |
| Conditional watertightness | 0.98086 |

This is a real target-free reconstruction result, but the clean structural
render remains stronger. The difference is an input-domain gap, not a
target-control leak.

## Decision

RC-01 passes its control screen. The result is not yet a SOTA claim:

- training used 1,024 cases rather than the full 5,126;
- only val491 was evaluated;
- the branch is not yet five-seed;
- wall/opening/furniture still need a stronger control bank;
- test503 remains closed.

The next task, RC-02, scales this validated branch to the full training split
and trains the complete control bank.
