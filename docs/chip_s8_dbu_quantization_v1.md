# S8 DBU-Native Representation v1

Date: 2026-11-01

## Scope

`quantize_boxes_to_dbu` converts continuous chip boxes into an integer-DBU
representation without claiming to solve overlap, boundary or packing.

- soft block centers and dimensions lie on the declared grid;
- area closure is checked against the official tolerance;
- fixed-shape and preplaced inputs are preserved;
- MIB members receive one shared quantized shape.

## Contract Result

Focused tests cover grid alignment, area error, fixed/preplaced preservation,
MIB sharing, permutation equivariance and invalid resolution.

On the frozen 92-case development subset:

- maximum soft-area relative error: `0.1228%`, within the `1%` tolerance;
- position-grid residual: `0`;
- maximum soft-shape grid residual: `3.81e-6 um`;
- area failing blocks: `0`.

## What This Does Not Solve

Changing continuous shapes to DBU sizes invalidates the old legal layout:

- `0/92` cases remain hard-feasible after shape snapping;
- `3,770` overlap pairs remain;
- grouping and boundary regress on all `92` cases.

This is expected: DBU shape quantization and DBU legalization are separate
layers. The next S8 step is a projector/packer that operates on integer shapes
and positions, rather than rounding an already-legal Euclidean layout.

## Artifacts

- `src/floorflow/chip_flow/dbu_quantization.py`
- `tests/test_chip_dbu_quantization.py`
- `outputs/reports/chip_s8_dbu_endpoint_probe_val92_v1.json`

