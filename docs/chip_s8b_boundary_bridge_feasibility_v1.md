# S8B Boundary-Contact Bridge Feasibility

Date: 2026-11-02

## Question

Can the DBU shape/contact representation preserve the raw grouping and boundary
violation counts for cases 57 and 231 without changing fixed shapes, moving
preplaced blocks, or exceeding the 1% area tolerance?

## Result

Yes at the shape/contact representation stage, before hard overlap projection.

The key case was block 98 in case 57: a fixed-shape, left-boundary block sat
`1.232e-6 um` inside the global x-min because moving it alone would break its
same-axis cluster contact. The bridge pass moves the boundary block to the exact
bbox edge and translates one unrestricted same-axis neighbor by the same delta.
It accepts only when grouping does not regress and boundary decreases.

| case | raw grouping / boundary | quantized grouping / boundary | area / fixed / preplaced violations |
|---|---:|---:|---:|
| 57 | 0 / 0 | 0 / 0 | 0 / 0 / 0 |
| 231 | 0 / 1 | 0 / 1 | 0 / 0 / 0 |

On the 63 raw-reference-failing val92 cases:

- grouping preserved: `55/63`;
- boundary preserved: `28/63` after the bridge, up from `9/63`;
- area violations: `0`;
- fixed-shape violations: `0`;
- preplaced violations: `0`;
- maximum soft-grid residual: below `3e-5 um`.

The shape stage intentionally permits overlaps; hard feasibility remains the
responsibility of `project_quantized_layout`. It does not yet establish raw-reference
soft preservation after the final integer hard projection.

## Artifacts

- `src/floorflow/chip_flow/dbu_shape_quantization.py`
- `scripts/audit_chip_dbu_shape_quantization.py`
- `outputs/reports/chip_s8b_shape_quantization_rawfailed_val92_v6.json`
