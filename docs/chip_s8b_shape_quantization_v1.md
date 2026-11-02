# S8B Adjacency-Aware Shape Quantization v1

Date: 2026-11-02

## Purpose

The original DBU quantizer chose each soft width/height independently by area
closure and aspect proximity. That preserved area but changed raw grouping and
boundary topology.

`quantize_boxes_adjacency_aware` adds a representation-only postprocess:

1. Reuse `quantize_boxes_to_dbu` for integer shapes and fixed/preplaced anchors.
2. Build the raw cluster contact tree and project quantized centers through it.
3. For movable boundary-coded soft blocks, adjust width/height only when the
   official 1% area tolerance remains satisfied.
4. For fixed-shape boundary blocks, move position while preserving exact shape.
5. Translate contact components with boundary-derived integer shifts.
6. Audit on the original instance; preplaced blocks are never modified.

The postprocess may leave overlaps because hard legalization belongs to
`project_quantized_layout`, not this shape/contact representation layer.

## Frozen Probe

Reports:

- two-case probe: `outputs/reports/chip_s8b_shape_quantization_probe_v1.json`;
- 63 raw-reference failures from the frozen val92 screen:
  `outputs/reports/chip_s8b_shape_quantization_rawfailed_val92_v6.json`.

| case | raw grouping | quantized grouping | raw boundary | quantized boundary |
|---|---:|---:|---:|---:|
| 57 | 0 | 0 | 0 | 1 |
| 231 | 0 | 0 | 1 | 2 |

Across both cases:

- grouping preserved: `2/2`;
- area violations: `0`;
- fixed-shape violations: `0`;
- preplaced violations: `0`;
- maximum soft-grid residual: below `2.6e-5 um`.

The earlier independent quantizer reached grouping `5` on case 57 and
grouping/ boundary much farther from raw. Contact-tree recovery is therefore
effective, but one boundary unit remains in each probe case.

On the 63 raw-reference failures:

- grouping is preserved on `55/63`;
- boundary is preserved on `28/63` after the boundary-contact bridge;
- area, fixed-shape, and preplaced violations are all zero;
- maximum soft-grid residual is below `3.0e-5 um`.

## Remaining Gap

The residual is concentrated in fixed-shape, boundary-coded blocks whose source
edge does not land exactly on the declared grid. Moving them to the exact bbox
edge breaks a cluster contact; changing a neighboring soft shape is constrained
by the 1% area tolerance. The current implementation records this as a
`partial_raw_boundary_gap` rather than relaxing DBU or area constraints.

## Full-Pipeline Audit

`outputs/reports/chip_s8b_shape_projection_val92_v1.json` couples this
representation stage to the raw-contact-aware integer projection. It remains
below promotion:

- quantized-continuous primary gate: `66/92`;
- unquantized raw-reference soft gate: `40/92`.

The bridge improves the intermediate representation but hard projection changes
the contact geometry. The standard shape quantizer remains the deployed DBU
endpoint until an atom- or generator-level contact conditioner closes this gap.

## Artifacts

- `src/floorflow/chip_flow/dbu_shape_quantization.py`
- `tests/test_chip_dbu_shape_quantization.py`
- `scripts/audit_chip_dbu_shape_quantization.py`
- `outputs/reports/chip_s8b_shape_quantization_rawfailed_val92_v6.json`
- `outputs/reports/chip_s8b_shape_projection_val92_v1.json`
