# S8 Integer-DBU Projection v2

Date: 2026-11-02

## Scope

`project_quantized_layout` treats DBU materialization as a representation step:

1. Quantized shapes define a packing-only `FloorplanInstance`.
2. Existing Trinity `scale_pack(reshape=False)` produces a continuous legal layout.
3. SCIP projects positions onto the declared `0.001 um` grid while preserving
   nearby continuous pair relations.
4. Contact components receive exact floating-point edge assignments; this avoids
   the `1e-15` mismatch that makes shapely split touching rectangles.
5. Every result is audited on the original instance. Area, fixed/preplaced,
   overlap, grouping and boundary failures are fail-closed.

The implementation reuses `dbu_quantization`, Trinity `scale_pack`, Trinity
feasibility/soft checks, and the project's existing report-backed evaluation data.

## Five-Case Gate

Report: `outputs/reports/chip_s8_dbu_packing_gate_5case_v4.json`

- Primary hard gate: `5/5`.
- Zero overlap, area, fixed and preplaced violations: `5/5`.
- Grouping and boundary do not regress against the quantized continuous legal
  reference: `5/5`.
- Maximum projected grid residual: below `4e-6 um`.
- Total CPU projection time: `2.47 s`.

Against the unquantized report reference, only `3/5` pass. Cases 57 and 231 lose
one grouping or boundary unit during shape quantization. This residual is recorded
separately and is not hidden inside the primary gate.

## Validation Expansion

Report: `outputs/reports/chip_s8_dbu_packing_gate_val92_v11_adaptive.json`

- Primary hard gate: `92/92`.
- One case (`8020`) needed the declared 10-second retry after the 2-second pass.
- Raw-reference pass rate: `29/92`.
- Full 92-case CPU time: about `33 s`.

The initial 18 failures from the first 500-case probe are cleared in
`chip_s8_dbu_packing_gate_failure18_time10s_v5_final.json` (`18/18`). The fixes use
minimum-gap contact trees, one-DBU cycle handling, canonical shapes, deterministic
relation tie-breaks, and a failure-only SCIP retry. The first 500-case probe passed
`482/500` before those fixes; it is now development data and cannot serve as the
fresh confirmation after tuning.

## Fresh Complexity-Stratified Confirmation

Development report: `outputs/reports/chip_s8_dbu_packing_gate_fresh500_v2_current.json`

Final untouched confirmation:
`outputs/reports/chip_s8_dbu_packing_gate_confirm500_final_v1.json`

The first case-disjoint confirmation used five cases per block count and excluded
the prior 1,000 development cases.

- Primary hard gate: `493/500` (`98.6%`).
- Raw-reference gate: `170/500`.
- 21-40 blocks: `99/100` pass.
- 41-60 blocks: `100/100`.
- 61-80 blocks: `100/100`.
- 81-100 blocks: `98/100`.
- 101-120 blocks: `96/100`.

An earlier `110/500` result was traced to a `contact_reference` wiring bug and is
invalid. The corrected target-free relation portfolio resolves the remaining
seven development cases (`7/7`). On a new post-tuning case-disjoint cohort using
the endpoint portfolio:

- primary hard/soft gate: `498/500` (`99.6%`);
- raw-reference gate: `186/500`;
- 21-40, 41-60, 61-80 and 101-120 block strata: `100/100` each;
- 81-100 blocks: `98/100`.

The two misses (`5324`, `6341`) are hard-feasible with zero area/fixed/preplaced
violations and non-worse grouping/boundary. They fail only the declared
translation-normalized p95 displacement guard. This remains an explicit
limitation, not a 500/500 claim.

## Claim Boundary

The current result supports a validated integer projection layer for most frozen
validation cases. It does not yet support:

- a full 500/500 DBU survival claim: the final fresh cohort is `498/500`, with
  two displacement-only outliers;
- preservation of all unquantized soft metrics;
- real DEF routed-PPA improvement;
- a SOTA claim.

## Artifacts

- `src/floorflow/chip_flow/dbu_projection.py`
- `tests/test_chip_dbu_projection.py`
- `scripts/evaluate_chip_dbu_packing_gate.py`
- `outputs/reports/chip_s8_dbu_packing_gate_5case_v4.json`
- `outputs/reports/chip_s8_dbu_packing_gate_val92_v11_adaptive.json`
- `outputs/reports/chip_s8_dbu_packing_gate_failure18_time10s_v5_final.json`
- `outputs/reports/chip_s8_dbu_packing_gate_fresh500_v2_current.json`
- `outputs/reports/chip_s8c_endpoint_portfolio_failure7_v1.json`
- `outputs/reports/chip_s8_dbu_packing_gate_confirm500_final_v1.json`
