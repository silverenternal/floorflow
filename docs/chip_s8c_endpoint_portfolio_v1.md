# S8C Scale-Aware DBU Endpoint Portfolio v1

Date: 2026-11-02

## Problem

The first fresh complexity-stratified DBU screen exposed a scale-dependent
projection weakness. After correcting a `contact_reference` wiring bug, the
current endpoint passed `493/500` primary cases. Seven remained:

- four exceeded the absolute max-displacement guard;
- two were rejected after both contact-snap routes;
- one integer projection was infeasible.

All seven were between 39 and 120 blocks except one 39-block case.

## Portfolio

`project_dbu_endpoint_portfolio` uses only target-free endpoint state:

1. Try scale-pack relation `expanded`.
2. Re-run with relation `restored` when the first endpoint is rejected or its
   p95 displacement exceeds the declared fraction of layout span.
3. If neither is accepted, retry both relations with the candidate's own raw
   legalized contact reference.
4. Choose the accepted endpoint with the smallest p95 displacement, then max
   displacement.

Official cost, target positions, and case IDs do not select the route. The
original instance remains the hard/soft audit oracle.

The gate uses translation-normalized p95 displacement for scale-robust selection
and still records the raw max displacement in the repair audit. A common global
translation is removed because the official FloorSet metrics and boundary checks
are invariant to the layout origin, while non-rigid movement remains measurable.

## Failure-Only Result

Report: `outputs/reports/chip_s8c_endpoint_portfolio_failure7_v1.json`

- Cases: `7/7` pass.
- Hard feasibility and zero area/fixed/preplaced violations: `7/7`.
- Grouping and boundary no worse than the selected continuous reference: `7/7`.
- Five cases selected `restored_continuous`; two preferred
  `expanded_continuous`.
- Total CPU time: `12.57 s`.

## Final Fresh Confirmation

Report: `outputs/reports/chip_s8_dbu_packing_gate_confirm500_final_v1.json`

On a new post-tuning case-disjoint 500-case cohort:

- `498/500` pass hard feasibility, zero area/fixed/preplaced violation, and
  grouping/boundary non-regression;
- the two misses (`5324`, `6341`) pass those same hard/soft gates and fail only
  the translation-normalized p95 displacement guard;
- all block-count strata except 81-100 pass `100/100`; 81-100 passes `98/100`.

The two outliers remain a declared failure boundary. The endpoint is not claimed
as 500/500.

## Artifacts

- `src/floorflow/chip_flow/dbu_endpoint_portfolio.py`
- `tests/test_chip_dbu_endpoint_portfolio.py`
- `scripts/evaluate_chip_dbu_packing_gate.py --endpoint-portfolio`
- `outputs/reports/chip_s8_dbu_packing_gate_fresh500_v2_current.json`
- `outputs/reports/chip_s8c_endpoint_portfolio_failure7_v1.json`
- `outputs/reports/chip_s8_dbu_packing_gate_confirm500_final_v1.json`
