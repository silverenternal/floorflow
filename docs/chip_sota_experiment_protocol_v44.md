# Chip SOTA Experiment Protocol v44

## Purpose

This protocol separates model quality, inference budget, and routed-PPA evidence.
It is an experiment contract, not manuscript text.

## Canonical FloorSet Comparison

- Dataset: FloorSet Official-100.
- Train seeds: 19, 20, 21.
- Sampling: NFE4 with the promoted EMA checkpoint.
- Projection: eight deterministic hard-constraint iterations.
- Refiner: shared Trinity closed-form refiner, linear LR 1e-3, boundary weight 2,
  1600 steps unless explicitly labeled otherwise.
- Execution: deterministic PyTorch/CUDA kernels.
- Legalizer: scale_pack with a 2000 ms LP limit and one GLOP solver thread.
- Primary metrics: legalized cost and V_rel.
- Structural diagnostics: grouping, boundary, MIB, b2b HPWL, p2b HPWL,
  area/outline gap, and aspect error.
- Legality: every reported score must have feasible fraction 1.0.

The fixed Trinity anchor is the reproducible three-seed result at its standard
400-step refiner setting. A second seed-19 run gives Trinity the same 1600-step
refiner budget and is kept as a separate matched-budget boundary.

## Required Comparisons

| Comparison | Role |
|---|---|
| Candidate versus fixed Trinity d192 | Main fixed-anchor result |
| Candidate versus Trinity at 1600 steps | Matched-budget diagnostic |
| Typed hierarchy versus graph-token control | Architecture-selection diagnostic |
| Candidate versus graph-token at 400 steps | Original mechanism comparison |
| NFE1/4/8 and refiner-step curves | Efficiency diagnostics |

## External PPA

- Selection is frozen before candidate routed-PPA outcomes:
  `configs/chip_external_holdout_adapter_v44.json`.
- Eligible designs: eight nangate45 OpenROAD designs with complete source license,
  platform license, nonzero macro instances, and baseline PDN inputs.
- Planned scale: 24 design-seed pairs.
- Statistical unit: design cluster; seeds are repeated measurements, not independent
  external cases.
- All candidate/control pairs use the same refiner, route budget, ORFS image,
  placement writer, and legalizer route.
- If legalizer routes differ, both arms are regenerated with the same shelf fallback.
- Primary routed endpoint: detailed-route wirelength.
- Required guards: global-route overflow, DRC, setup WNS/TNS, power, and area.

## Claim Policy

- A fixed-anchor SOTA claim requires all three candidate seeds to beat both
  reproducible Trinity metrics and requires legal feasibility 1.0.
- A matched-compute claim requires beating Trinity at the same refiner budget.
- A PPA claim requires routed artifacts, no DRC regression, and no surrogate
  substitution.
- MIB is reported as non-discriminative when both arms are zero.
- Seed-dependent architecture ranking is reported, not hidden.

## Primary Evidence

- `outputs/reports/chip_refiner_official_1600_three_seed_v44.json`
- `outputs/reports/chip_matched_budget_1600_detglop_seed19_v44.json`
- `outputs/reports/chip_model_promotion_v44.json`
- `outputs/reports/chip_seed_rootcause_v44.json`
- `outputs/reports/chip_external_holdout_inventory_v44.json`
- `outputs/reports/chip_external_ppa_holdout_v44_all.json`
- `outputs/reports/chip_ppa_endpoint_certification_v44.json`
- `outputs/reports/chip_p2b_routing_rootcause_v44_holdout.json`
- `outputs/reports/chip_final_sota_gate_v44.json`

## Engineering Contract

- Every first-party Python file remains below 600 lines.
- Reuse the existing checker, scorer, refiner, legalizer, ORFS flow, and caches.
- Preserve failed sweeps and earlier negative reports.
- Generated data and checkpoints remain under `outputs/` and are not committed.
