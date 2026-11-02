# ChiPBench 15-Design Adapter Round-Trip v1

Date: 2026-10-31

Status: complete; adapter evidence only.

## Scope

Run the existing `run_chipbench_external_screen.py` over all 13 frozen
ChiPBench-D designs plus two ORFS supplements using the production checkpoint
at NFE1 with no refiner.
This verifies external graph import, model invocation and DEF materialization;
it does not score or route placements.

## Result

All 15/15 designs pass:

- placement report;
- DEF materialization;
- expected macro-instance count;
- round-trip coordinate preservation.

Macro counts range from 6 to 220. The largest ChiPBench coordinate error is
`0.000215 um`; the largest supplement error is `0.000186 um`. Feasibility,
congestion, DRC, timing, power and routed
wirelength are intentionally not claimed.

## Remaining Cohort Work

- freeze candidate/control placement identities;
- run the external routed-PPA flow symmetrically.

## Routed Pilot Findings

The first ORFS route attempts exposed two upstream blockers:

- flattened ChiPBench Verilog can fail Yosys `check -assert` with multiple
  conflicting drivers; VeriGPU is the first observed case;
- model-generated legal placements can still be too congested for GRT.

For sky130hd/chameleon, the design-default macro placement completes detailed
route with `707841 um` routed wirelength, zero DRC and `0.796568` global-place
congestion. Candidate and control placements under both forced-shelf and
scale-pack protocols fail GRT congestion. This is retained as a real negative;
the design default is not substituted for either arm.

External configs are now mounted through `/work`, `DESIGN_HOME` is set
explicitly, and ChiPBench data is staged into the `rtl/lef/lib` layout expected
by its ORFS configs.

## Artifacts

- `outputs/reports/chipbench_adapter_full13_v1.json`
- `outputs/reports/chipbench_orfs_supplement_v1.json`
- `outputs/reports/chip_chipbench_holdout_manifest_v1_verified.json`
- `outputs/chipbench_adapter_full13_v1/<design>/placement.json`
- `outputs/chipbench_adapter_full13_v1/<design>/materialize.json`
- `outputs/chipbench_adapter_full13_v1/<design>/candidate_seed19.def`
- `outputs/chipbench_orfs_supplement_v1/<design>/candidate_seed19.def`
