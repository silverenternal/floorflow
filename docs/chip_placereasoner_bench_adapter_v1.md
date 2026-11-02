# PlaceReasoner-Bench External Adapter v1

Date: 2026-11-01

## Source

- code: <https://github.com/fenggev56/PlaceReasoner-Beta>
- commit: `5a0c65b88f1931a4c3efb4d9c28f81251f296449`
- code license: GPL-3.0
- data: `fenggev566/PlaceReasoner-Bench` on HuggingFace
- data license: mixed open source; component-level RTL/PDK/FakeRAM terms apply
- canonical local data: `.data/place_reasoner_bench`
- compatibility path: `third_party/PlaceReasoner-Beta/Benchmark/PlaceReasoner-Bench`
  (symlink into `.data`)
- local size: 589 MB

The benchmark contains eight Nangate45 design directories and two predefined
floorplans per design, giving 16 placement tasks. This is **not** 16
independent source designs. Statistical confirmation must cluster by design,
and this cohort does not satisfy the earlier 15-independent-design gate.

## Reused Flow

`scripts/run_chip_placereasoner_external.py` reuses the benchmark's fixed
`detail_place.tcl` and `cts_route.tcl` templates. It imports the pinned
`macroflow` package, injects a no-op VLM, initializes the supplied floorplan
DEF, validates macro names against the loaded OpenROAD design, then runs:

1. PlanB legalized macro coordinates;
2. fixed detailed placement and PDN;
3. CTS and timing repair;
4. global/detailed route;
5. routed WNS/TNS, power, area, wirelength and DRC report parsing.

No LLM or external API is used. The adapter explicitly refuses OpenROAD
simulation mode.

## Provenance

- contract: `configs/chip_placereasoner_bench_v1.json`
- inventory builder: `scripts/build_chip_placereasoner_inventory.py`
- inventory: `outputs/reports/chip_placereasoner_inventory_v2.json`
- adapter: `scripts/run_chip_placereasoner_external.py`
- tests: `tests/test_chip_placereasoner_bench.py`

The inventory records the repository commit, dataset ID, benchmark README
SHA-256, and per-task Verilog/SDC/floorplan-DEF SHA-256.

## Verified Preflight

The adapter completed a real OpenROAD init in the pinned container
`openroad/orfs:chips-cvpr-703649ba` for `bp_fe/ar1`:

- no VLM;
- 11 macros discovered;
- core bounds `(20.1,21.0)-(615.2,614.6)`;
- macro footprint utilization `0.456`;
- no detail/CTS/route job launched.

Report:
`outputs/reports/chip_placereasoner_bp_fe_ar1_init_v1.json`.

## Restart and Headless Fixes

The first external detail attempt exposed three adapter defects:

- PlaceReasoner's `place_macro` path triggered OpenROAD restarts on hierarchical
  escaped macro names;
- restart re-sourced the design but not the custom placement proc;
- the pinned image requires Qt `offscreen`, while the inherited default
  `minimal` aborted during `save_image`.

The adapter now installs a restart-safe `safe_place` using OpenDB
`setLocation`/`setOrient`/`setPlacementStatus`, appends that proc to the init
script, bypasses GUI snapshots, and sets `QT_QPA_PLATFORM=offscreen`.

## bp_fe/ar1 Routed Pilot

The first legal pair was produced from the retained M27 seed checkpoints:

- candidate: `m27_seed21_5k`, coordinates
  `2db94b71655df5e6f329b19ecf0cc30ca28a8fac36b7b668636d8d03bb6137ad`;
- control: `m27_seed19_5k`, coordinates
  `6574e74f37acc78f03cad464a6ded0326d784784eae678bc60a8cfac841b3e51`;
- both legalize to 11/11 macros, zero overlap, inside core and exact fixed shape;
- both pass detail placement and complete the fixed timing-mode CTS/route.

| Metric | Candidate | Control | Candidate relative |
| --- | ---: | ---: | ---: |
| Routed WL (um) | 1,866,097 | 1,854,588 | +0.621% |
| WNS (ns) | -0.26 | -0.21 | -0.05 ns |
| TNS (ns) | -6.16 | -7.07 | +0.91 ns |
| Power (W) | 0.131 | 0.130 | +0.769% |
| Area (um²) | 216,730 | 216,327 | +0.186% |
| DRC | 0 | 8 | -8 |

This is a pipeline-validation pair only. The candidate is DRC-clean and has
better TNS, but routed WL, power, area and WNS do not improve. It cannot support
external PPA superiority.

Artifacts:

- `outputs/reports/chip_placereasoner_bp_fe_ar1_candidate_detail_v3.json`
- `outputs/reports/chip_placereasoner_bp_fe_ar1_control_detail_v3.json`
- `outputs/reports/chip_placereasoner_bp_fe_ar1_candidate_route_v2.json`
- `outputs/reports/chip_placereasoner_bp_fe_ar1_control_route_v2.json`
- `outputs/reports/chip_placereasoner_bp_fe_ar1_pair_v1.json`

## Remaining Gate Before PPA Claims

The `bp_fe/ar1` diagnostic now passes the execution gate. Before the remaining
15 tasks run, verify candidate/control generator stability and reserve a fresh
aspect task if the pair is used for tuning. Continue to report all PPA
directions; DRC improvement does not offset routed-WL regression.

Because only eight design directories exist, routed PPA results may support an
external robustness study with design-clustered uncertainty, but not a claim of
15-20 independent external designs.
