# Chip External M27 Pilot v1

Date: 2026-10-31

## Scope

The prior 15-design ChiPBench work opened simulation/ungrouping and adapter
plumbing only. It did not contain routed outcomes for the canonical M27 model.

This pilot freezes:

- candidate: M27 canonical seed21 typed hierarchy;
- control: M24 scratch depth12 typed hierarchy;
- common candidate/control generation protocol;
- three intermediate designs: `ariane81`, `dft68`, `or1200`.

The cohort was already used for adapter smoke tests, so this is external pilot
evidence only. It cannot replace a fresh confirmation cohort.

## Gate

1. Generate both arms.
2. Require legal, zero-overlap macro placements with a common legalizer route.
3. Materialize DEF and verify every macro round-trips with the expected count.
4. Only after both arms pass may ORFS global/detail route be opened.

No routed, congestion, DRC, timing or power result exists before step 4.

## Positioned Pair Repair

The initial pilot collapsed because both arms were emitted through the same
shelf fallback. Position-aware shelf packing was then run independently for both
arms. All three pairs are now legal, zero-overlap, use a common route and
round-trip exactly. Candidate/control TCL hashes differ on all three cases.

The frozen pair manifest is:
`outputs/reports/chipbench_m27_pilot_positioned_pairs_v1.json`.

## ORFS Pilot Result

The `or1200` candidate was run first through the pinned ORFS image using the
DEF2DB target and the dependency-free continuation stages:

```text
def2db -> tapcell -> PDN -> global place -> resize -> detail place
```

Synthesis rebuilding, the unavailable Python `getMacroMetrics`, and the
hierarchical macro-name mismatch were avoided. Def2DB and all floorplan stages
completed.

The first attempt without ODB initialization timed out in detail placement after
3600 seconds. The repository's existing pre-macro Tcl then marked macros FIRM
and initialized standard-cell placement; detail placement subsequently
completed in 0.36 seconds.

CTS repair timing next exceeded the machine memory limit at roughly 99 GB. The
candidate was rerun with `SKIP_CTS_REPAIR_TIMING=1`, allowing CTS, global route
and detail route to complete. The frozen `droute_end_iter=1` setting left
49,712 DRCs and final extraction failed on a short, so this sensitivity run is
invalid routed evidence.

Before opening the control outcome, the common protocol was amended for both
arms:

- `SKIP_CTS_REPAIR_TIMING=1` to avoid the CTS memory failure;
- `droute_end_iter=10` to avoid the intentionally unconverged one-iteration
  route.

Candidate and control must both be rerun under those identical settings.

The amended candidate run completed detailed routing in 41m46s but still
reported 44,485 DRCs; final extraction again failed on a short. Increasing the
route iteration budget is not a credible fix because the last several
iterations reduced DRC by only a few hundred each while taking minutes per
iteration.

The control arm was therefore not opened. The M27 external routeability route
is retained as negative evidence under this ChiPBench ORFS stack.

## Correct-DEF Follow-up

The 44,485-DRC runs had imported `pre_place.def` into ORFS. That DEF does not
contain the materialized model placement, so the pre-macro step fixed macros
at their original locations. Importing the verified candidate DEF instead:

- reduced detail-route DRC to 4,196;
- completed final extraction and all metrics;
- produced routed WL `1,007,925`, congestion `0.867361`, setup WNS
  `-3.65592 ns`, TNS `-362.556 ns`, power `0.0287176`, and std-cell area
  `56,230`.

This run remains invalid for PPA comparison because DRC is nonzero.

Snapping macro origins from the 10-DBU manufacturing grid to a 560-DBU grid
reduced DRC further to 3,101, but final extraction still failed on a short.
Residual failures are concentrated on macro pins, not generic routing
congestion. Therefore the next external gate is pin-aware macro placement or a
predeclared routeable design subset, not another route-iteration sweep.

## Pin-Residue Alignment

A 0.28 um routing halo followed by reference-DEF pin-residue alignment produced
0 macro overlaps and 0 out-of-core macros. Relative to the unaligned correct-DEF
run, detail-route DRC fell from 4,196 to 982 while the flow still completed and
reported all metrics.

This is the strongest routeability sensitivity observed so far, but it is still
invalid for PPA because DRC is nonzero. Routed WL is 1,096,979, congestion
0.874001, setup WNS -3.70769 ns, TNS -387.167 ns, power 0.0314545 and
standard-cell area 57,448.3. The control arm remains unopened.

The exact timeout report is:
`outputs/reports/orfs/chipbench_m27_positioned_or1200_candidate_s19.json`.

## Next Gate

Diagnose whether detail-placement infeasibility comes from preserved standard
cell placement, macro blockage, or a missing common protocol adjustment. Any
retry must use the same predeclared change for candidate and control. Partial
global-placement HPWL or congestion must not be substituted for routed PPA.
