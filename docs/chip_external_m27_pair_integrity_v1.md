# M27 Positioned External Pair Integrity

Date: 2026-10-31

## Result

The canonical M27 candidate and M24 scratch depth-12 control completed placement
and DEF materialization on `ariane81`, `dft68` and `or1200`.

- all candidate/control pairs are checker-feasible;
- all pairs have zero overlap;
- both arms use the same legalizer route per case;
- all macro counts and coordinate round-trips match;
- candidate/control TCL hashes differ for all three cases.

This resolves the earlier collapsed-pair blocker, where both arms fell back to
the same shelf TCL. The original failed evidence remains archived and is not
replaced.

## Frozen Hashes

| Case | Candidate TCL | Control TCL |
|---|---|---|
| ariane81 | `f6e7010db24b80c36f7b26fec82ca1fbc55f3208f5aa3359eda7247af8089f5b` | `2dc59626d1864a677e5620ae05852032d48e1cae04bd8865d6063ec85765bd3d` |
| dft68 | `8c4919e3b63441168b7e94a613be9aa07aa6458c2e222fb802dad0617bddf476` | `9f5da46b4e9c75f3822f152d814e0d5a69a4dddd250359971e7bafab8a58df37` |
| or1200 | `0bbd39527636d8d9b944945ee754cb4f4f88620ef459fc44b8b4b922d2c17ecd` | `7199cb78a2fe99ed63573760a379f9e39530bb6ae10852addbb840ab97f91b31` |

Machine-readable pair manifest:
`outputs/reports/chipbench_m27_pilot_positioned_pairs_v1.json`.

## Engineering Repairs

Three reproducibility defects were fixed before opening ORFS:

1. the pair-manifest builder now computes TCL SHA-256 and rejects collapsed
   candidate/control pairs;
2. ORFS manifest and runner scripts resolve explicit repository-relative paths;
3. ChiPBench staging preserves the source Verilog suffix expected by
   `rtl/*.v`.

For DEF2DB runs, ORFS now continues from the generated
`2_3_floorplan_macro.odb` through the Makefile's dependency-free `do-` stages.
This avoids re-synthesis, which changed macro names and made the frozen TCL
unresolvable. It also avoids the image's unavailable Python-based
`getMacroMetrics`.

## Boundary

Pair integrity is not routed-PPA evidence. Routed wirelength, congestion, DRC,
timing, power and area must come from the completed ORFS runs, and this
three-case cohort remains an inspected pilot rather than a fresh confirmation
set.
