# M75 Tail-Safe Seed-Diverse Selection

Date: 2026-11-01

## Decision

The fixed `legality_guarded_consensus` policy passes the internal stable
primary gate under an explicitly disclosed absolute candidate-search budget:

- mean cost is lower than each of the three Trinity d192 anchors;
- mean V_rel is lower than each anchor;
- the case-cluster pooled cost change is `-1.57%`, 95% CI
  `[-2.99%, -0.43%]`;
- the case-cluster pooled V_rel change is `-51.24%`, 95% CI
  `[-1.43%, -0.63%]`;
- candidate feasibility is `1.0`;
- there are no legalizer rung or solve escalations relative to the per-case
  M48 seed19 baseline;
- all selected candidates have complete target-free geometry.

This is a named absolute-budget result. It is not a same-inference result.

## Selection Change

The original legality-first selector minimizes V_rel and can choose a
catastrophic area/b2b endpoint. The new policy keeps legality-first unless one
of these fixed target-free risks triggers:

- outline compactness below `0.65`;
- absolute outline aspect above `0.75`;
- b2b HPWL above `1.25x` the candidate-set median;
- p2b HPWL above `1.25x` the candidate-set median.

When triggered, the selector uses a permutation-invariant consensus rank over
V_rel, b2b, p2b, outline compactness and outline aspect. Selection is restricted
to candidates that do not increase legalizer rung or solve count. It never sees
target boxes, official cost or Trinity outcomes.

## Candidate Pool

The pool reuses already-generated outputs from:

- M48 inference seeds 19, 20 and 21;
- deterministic M61B candidates;
- deterministic M65 candidates;
- `baseline`, `expanded` and `projector32` legalizer variants.

The budget is 10 trajectories, total NFE `14`, total refiner steps `16000` and
15 legalizer calls. This budget must remain visible in every paper-facing
comparison.

## Tail Result

The rescue changes case 89 from the M48 seed20 catastrophic endpoint to the
M48 seed21 projector32 endpoint:

| Quantity | M74 seed20 selected | M75 rescue |
|---|---:|---:|
| Case 89 cost | 2.206994 | 1.450626 |
| Case 89 area gap | 1.0182 | 0.1185 |
| Case 89 b2b HPWL gap | 0.4721 | 0.0356 |

This is a large improvement but not parity with Trinity on the tail. Case 89
remains worse than the matched Trinity anchor.

## Remaining Weakness

Area remains worse than Trinity:

- pooled area-gap delta `+0.01847`, conservative 95% CI
  `[+0.00957, +0.02715]`;
- pooled b2b mean delta `+0.00470`, conservative CI crossing zero;
- the b2b component cannot be claimed as improved.

The candidate therefore wins cost and V_rel through the robust structural
mechanism while retaining a material area weakness. M76 or M56 must target
that component before any claim that the method dominates every primary
physical metric.

## Artifacts

- `configs/chip_m75_tail_safe_selection_v1.json`
- `outputs/reports/chip_m75_seed_diverse_guarded_official100_v1.json`
- `outputs/reports/chip_m75_decision_v1.json`
- `outputs/reports/chip_m75_seed_diverse_guarded_tail_seed19_v1.json`
- `outputs/reports/chip_m75_seed_diverse_guarded_tail_seed20_v1.json`
- `outputs/reports/chip_m75_seed_diverse_guarded_tail_seed21_v1.json`
