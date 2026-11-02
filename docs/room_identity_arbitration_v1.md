# Room Identity Arbitration v1

## Decision

The learned dense/query gate is rejected as an unnecessary layer. A simpler
target-free safety fallback retains the useful instance-query behavior and is at
least as strong end to end. The retained mechanism is:

1. Freeze the dense 384px room endpoint.
2. Produce a class-region-routed room prediction from the direct-GT instance
   query decoder.
3. Run the target-free explicit-plan validity and wall-continuity checks.
4. Fall back to the dense prediction for a case when the routed plan fails.

This preserves strict locality of the evaluation protocol: no target mask,
polygon, count, or metric participates in inference.

## Why the Learned Gate Was Rejected

The gate was trained on 256 train cases with target-free features and an 8-epoch
screen. Train positives occurred in 7.0% of cases. Its calibrated threshold was
0.4.

The learned safe-gate result on val491 was:

| Metric | Dense | Learned safe-gate | Delta |
|---|---:|---:|---:|
| Room IoU | 0.33386 | 0.33438 | +0.00053 |
| Instance F1 | 0.28883 | 0.29217 | +0.00334 |
| Count MAE | 3.59267 | 3.48065 | -0.11202 |
| Node F1 | 0.71271 | 0.71509 | +0.00238 |
| Edge F1 | 0.46500 | 0.47038 | +0.00538 |
| GED | 11.24236 | 10.89206 | -0.35031 |
| Plan plausible | 0.95519 | 0.95927 | +0.00407 |
| Wall continuity | 0.98856 | 0.98887 | +0.00031 |

The same query route with only the deterministic safety fallback, without the
learned gate, produced:

| Metric | Dense | Global query + safety | Delta |
|---|---:|---:|---:|
| Room IoU | 0.33386 | 0.33383 | -0.00003 |
| Instance F1 | 0.28883 | 0.29271 | +0.00388 |
| Count MAE | 3.59267 | 3.48473 | -0.10794 |
| Type accuracy | 0.68362 | 0.68048 | -0.00313 |
| Node F1 | 0.71271 | 0.71562 | +0.00291 |
| Edge F1 | 0.46500 | 0.47079 | +0.00579 |
| GED | 11.24236 | 10.90020 | -0.34216 |
| Plan plausible | 0.95519 | 0.95927 | +0.00407 |
| Wall continuity | 0.98856 | 0.98890 | +0.00034 |

Paired comparison of the learned gate against the safety-only route:

- Room IoU: +0.00056, CI [-0.00062, 0.00238], p=0.821
- Instance F1: -0.00054, CI [-0.00242, 0.00105], p=0.641
- Count MAE: -0.00407, CI crosses zero
- Node F1: -0.00053, CI crosses zero
- Edge F1: -0.00041, CI crosses zero
- GED: -0.00815, CI crosses zero

The learned gate therefore adds no reliable benefit over its safety-only
ablation. Keeping it would increase complexity and weaken the causal story.

## Safety-Only Paired Evidence

Against the dense endpoint, the safety-only route has:

- Count MAE -0.10794 (95% CI [-0.19145, -0.02648], p=0.031).
- Node F1 +0.00291 (95% CI [0.00013, 0.00603], p=0.013).
- Edge F1 +0.00579 (95% CI [0.00184, 0.01106], p=0.0033).
- GED -0.34216 (95% CI [-0.44807, -0.24440], p=1.5e-10).
- Room IoU -0.00003 and Instance F1 +0.00388; both paired CIs cross zero.
- Plan plausible +0.00407 with zero worsened cases.
- Wall continuity +0.00034 with CI crossing zero.

The safe route is therefore an identity/graph-consistency improvement, not a
Room-IoU SOTA claim.

## Evidence

- `outputs/train/room_identity_arbitration_v1/report.json`
- `outputs/train/room_identity_arbitration_v1/best.pt`
- `outputs/reports/room_identity_arbitration_cal64_v1.json`
- `outputs/reports/room_identity_arbitration_holdout64_v1.json`
- `outputs/reports/room_identity_arbitration_safe_val491_v1.json`
- `outputs/reports/room_identity_arbitration_safe_val491_stats_v1.json`
- `outputs/reports/room_identity_safety_global_val491_v1.json`
- `outputs/reports/room_identity_safety_global_val491_stats_v1.json`
- `outputs/reports/room_identity_arbitration_vs_global_safe_val491_stats_v1.json`
