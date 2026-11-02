# M76 Candidate Legalizability Reranker

Date: 2026-11-01

## Decision

M76 is closed as a negative result. Two predeclared implementations failed the
grouped case-out development gate:

| Variant | Cost vs M75 guarded | V_rel vs M75 guarded | Improved / worsened cases |
|---|---:|---:|---:|
| v1, 16-hidden listwise | +24.52% | +160.86% | 1 / 31 |
| v2, 1-hidden with safety veto | +4.58% | +50.04% | 3 / 17 |

Both remain feasible, but neither preserves the deterministic M75 mechanism.
Further temperature, epoch or threshold tuning on the same 32 cases would be
validation overfitting.

## Why v1 Failed

The first model changed all 32 cases and had zero top-1 agreement with the
within-case oracle. It selected high-area/high-b2b candidates despite their
rank features ranking worse. With 28 training cases per fold, the 16-hidden
scorer had enough capacity to memorize fold-specific combinations instead of
learning a stable legalizability rule.

## Why v2 Also Failed

The repair used:

- a 1-hidden, near-linear scorer;
- stronger weight decay;
- longer cost temperature;
- M75 as the default selection;
- a hard target-free geometry veto;
- solver non-escalation.

The safety veto reduced the catastrophe, but the model still overrode 20 cases.
Seventeen worsened, including large V_rel regressions. Geometry non-regression
is insufficient because cost is exponential in V_rel. The v2 rule therefore
optimizes the wrong learned ordering even when geometry is protected.

## Reusable Components

The implementation remains reusable for a larger development cohort:

- `src/floorflow/chip_flow/candidate_legalizability.py`;
- permutation-equivariant candidate rank features;
- listwise candidate loss;
- 8-fold grouped case-out runner;
- target-free geometry safety veto;
- explicit diagnostic/oracle separation.

No model component or checkpoint from M76 enters the finalist.

## Required Follow-up

Do not retune this family on dev32 or Official-100. Reopen only with:

1. at least 10x more independent development cases;
2. locked features, architecture and training horizon before cohort opening;
3. a held-out test cohort disjoint from model selection;
4. a within-case oracle-gap diagnostic showing the feature set has enough
   signal before fitting;
5. evidence that V_rel, not only geometry, is predictable from target-free
   features.

## Artifacts

- `configs/chip_m76_candidate_reranker_v1.json`
- `configs/chip_m76_candidate_reranker_v2.json`
- `outputs/reports/chip_m76_dev_selection_v1.json`
- `outputs/reports/chip_m76_dev_selection_v2.json`
- `tests/test_chip_candidate_legalizability.py`
