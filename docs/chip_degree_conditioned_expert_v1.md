# Degree-Conditioned Relational Expert v1

Date: 2026-10-30

Status: rejected at the frozen-base adapter screen.

## Hypothesis

M17 showed that adding more sparse context slots does not help the
high-complexity tail. The next hypothesis was that high-degree and high-pin
blocks need a different computation, not more context.

## Architecture

`DegreeConditionedExpert` is placed inside `TypedGraphEncoder` after the
existing graph-message layers. It:

1. accumulates weighted b2b and p2b degree per block;
2. normalizes both degree fields;
3. predicts a per-block gate from the current token and degree pair;
4. applies a hidden/4 two-layer relational expert;
5. adds its output behind a zero-initialized global scale.

The existing block tokens, edge schema and counts are reused. D192 adds 28,854
parameters (`0.494%`). The production base was frozen and only the expert was
trained for 2k steps.

## Result

| Metric | Base | Expert | Relative |
|---|---:|---:|---:|
| Cost | 1.18966 | 1.20074 | +0.93% |
| V_rel | 0.04271 | 0.04659 | +9.10% |
| Feasibility | 1.0 | 1.0 | - |

The block strata show a narrow cost-only effect:

| Blocks | Cost | V_rel |
|---|---:|---:|
| 21-40 | +0.98% | +26.15% |
| 41-80 | +1.63% | +6.54% |
| 81-100 | -1.29% | +2.14% |
| 101-120 | +2.00% | +16.26% |

The expert is not enabled in production and does not enter confirmation.

## Artifacts

- `src/floorflow/chip_flow/degree_conditioned_expert.py`
- `tests/test_chip_degree_conditioned_expert.py`
- `configs/chip_degree_conditioned_expert_v1.json`
- `outputs/reports/chip_degree_conditioned_expert_v1.json`
- `outputs/chip_hierarchy/degree_expert_v1/eval_dev32_nfe4_refiner400.json`
