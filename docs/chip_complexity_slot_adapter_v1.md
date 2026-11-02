# Complexity-Conditioned Sparse Slot Adapter v1

Date: 2026-10-30

Status: rejected at the frozen-base adapter screen.

## Motivation

The architecture sweep showed that uniform residuals can improve a narrow
middle block-count band without improving the 101-120 block tail. The next
hypothesis was that high-complexity cases need more relational context, not a
uniformly stronger residual.

The project already contains `graph_complexity` and
`LearnedSparseSlotRouter` in the sparse hierarchy. Those components were reused
instead of implementing another router.

## Adapter

`ComplexitySlotAdapter`:

1. computes case complexity from graph size, b2b density and constraints;
2. routes typed block tokens through two to four active sparse slots;
3. mixes each block with its top-two slot contexts;
4. scales a zero-gated 3D trajectory residual by complexity and the learned
   soft slot budget.

The production base was frozen. Only the adapter was trained for 2k steps on the
frozen 5k subset. This isolates the architecture from whole-model drift.

The real-cache smoke exposed and fixed an existing bf16 autocast bug in
`LearnedSparseSlotRouter`: assignment weights were not explicitly cast to the
token dtype before `scatter_`.

## Result

Against the untouched production checkpoint at NFE4/refiner400:

| Metric | Base | Adapter | Relative |
|---|---:|---:|---:|
| Cost | 1.18966 | 1.24996 | +5.07% |
| V_rel | 0.04271 | 0.04936 | +15.58% |
| Feasibility | 1.0 | 1.0 | - |

The block strata also reject the hypothesis:

| Blocks | Cost | V_rel |
|---|---:|---:|
| 21-40 | +1.07% | +2.71% |
| 41-80 | +2.78% | +25.83% |
| 81-100 | +0.16% | +16.74% |
| 101-120 | +17.18% | +7.49% |

The router does allocate more capacity with complexity, but extra slot context
alone makes the hardest stratum worse. The path remains opt-in and is not
registered as a candidate.

## Artifacts

- `src/floorflow/chip_flow/complexity_slot_adapter.py`
- `tests/test_chip_complexity_slot_adapter.py`
- `configs/chip_complexity_slot_adapter_v1.json`
- `outputs/reports/chip_complexity_slot_adapter_v1.json`
- `outputs/chip_hierarchy/complexity_slot_adapter_v1/eval_dev32_nfe4_refiner400.json`
