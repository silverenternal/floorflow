# S6 Cluster-Reflow Quantization Audit v1

Date: 2026-11-01

## Trigger

The selected continuous-protocol reflow contained many sub-nanometer moves.
We replayed every selected action and quantized moved coordinates to the
`0.001 um` manufacturing grid before recomputing cluster connectivity.

## Findings

- Development92: 0 actions survive at `0.001 um`.
- Fresh500: 56 selected actions; 27/56 lose their grouping reduction after
  quantization.
- Of 73 selected block moves, 30 are `<1e-6 um` and 37 are `<1e-3 um`.

The continuous FloorSet scorer permits contact at floating precision. An
actual DEF uses integer database units; the selected tiny contacts therefore
cannot be promoted as robust chip behavior.

## Decision

Demote cluster reflow from deployable candidate to continuous-protocol
diagnostic. Do not claim its 269/130/53 anchor wins for real DEF designs.

The S6 replacement must generate candidate actions directly on the declared
coordinate lattice. Each action needs:

1. integer-DBU endpoint construction;
2. contact and separation checks after lattice snapping;
3. exact-area or fixed-shape validation after snapping;
4. target-free selection and the same b2b/p2b cap;
5. fresh confirmation independent of the continuous action data.

## Artifacts

- `src/floorflow/chip_flow/legalized_cluster_reflow.py`
- `scripts/collect_chip_cluster_reflow_actions.py`
- `tests/test_chip_legalized_cluster_reflow.py`
- `outputs/reports/chip_s6_cluster_reflow_actions_val92_quantized_v3.json`
- `outputs/reports/chip_s6_cluster_reflow_holdout500_vs_nfe16_v2.json`

