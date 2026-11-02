# Bidirectional b2b Graph Messages v1

Date: 2026-10-30

Status: implemented; rejected at the frozen warm-start quick screen.

## Source-Level Finding

FloorSet b2b edges are usually stored once per connection, but
`GraphMessageBlock._b2b_context` only sends the source feature to the target.
The degree normalization counts both endpoints, so the target receives a
partially normalized one-way message and the source receives no graph message.

Sampling the first eight development cases showed that only 24-4,448 reverse
pairs out of 336-6,146 edges are explicitly duplicated. Most connections
therefore lose half of their undirected message flow.

## Upgrade

`GraphMessageBlock` now supports `symmetric_edges`. For every active edge
`source -> target`, it sends the existing source message to the target and the
same weighted target message to the source. The existing degree,
edge-gate, projection, residual block and parameter tensors are reused. The
path adds no parameters and no dense tensor.

The default remains directional so existing checkpoints are unchanged unless
the typed hierarchy is explicitly constructed with
`symmetric_b2b_messages=True`.

## Validation

- a one-edge test distinguishes directional and symmetric endpoints;
- block permutation equivariance is preserved;
- checkpoint reconstruction enables the symmetric path;
- a real-cache one-step warm-start smoke completes with finite loss and finite
  gradients;
- parameter delta is zero.

## Frozen Screen

- Data: `configs/chip_train_screen_5k_v1.json`;
- seed: 23;
- quick: 2k steps, NFE4/refiner400;
- confirmation: 5k steps, NFE4/refiner1600;
- device: cuda:1;
- gate: no primary regression above `0.2%`, at least one improvement of
  `0.5%`, and legalized feasibility `1.0`.

## Result

The symmetric-message arm reached cost `+7.98%` and V_rel `+22.14%`. Both
paired intervals excluded zero on the regression side. The production
directional message path remains the default and checkpoint of record.

## Artifacts

- `src/floorflow/chip_flow/sparse_layers.py`
- `src/floorflow/chip_flow/typed_graph_encoder.py`
- `src/floorflow/chip_flow/hierarchical_flow.py`
- `tests/test_chip_symmetric_b2b_messages.py`
- `configs/chip_symmetric_b2b_warmstart_screen_v1.json`
- `outputs/smoke/chip_symmetric_b2b_v1/symmetric_b2b_warmstart_smoke.json`
- `outputs/reports/chip_symmetric_b2b_warmstart_screen_v1.json`
