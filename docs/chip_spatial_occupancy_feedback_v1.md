# Differentiable Spatial Occupancy Feedback v1

Date: 2026-10-30

Status: implemented; rejected at the frozen warm-start quick screen.

## Gap

The typed encoder sees static block features and netlist edges, and the
per-block backbone sees all block tokens. Neither path explicitly represents
the current spatial occupancy or crowding field. This is most relevant for
high-block-count cases, where local placement conflicts and long-tail legalized
cost remain the dominant failures.

## Upgrade

`SpatialOccupancyCoupler` builds a differentiable `16 x 16` field from the
current block trajectory:

1. map normalized block centers into grid coordinates;
2. soft-splat occupancy, exact-area pressure and aspect magnitude into three
   field channels;
3. encode the field with two small convolutions;
4. bilinearly sample the field at every block center;
5. add a hidden/4 zero-gated 3D residual to the block trajectory.

The field is target-free and has `O(N + G^2)` memory with `G=16`; no dense
`N x N` tensor is created.

## Budget

| Model | Parameters | Delta |
|---|---:|---:|
| Base d192 | 5,836,447 | - |
| Spatial-occupancy d192 | 5,849,323 | +12,876 (+0.221%) |

## Invariants

- zero gate preserves the base prediction exactly;
- block permutation changes the residual by the same permutation;
- direct gradient reaches the bottleneck output;
- checkpoint reconstruction enables the opt-in field;
- a one-step real-cache warm-start smoke reports nonzero spatial-field
  gradient.

## Frozen Screen

- Data: `configs/chip_train_screen_5k_v1.json`;
- seed: 23;
- quick: 2k steps, NFE4/refiner400;
- confirmation: 5k steps, NFE4/refiner1600;
- device: cuda:1;
- gate: no primary regression above `0.2%`, at least one improvement of
  `0.5%`, and legalized feasibility `1.0`.

## Result

The arm reached cost `+5.83%` and V_rel `+13.79%`. The cost interval was
entirely above zero, so the quick gate rejected the occupancy-field path.

## Artifacts

- `src/floorflow/chip_flow/spatial_occupancy_coupler.py`
- `tests/test_chip_spatial_occupancy_coupler.py`
- `configs/chip_spatial_occupancy_warmstart_screen_v1.json`
- `outputs/smoke/chip_spatial_occupancy_v1/spatial_occupancy_warmstart_smoke.json`
- `outputs/reports/chip_spatial_occupancy_warmstart_screen_v1.json`
