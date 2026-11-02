# Port-to-Block Consensus Feedback v1

Date: 2026-10-30

Status: implemented; rejected at the frozen warm-start quick screen.

## Gap

The typed hierarchy has a dedicated L3 port state and a port head, but the
predicted port geometry is only returned as an auxiliary prediction. It never
influences the L2 block trajectory that determines where those ports land.

This is the architecture-level counterpart of the observed p2b-to-routed-PPA
disconnect. The model can improve pin-to-block geometry without carrying that
information into legalized placement.

## Upgrade

`PortConsensusCoupler` reuses the existing port head and sparse incidence:

1. predict L3 port offsets before the block backbone;
2. aggregate predicted port offsets onto incident blocks with the existing
   `port_block_incidence`;
3. encode the pooled offsets, current block state and flow time in a hidden/4
   bottleneck;
4. add a zero-gated 3D residual to the block trajectory.

There is no new dense `N x N` tensor. The sparse `N x P` incidence already
exists in `TypedGraphOutput`. Target boxes, target labels, official metrics and
post-route measurements are never read.

## Budget

| Model | Parameters | Delta |
|---|---:|---:|
| Base d192 | 5,836,447 | - |
| Port-consensus d192 | 5,855,555 | +19,108 (+0.327%) |

## Invariants

- zero gate preserves the base prediction exactly;
- block permutation changes the residual by the same permutation;
- blocks without active ports remain unchanged;
- direct gradient reaches the bottleneck output;
- checkpoint reconstruction enables the opt-in path;
- a one-step real-cache warm-start smoke reports nonzero port-consensus
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

The whole-model warm-start screen reached cost `+1.26%` and V_rel `-0.23%`.
A clean frozen-base adapter screen reached cost `+4.11%` and V_rel `+6.89%`
against the untouched production checkpoint. Whole-model drift does not explain
the rejection.

## Artifacts

- `src/floorflow/chip_flow/port_consensus_coupler.py`
- `tests/test_chip_port_consensus_coupler.py`
- `configs/chip_port_consensus_warmstart_screen_v1.json`
- `outputs/smoke/chip_port_consensus_v1/port_consensus_warmstart_smoke.json`
- `outputs/reports/chip_port_consensus_warmstart_screen_v1.json`
