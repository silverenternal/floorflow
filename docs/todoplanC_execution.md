# todoplanC execution report (2026-10-26)

## P0: data and checkpoints acquired

The network proxy was reopened and HuggingFace became reachable. Acquired:

- official 100-case `LiteTensorDataTest` (255MB, already in the cloned repo);
- `dataset21_lite.pth`: **9,408 real 21-block FloorSet layouts** (38MB) —
  areas, ground-truth boxes `(w,h,x,y)`, two netlist tensors, pins;
- released **Trinity d192** checkpoint + config from `KBlueLeaf/Trinity`
  (23MB, 5.9M parameters).

The full 6.6GB 1M tar was not needed: dataset21_lite is a real FloorSet
training set and is sufficient for a controlled small-scale study.

## P1: physics parity and checkpoint reproduction

Physics parity against the official FloorSet checks on all 100 validation
cases: boundary 1.00, MIB 1.00, group 1.00; GT overlap exactly 0.

Released Trinity d192 (samples=1) on official cases:

| blocks | feasible | cost | hpwl_gap | v_rel | overlap pairs |
|---:|---|---:|---:|---:|---:|
| 21 | yes | 1.197 | 0.0123 | 0.0870 | 0 |
| 60 | yes | 1.059 | 0.0073 | 0.0263 | 0 |
| 120 | yes | 2.497 | 0.6208 | 0.1642 | 0 |

## P2: integrated model

`src/floorflow/chip_flow/model.py` is a block-token rectified-flow model with
exact-area `(cx/s, cy/s, log w/h)` decode, a shelf-pack source, optional top-k
sparse attention, and token OT reordering. `src/floorflow/chip_physics/`
implements the six Trinity-compatible terms.

## P3/P4: real-data 4-way ablation

dataset21_lite, 9,000 train / 408 val, 1,500 steps, batch 8:

| config | latent MSE | overlap | total soft cost |
|---|---:|---:|---:|
| dense, no OT | 0.1994 | 0.5027 | 0.01533 |
| dense, OT | 0.2333 | **0.1972** | **0.01239** |
| sparse k=16, no OT | **0.1963** | 0.5041 | 0.01565 |
| sparse k=16, OT | 0.2276 | 0.2010 | 0.01280 |

## Honest claim

On real 21-block FloorSet data, **token minibatch OT is both a legality and a
soft-cost lever**: overlap drops ~2.5x and total soft cost drops ~19% at a
1,500-step budget. Sparse top-k attention at k=16 is indistinguishable from
dense and slightly worse at this scale. We do not claim a Trinity win or full
FloorSet SOTA; the Trinity baseline is reported separately because its
`validate` cost uses a different normalisation.

## Next step for a paper-grade run

Download the full 1M tar, train the d192 config for 20k-40k steps on the 12k
dev split, and measure **steps-to-soft-cost-threshold** for the same four-way
ablation. That is the experiment that would turn the current positive signal
into a convergence claim.

## Evidence

- `outputs/reports/chip_physics_parity_v1.json`
- `outputs/reports/trinity_d192_baseline_v1.json`
- `outputs/reports/chip21_{dense_no_ot,dense_ot,sparse_no_ot,sparse_ot}_v1.json`
- `outputs/reports/chip21_ablation_summary_v1.json`
