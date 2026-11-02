# Atom Metric and Proxy Research Notes

Date: 2026-11-01

## Design Conclusions

The atom metric layer should not rank checkpoints from one scalar training loss.
It should treat every proxy as a calibrated surrogate with explicit coverage,
ranking, false-decision, and sensitivity audits.

Recent floorplan work supports separating metric families:

- FMLM (`arXiv:2604.04859`) formulates heterogeneous floorplan tasks through
  one structured markup representation. This supports keeping room, wall,
  opening, furniture, and graph outputs as separately auditable atoms.
- TLC-Plan (`arXiv:2602.07100`) reports RPLAN FID/MSE but also emphasizes
  vector topology and validity. Generation quality alone is insufficient.
- Boundary-constrained diffusion for floorplans (`arXiv:2602.01949`) shows that
  FID can miss diversity collapse and that boundary adherence needs separate
  measurement. This motivates structure and locality metrics outside the main
  likelihood/perceptual score.
- Space Syntax floorplan generation (`arXiv:2602.22507`) separates functional
  organization metrics from native visual/geometric fidelity. FloorFlow should
  therefore keep graph and room-adjacency metrics independent from Room IoU.
- Causal Judge Evaluation (`arXiv:2512.11150`) demonstrates calibrated cheap
  surrogates against a small oracle slice, including ranking accuracy,
  transport audits, and refusal to make level claims when calibration fails.

## FloorFlow Adaptation

`PROXY-REAL-V1` uses a stricter engineering contract than a single correlation:

1. target-free inference;
2. identical case manifest and seed;
3. at least six checkpoint snapshots;
4. per-case Spearman and Pearson;
5. tolerant top-checkpoint agreement;
6. paired sign agreement;
7. false-pass and false-fail rates;
8. deterministic sensitivity;
9. rejection when the proxy is flat or the canonical metric lacks variation.

The proxy may be a cheaper faithful implementation of the canonical metric, not
necessarily a different statistical quantity. For example,
`fast_room_macro_iou` computes only the room macro-IoU component of the full room
metric suite and is accepted only when its checkpoint behavior matches the full
evaluator.

## Current Evidence

The two-seed room-adapter screen uses eight checkpoints per seed and 64
validation cases. Three room proxies pass every check:

- `fast_room_macro_iou` -> full room macro IoU;
- `dense_predicted_count_abs_error` -> canonical count MAE;
- `fast_instance_type_accuracy` -> canonical matched-instance type accuracy.

All three have Spearman/Pearson near `1.0`, tolerant top-checkpoint agreement
`1.0`, zero false-pass/false-fail, and sensitivity `1.0`.

These proxies are accepted for room checkpoint selection only. They do not
validate wall, opening, furniture, graph, repair, or editing atoms.

Source:

- `outputs/reports/atom_proxy_measurements_roomadapter128_twoseed_v1.json`
- `outputs/reports/atom_proxy_real_consistency_v1.json`
- `outputs/train/atom_proxy_roomadapter_screen128_cuda0_v3/atom_proxy_rows.jsonl`
- `outputs/train/atom_proxy_roomadapter_screen128_cuda0_seed23_v1/atom_proxy_rows.jsonl`
