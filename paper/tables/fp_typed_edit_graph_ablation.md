# Typed Edit-Graph Ablation

| Arm | Outside exact | Outside room | ΔIoU | ΔMSE | Watertight | Node F1 | Edge | GED | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| typed_edit_graph | 1.000000 | 1.000000 | 0.939087 | 0.003448087 | 0.963292 | 0.955515 | 0.839107 | 2.098023 | 3.128 |
| dense_cross_attention | 0.999680 | 0.999601 | 0.897817 | 0.005067047 | 0.963695 | 0.953877 | 0.835177 | 2.163776 | 1.055 |
| no_edit_gating | 0.989987 | 0.987453 | 0.866409 | 0.010374545 | 0.964098 | 0.950850 | 0.825788 | 2.306575 | 0.273 |
| no_structural_graph | 1.000000 | 1.000000 | 0.898552 | 0.004912629 | 0.963695 | 0.953877 | 0.835116 | 2.164179 | 0.262 |

The four arms use identical cases, checkpoints, edit masks, and graph scorer.
`no_edit_gating` accepts raw output over the full footprint; `dense_cross_attention` expands the edit region by two pixels; `no_structural_graph` accepts all channels inside the declared region; `typed_edit_graph` shuts weak connections and admits only semantic families induced by the edit.

Hypothesis status: **supported_with_topology_tradeoff**.

The typed graph should preserve outside exactness, improve Delta-IoU and graph consistency over dense and graphless routing, and show no statistically supported watertightness loss. A lower point estimate with a paired interval crossing zero is reported as a topology trade-off, not as evidence of harm.
