# Main Mechanism and Compute Ablation v1

Status: frozen candidate over the declared FEVP aggregate row set.

| Arm | Outside pixel | Outside room | Violation | Delta-IoU | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Typed routing | 1.000000 | 1.000000 | 0.000000 | 0.937473 | 0.955068 | 0.890007 | 2.120855 |
| Dense cross-attention | 0.999545 | 0.999470 | 0.000455 | 0.887741 | 0.953797 | 0.887080 | 2.185478 |
| No edit gating | 0.986115 | 0.486325 | 0.013885 | 0.853715 | 0.949057 | 0.874121 | 2.448649 |
| No structural graph | 1.000000 | 1.000000 | 0.000000 | 0.888817 | 0.953819 | 0.887200 | 2.182251 |

At matched observable compute with no added learned parameters, typed routing
raises Delta-IoU by `+0.048656` (`[0.045549, 0.051844]`) and lowers GED by
`0.061396` (`[-0.070432, -0.052844]`).

Typed routing consumes more latency. It is not an efficiency contribution.
