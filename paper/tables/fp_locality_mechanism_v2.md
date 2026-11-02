# Locality-Quality Mechanism v2

Status: aggregate mechanism table. The current per-operation submission result
is `residential_sota_main_v5.md`; do not replace the main range with the
aggregate `0.987084` reference.

| Arm | Outside exact | Contract failures | Delta-IoU | Watertight | Node F1 | Edge | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed_edit_graph | 1.000000 | 0/12395 | 0.937473 | 0.966841 | 0.955068 | 0.837563 | 2.121823 |
| dense_cross_attention | 0.999545 | 2787/12395 | 0.887741 | 0.967084 | 0.953797 | 0.833553 | 2.186123 |
| no_edit_gating | 0.986115 | 7553/12395 | 0.853715 | 0.968132 | 0.949057 | 0.818217 | 2.448891 |
| no_structural_graph | 1.000000 | 0/12395 | 0.888817 | 0.967164 | 0.953819 | 0.833711 | 2.182896 |

Full control-gated FloorFlow reference: outside exact 1.000, Delta-IoU 0.987084.

The average outside-exact gap is categorical under FEVP: every case below 1.0 is a contract failure, not a small quality loss.
