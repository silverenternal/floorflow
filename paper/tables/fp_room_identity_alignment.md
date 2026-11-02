# Room Identity Alignment Diagnostic

| Readout | Node F1 | Edge overlap | Pred nodes | Target nodes |
|---|---:|---:|---:|---:|
| wall_wall_primary | 0.391224 | 0.333890 | 11.670 | 5.606 |
| direct_raw | 0.554042 | 0.125992 | 7.232 | 9.120 |
| direct_tiled | 0.837135 | 0.657761 | 6.687 | 8.303 |
| direct_tiled_geometry | 0.844104 | 0.661599 | 6.687 | 8.303 |
| direct_tiled_no_undefined | 0.837135 | 0.657761 | 6.687 | 8.303 |
| direct_tiled_purity_ge_0_5 | 0.848212 | 0.665116 | 6.448 | 8.087 |

| Candidate delta vs primary | Node F1 | 95% CI | Edge overlap | 95% CI |
|---|---:|---:|---:|---:|
| direct_raw | +0.162819 | [+0.150817, +0.174465] | -0.207898 | [-0.223704, -0.192101] |
| direct_tiled | +0.445911 | [+0.435434, +0.456249] | +0.323872 | [+0.307679, +0.339598] |
| direct_tiled_geometry | +0.452881 | [+0.442410, +0.463137] | +0.327709 | [+0.311432, +0.343449] |
| direct_tiled_no_undefined | +0.445911 | [+0.435434, +0.456249] | +0.323872 | [+0.307679, +0.339598] |
| direct_tiled_purity_ge_0_5 | +0.456989 | [+0.446535, +0.467427] | +0.331226 | [+0.314759, +0.346761] |

This is a graph-readout and post-hoc alignment audit. Pixels, labels, outside exactness, validity, and watertightness are unchanged.
