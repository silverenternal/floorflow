# Trinity d192 Gap Decomposition

Scope: matched FloorSet-Lite official-100. The predeclared comparison is an appendix anchor; the three-seed result is primary.

| Component | Candidate | Trinity | Relative gap | Paired 95% CI |
|---|---:|---:|---:|---:|
| b2b connectivity | 0.05784 | 0.04088 | +41.50% | [+7.14%, +80.30%] |
| area outline | 0.12509 | 0.10021 | +24.83% | [-6.76%, +59.50%] |
| grouping | 1.01667 | 0.76333 | +33.19% | [+11.79%, +55.90%] |
| p2b port | 0.28202 | 0.36285 | -22.28% | [-83.85%, +36.57%] |
| boundary | 1.37000 | 1.25000 | +9.60% | [-8.27%, +29.33%] |
| mib | 0.00000 | 0.00000 | n/a | [n/a, n/a] |
| rho aspect | 0.30484 | 0.27481 | +10.92% | [+6.42%, +15.86%] |

## Headline Reconciliation

| Comparison | Cost gap | V_rel gap |
|---|---:|---:|
| Predeclared anchor | +5.76% | +30.45% |
| Component-row source | +6.30% | +30.03% |
| Candidate source residual | +0.51% | -0.32% |

The predeclared +5.76%/+30.45% anchor remains frozen. Component rows use a slightly different candidate summary; the disclosed sub-1% source residual is not allocated to a model component.

## V_rel Attribution

| Slots | Predeclared share | 3-seed share |
|---|---:|---:|
| grouping | 57.89% | 63.56% |
| boundary | 42.11% | 36.44% |
| mib | 0.00% | 0.00% |

## Hard-Cost Split

Using `ALPHA=0.5` and `BETA=2.0`, `log(cost)` separates exactly into geometry and soft-constraint terms.

| Term | Candidate | Trinity | Log gap | Share |
|---|---:|---:|---:|---:|
| Geometry | 0.09090 | 0.07039 | 0.02051 | 57.99% |
| Soft constraints | 0.10205 | 0.08719 | 0.01486 | 42.01% |

The remaining gap is structural (b2b and grouping) and geometric (area/outline and rho/aspect). The candidate retains an advantage on p2b/port HPWL. MIB has no official-100 signal, and exact-area-decoder/training attribution remains unresolved.

Figure: `paper/figures/chip_trinity_gap_decomposition.svg`.
