# Ablation evidence

All rows are validation-only model-selection evidence unless explicitly marked
test. No row uses best-of-N.

| Comparison | Setting | Room macro IoU | Instance room F1 | Source |
|---|---|---:|---:|---|
| Attention at full control | MHA | 0.09701 | 0.05969 | `outputs/reports/a1_mita_ablation.json` |
| Attention at full control | MiTA | 0.10228 | 0.06806 | `outputs/reports/a1_mita_ablation.json` |

## Recursive graph and geometry/semantic gate (val64)

| Candidate | Room IoU | Instance IoU | Type accuracy | Polygon F1 | Balanced CAD |
|---|---:|---:|---:|---:|---:|
| Recursive control | 0.13559 | 0.50584 | 0.47161 | 0.16465 | 0.31986 |
| Ungated graph | 0.12803 | 0.52620 | 0.44055 | 0.16425 | 0.31800 |
| Gated graph, tau=0.2 | 0.13436 | 0.52701 | 0.46901 | 0.17220 | 0.32257 |

The gate was fixed on val64 before the final test run. It routes graph
confidence into geometry while assigning each graph component a class from
the base semantic head. Frozen test503 is reported separately and is not used
for selection.

## Full-data curriculum comparison

Equal-budget control and contour-inward/stage training improved 2 metrics and
regressed 5. Throughput was `34.43` samples/s for control and `28.05` samples/s
for treatment. The curriculum was rejected; evidence is preserved in
`outputs/reports/inward_curriculum_comparison_v2.json`.

## Dynamic-mask sensitivity

| Coarse-control mask ratio | MiTA room IoU | MHA room IoU | MiTA room F1 | MHA room F1 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.69488 | 0.68690 | 0.18652 | 0.18854 |
| 0.25 | 0.17949 | 0.18428 | 0.05652 | 0.04963 |
| 0.50 | 0.05402 | 0.06474 | 0.02844 | 0.02051 |
| 0.75 | 0.03502 | 0.04824 | 0.02262 | 0.01265 |
| 1.00 | 0.03369 | 0.04762 | 0.02368 | 0.01291 |

Source: `outputs/reports/c3_attention_ablation.json`. MiTA helps the
fully-controlled refinement regime, but its advantage is not monotonic under
heavy masking. This limitation must remain explicit in the paper.

## Style-learning progression

| Stage | Conditions | Samples/condition | Quality mean | Coverage mean | Source |
|---|---:|---:|---:|---:|---|
| Style-semantic warm start | 8 | 2 | 0.03050 | 0.00000 | `outputs/reports/distribution_style_semantic_sweep_val8.json` |
| R94 style prompt, val64 | 64 | 1 | 0.09479 | 0.80469 | `outputs/reports/distribution_style_prompt_r94_v2_all_styles_val64.json` |
| R94 style prompt, test503 | 503 | 1 | 0.10316 | 0.72954 | `outputs/reports/distribution_style_prompt_r94_v2_all_styles_test503.json` |

The val/test rows are distributional diagnostics, not checkpoint-selection
evidence. Test was accessed only after the candidate gate and manifest were
frozen.

## Distribution decoder and coupling

| Candidate | Training/decoding change | Val64 room IoU | Coverage | Relative gain vs flow |
|---|---|---:|---:|---:|
| R94 flow | four-step CFM | 0.08726 | 0.85938 | — |
| Full data, 2 epochs | longer training | 0.06598 | 0.81250 | -24.38% |
| Element blend 0.5 | element→CFM bridge active | 0.08959 | 0.90625 | +2.67% |
| Dynamic mask + OT | masked source, multi-stage OT | 0.08831 | 0.89063 | +1.20% |
| Dynamic mask + VP | masked source, VP matcher | 0.08740 | 0.89063 | +0.16% |
| Room-loss rebalance | room IoU 1.0, Tversky 0.5, clip 50 | 0.08709 | 0.87500 | -0.20% |
| Distribution overlay | pixel/element + train-only retrieval, weight 0.5 | 0.10138 | 0.91811 | +16.18% |
| Generated dynamic head | sample-generated kernels, one 1024-case epoch | 0.10486 | 0.92693 | +20.17% |
| Query dynamic head | learned query budget, one 1024-case epoch | 0.10622 | 0.92224 | +21.77% |
| Fixed head | retained target-free baseline | 0.10716 | 0.91809 | +22.80% |
| DMD dynamic adversarial, 1024 | conditional dynamic-mask discriminator | 0.10034 | 0.91322 | +15.00% |
| Fixed + adversarial ensemble | adversarial weight 0.25 | 0.10763 | 0.91322 | +23.77% |

The selected branch is therefore the previously underused deterministic
distribution/overlay decoder, not a larger flow model. On all eight styles and
a 300-case validation holdout it reaches mean Room IoU `0.13403`, worst
`0.13132`, coverage `0.92677`, and zero generated pixels outside the footprint.
Room-conditioned style disagreement is `0.18076`; unrestricted pixel
disagreement is diluted by background and is `0.03490`.

### Retrieval fusion sweep

| Retrieval weight | Val64 Room IoU | Instance F1 | Style type accuracy |
|---:|---:|---:|---:|
| 0.25 | 0.09833 | 0.12499 | 0.25495 |
| 0.50 | 0.10138 | 0.10918 | 0.26406 |
| 0.75 | 0.10262 | 0.10744 | 0.28776 |
| 1.00 | 0.10715 | 0.09275 | 0.27214 |
| 1.50 | 0.10262 | 0.11378 | 0.26302 |
| 2.00 | 0.09908 | 0.12500 | 0.27344 |
| 4.00 | 0.09775 | 0.13751 | 0.25547 |

Weight 0.5 is selected to balance Room IoU, style separation, and instance F1;
the test split is not used for this choice. One retrieved neighbor is best for
Room IoU (`0.10138`); top-k 8 improves instance F1 to `0.12040` but lowers Room
IoU to `0.10024`.

## Self-predicted controls

The full-data stage-2 refiners were trained on 5,126 target-free distribution
controls. Opening IoU improves from `0.00944` to `0.19078`, furniture IoU from
`0.00767` to `0.11749`, and support IoU reaches `0.97598`. Raw-union-refined
wall preserves wall IoU within `0.00854` of the original room decoder. The
target-derived control result (`0.97383` Room IoU) is an oracle diagnostic only
and is excluded from the target-free main result.

## Dynamic adversarial training

The DMD-style discriminator masks feature channels according to drift from an
EMA reference. On the same 1,024-case protocol, adversarial flow improves Room
IoU from `0.08726` to `0.10034`. The full 5,126-case run reaches `0.09928` and
improves the CAD balanced score to `0.26996`. On frozen test503, the adversarial
overlay improves opening IoU to `0.17202` and furniture IoU to `0.07310`; adding
self-control refiners raises them to `0.20481` and `0.12627`.
