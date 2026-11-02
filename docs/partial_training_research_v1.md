# Partial-Training Research v1

Date: 2026-11-01

## Papers

| Work | arXiv | Relevant idea | Adopted here |
|---|---|---|---|
| LoRA | `2106.09685` | low-rank trainable weight residuals | Q/KV and output residuals for cross-attention |
| AdaLoRA | `2303.10512` | adaptive rank allocation | trainable rank gates with a fixed expected rank budget |
| AdaptFormer | `2205.13535` | frozen backbone with zero-initialized bottleneck adapter | zero-initialized semantic-family query adapters |
| SSF | `2210.08823` | per-channel scale and shift | per-family query scale/shift |
| BitFit | `2106.10199` | train bias vectors only | `bias_only` scope |
| Conv-Adapter | `2208.07463` | lightweight convolutional adapters for frozen ConvNets | zero-initialized depthwise bottleneck spatial adapter |

## Cloned References

These repositories are local references only and are not imported into the
installed package:

| Repository | Commit | License |
|---|---|---|
| `huggingface/peft` | `03412afcf1e96f74a014e467a313b22baeb74498` | Apache-2.0 |
| `microsoft/LoRA` | `c4593f060e6a368d7bb5af5273b8e42810cdef90` | MIT |
| `ShoufaChen/AdaptFormer` | `6967d676c1a5e5a11be2e2768a6e5c604bb043ed` | MIT |
| `dongzelian/SSF` | `e94e0e704a4ece1986a537c97a95158b46838f71` | MIT |

## Applied Design

### Cross-Attention LoRA

The base cross-attention projections stay frozen. `CrossAttentionLoRA` adds:

```text
Q' = Q + Bq(Aq(Q))
KV' = KV + Bkv(Akv(KV))
O' = O + Bo(Ao(O))
```

The up projections are zero-initialized, so enabling the module is initially an
exact no-op. Only 18,435 parameters train in `cross_attention_lora`.

### Channel-Family SSF and Adapter

Every semantic family owns independent:

- per-channel scale;
- per-channel shift;
- gated query delta;
- source-dependent gate.

Room/wall/opening/furniture therefore do not share a trainable query slice.

### BitFit Control

`bias_only` provides a strict baseline that conditions only bias vectors, with
39,802 trainable elements.

## Adaptive Rank and Spatial Adaptation

Cross-attention LoRA contains independent query, KV, and output rank gates. The
training gate keeps a fixed expected rank budget; evaluation applies hard top-k
rank selection. `lora_ranks` and `adalora` expose rank-only and full adaptive
rank scopes.

The Conv-Adapter scope adds a depthwise bottleneck residual with a
zero-initialized output projection. It trains only 4,401 elements.

External PEFT code remains a reference rather than an import dependency, avoiding
checkpoint-format and transitive dependency coupling.
