# Final FEVP Claim v4

Status date: 2026-11-02. This remains the FEVP claim document; the current
oracle-control result summary is
`docs/fp_oracle_control_protocol_status_v1.md`.

Floorplan editing has lacked a reusable, attack-tested audit of whether an edit is actually local. FEVP v1.0-hardened exposes this problem by turning locality into three-layer outside persistence, continuous violation rate, negative controls, deterministic protocol attacks, and paired statistics. FloorFlow is the only complete row in the evaluated protocol matrix: across 12,395 five-seed rows and 7,345 single-room controls, strict pixel/room/label/polygon/edge exactness is 1.0 and violation rate is 0.0, whereas removing edit gating lowers room exactness to 0.4863 and raises violation rate to 0.0139. Typed routing is on the audited locality-quality frontier and dominates the graphless arm on Delta-IoU, Node F1, Edge F1, and fixed-evaluator GED. HouseMind addresses unified tokenization and multimodal instruction editing; its public evaluation does not report outside exactness, violation rate, negative controls, protocol attacks, or paired locality statistics. We do not claim to beat HouseMind. We claim that FEVP exposes an audit layer missing from the public comparison. The same FEVP evaluator was applied to 200 public CE2EPlan generation rows without a FloorFlow-specific scorer. It decomposed five-check validity (0.9600), watertightness (1.0000), Node F1 (0.9479), Edge F1 (0.7703), and GED (2.9750); outside exact is not applicable to a generation-only task. On 400 real matched rows organized as adversarial edit-prompt proxies, typed routing retained outside exactness 1.0 and violation rate 0.0 with Delta-IoU 0.9305; no-gate violation rose to 0.0153. Typed GED 2.6225 was lower than dense 2.6900, no-gate 2.9325, and graphless 2.6900.

## 中文

平面图编辑长期缺少可复用、可攻击测试的局部性审计。FEVP v1.0-hardened 把这个缺口显式化：用三层 outside 持久性、连续 violation rate、负控制、确定性协议攻击和配对统计来定义“编辑是否真的局部”。FloorFlow 在所评估协议矩阵中是唯一完整行：12,395 个五 seed rows 和 7,345 个单房间负控制中，strict 五层 exactness 为 1.0、violation rate 为 0；移除 edit gating 后 room exactness 降至 0.4863，violation rate 升至 0.0139。Typed routing 位于 locality-quality frontier，并在 Delta-IoU、Node F1、Edge F1 与固定 evaluator GED 上优于 graphless arm。HouseMind 解决 tokenization 与多模态指令编辑；其公开评测未报告 outside exact、violation rate、负控制、协议攻击或局部性配对统计。我们不声称击败 HouseMind，而声称 FEVP 补上了公开比较中缺失的审计层。同一套 FEVP evaluator 已迁移到 200 个 CE2EPlan 公开生成输出，分解得到 validity 0.9600、watertightness 1.0000、Node F1 0.9479、Edge F1 0.7703、GED 2.9750；生成任务不适用 outside exact。在 400 个真实 matched rows 构造的 adversarial prompt proxy 中，typed routing 保持 outside exact 1.0、violation 0.0、Delta-IoU 0.9305；no-gate violation 升至 0.0153，typed GED 2.6225 低于 no-gate 2.9325。

## Current Dual-Metric Anchor

The current oracle-control editing result is stronger than the older aggregate
headline and is reported per operation:

| Operation | Delta-IoU | Delta-MSE |
|---|---:|---:|
| room type | 0.999199 | 0.000069 |
| room add | 0.999801 | 0.000030 |
| room remove | 0.999756 | 0.000019 |
| opening delete | 0.956703 | 0.000096 |
| furniture delete | 0.987812 | 0.000083 |

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019`. FloorFlow
numerically exceeds both anchors for all five operations under aligned metric
definitions. The task inputs and datasets differ, so the result is a scoped
protocol anchor rather than a HouseMind leaderboard rank.

The matching FloorFlow run preserves exact outside pixel, room, polygon, wall,
opening, and unaffected-edge content. FP-Eval validity is `1.0`, conditional
watertightness is `0.9579-0.9675`, and the independent FEVP five-check validity
is `0.9124-0.9569`.

## Portability Boundary

- CE2EPlan is a generation-only portability target.
- HouseMind remains blocked by unreleased inference and machine-readable outputs.
- Adversarial prompt proxies are not a general NLU benchmark.

## Forbidden Claims

- FloorFlow beats HouseMind or HouseMind-E on its complete method or benchmark.
- HouseMind lacks locality because its public metrics omit the FEVP audit.
- A dash for HouseMind is a zero or a failure.
- HouseMind-G is an editing baseline.
- The watertight guard provides a net improvement.
- Hardened validity is 1.0.
- CE2EPlan absolute GED is aligned with FEVP.
- Raster2Seq forms a 509-vs-503 identity ranking with FloorFlow.
- Delta-IoU is a defeat of HouseMind.
- Five seeds are population-level inference.
- CE2EPlan portability is an editing or leaderboard comparison.
- Outside exact is reported for CE2EPlan generation outputs.
- Adversarial prompt proxies are a natural-language understanding benchmark.
- Prompt-proxy evidence proves robustness to arbitrary free-form text.
