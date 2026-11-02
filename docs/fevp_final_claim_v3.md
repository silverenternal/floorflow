# Final FEVP Claim v3

## Problem

Editing quality metrics do not by themselves establish whether an edit is auditable as local.

## Claim

Floorplan editing has lacked a reusable, attack-tested audit of whether an edit is actually local. FEVP v1.0-hardened exposes this problem by turning locality into three-layer outside persistence, continuous violation rate, negative controls, deterministic protocol attacks, and paired statistics. FloorFlow is the only complete row in the evaluated protocol matrix: across 12,395 five-seed rows and 7,345 single-room controls, strict pixel/room/label/polygon/edge exactness is 1.0 and violation rate is 0.0, whereas removing edit gating lowers room exactness to 0.4863 and raises violation rate to 0.0139. Typed routing is on the audited locality-quality frontier and dominates the graphless arm on Delta-IoU, Node F1, Edge F1, and fixed-evaluator GED. HouseMind addresses unified tokenization and multimodal instruction editing; its public evaluation does not report outside exactness, violation rate, negative controls, protocol attacks, or paired locality statistics. We do not claim to beat HouseMind. We claim that FEVP exposes an audit layer missing from the public comparison.

## 中文

平面图编辑长期缺少可复用、可攻击测试的局部性审计。FEVP v1.0-hardened 把这个缺口显式化：用三层 outside 持久性、连续 violation rate、负控制、确定性协议攻击和配对统计来定义“编辑是否真的局部”。FloorFlow 在所评估协议矩阵中是唯一完整行：12,395 个五 seed rows 和 7,345 个单房间负控制中，strict 五层 exactness 为 1.0、violation rate 为 0；移除 edit gating 后 room exactness 降至 0.4863，violation rate 升至 0.0139。Typed routing 位于 locality-quality frontier，并在 Delta-IoU、Node F1、Edge F1 与固定 evaluator GED 上优于 graphless arm。HouseMind 解决 tokenization 与多模态指令编辑；其公开评测未报告 outside exact、violation rate、负控制、协议攻击或局部性配对统计。我们不声称击败 HouseMind，而声称 FEVP 补上了公开比较中缺失的审计层。

## HouseMind Boundary

We do not claim that FloorFlow beats HouseMind or that HouseMind lacks locality. We claim that the public HouseMind evaluation does not expose the corresponding audit evidence, so no matched locality ranking is possible.

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
