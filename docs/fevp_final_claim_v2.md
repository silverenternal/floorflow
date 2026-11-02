# Final FEVP Claim v2

> Historical document, superseded by `docs/fevp_final_claim_v4.md`. It is kept
> for provenance only and is excluded from the canonical submission package.

## Canonical Claim

Under FEVP v1.0-hardened, FloorFlow is the only method in the evaluated protocol matrix that jointly reports strict locality, topology safety, structure consistency, edit fidelity, protocol attacks, and paired statistics. Across 12,395 five-seed rows and 7,345 single-room negative controls, the strict arm produced no auditable outside leakage: pixel, room-level, label, polygon, and edge exactness were 1.0 and locality violation rate was 0.0. Removing edit gating reduced room-level exactness to 0.4863 and increased violation rate to 0.0139. Typed routing improves Node F1, edge overlap, and fixed-evaluator GED over dense, no-gate, and graphless arms; paired Wilcoxon p-values, rank-biserial correlations, and Cliff's deltas are reported in `fevp_graph_metrics_v2.json`. Validity is reported as a five-check safety baseline, and the watertight guard is explicitly not claimed as a net benefit.

## 中文 Claim

在 FEVP v1.0-hardened 下，FloorFlow 在所评估协议矩阵中唯一同时报告严格局部性、拓扑安全基线、结构一致性、编辑保真度、协议攻击与配对统计证据。12,395 个五 seed 样本和 7,345 个单房间负控制中，严格臂无任何可审计的区外泄漏；移除 edit gating 后 room-level exact 降至 0.4863，violation rate 升至 0.0139。typed routing 在 Node F1、edge overlap 与固定 evaluator GED 上相对 dense、no-gate、graphless 的配对 Wilcoxon p 值、rank-biserial correlation 与 Cliff's delta 均记录在 `fevp_graph_metrics_v2.json`。validity 仅作为五项安全基线，watertight guard 不声称净收益。

## Forbidden Claims

- The watertight guard provides a net improvement.
- Validity is 1.0 under the hardened five-check definition.
- FloorFlow beats HouseMind-E on its complete method or RPLAN leaderboard.
- FloorFlow beats CE2EPlan absolute GED or forms one RPLAN/CubiCasa leaderboard.
- FloorFlow beats Raster2Seq on a matched 503-case identity ranking.
- Delta-IoU is presented as defeating HouseMind.
- Five seeds are population-level inference.
- A protocol attack validates model quality rather than evaluator sensitivity.

## Source Discipline

- Every quantitative sentence is backed by a machine-readable report.
- External methods remain separate and are not ranked on missing FEVP cells.
- Protocol attacks test evaluator sensitivity, not model quality.
