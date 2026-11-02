#!/usr/bin/env python3
"""Build the final offensive FEVP claim with explicit forbidden overclaims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_final_claim_v2.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fevp_final_claim_v2.md",
    )
    return parser.parse_args()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    locality = _read(ROOT / "outputs/reports/fevp_locality_claim_v2.json")
    pareto = _read(ROOT / "outputs/reports/fevp_mechanism_pareto_v2.json")
    safety = _read(ROOT / "outputs/reports/fevp_safety_baseline_v2.json")
    strength = _read(ROOT / "outputs/reports/fevp_edit_strength_v2.json")
    graph = _read(ROOT / "outputs/reports/fevp_graph_metrics_v2.json")
    stability = _read(ROOT / "outputs/reports/fevp_fiveseed_stability_v2.json")
    attacks = _read(ROOT / "outputs/reports/fevp_protocol_attacks_compact_v2.json")
    typed = graph["summary"]["seed_mean"]["typed_edit_graph"]
    no_gate = locality["no_edit_gating"]
    strict = locality["strict"]
    claim_en = (
        "Under FEVP v1.0-hardened, FloorFlow is the only method in the evaluated "
        "protocol matrix that jointly reports strict locality, topology safety, "
        "structure consistency, edit fidelity, protocol attacks, and paired "
        f"statistics. Across {locality['cases']:,} five-seed rows and "
        f"{locality['single_room_negative_controls']:,} single-room negative "
        "controls, the strict arm produced no auditable outside leakage: pixel, "
        "room-level, label, polygon, and edge exactness were 1.0 and locality "
        "violation rate was 0.0. Removing edit gating reduced room-level "
        f"exactness to {no_gate['outside_room_level_exact']:.4f} and increased "
        f"violation rate to {no_gate['locality_violation_rate']:.4f}. Typed "
        "routing improves Node F1, edge overlap, and fixed-evaluator GED over "
        "dense, no-gate, and graphless arms with five-seed paired Wilcoxon "
        "significance. Validity is reported as a five-check safety baseline, and "
        "the watertight guard is explicitly not claimed as a net benefit."
    )
    claim_zh = (
        "在 FEVP v1.0-hardened 下，FloorFlow 在所评估协议矩阵中唯一同时报告严格局部性、"
        "拓扑安全基线、结构一致性、编辑保真度、协议攻击与配对统计证据。"
        f"{locality['cases']:,} 个五 seed 样本和 "
        f"{locality['single_room_negative_controls']:,} 个单房间负控制中，"
        "严格臂无任何可审计的区外泄漏；移除 edit gating 后 room-level exact 降至 "
        f"{no_gate['outside_room_level_exact']:.4f}，violation rate 升至 "
        f"{no_gate['locality_violation_rate']:.4f}。typed routing 在 Node F1、"
        "edge overlap 与固定 evaluator GED 上相对 dense、no-gate、graphless "
        "均取得五 seed 配对显著性。validity 仅作为五项安全基线，watertight guard "
        "不声称净收益。"
    )
    forbidden = [
        "The watertight guard provides a net improvement.",
        "Validity is 1.0 under the hardened five-check definition.",
        "FloorFlow beats HouseMind-E on its complete method or RPLAN leaderboard.",
        "FloorFlow beats CE2EPlan absolute GED or forms one RPLAN/CubiCasa leaderboard.",
        "FloorFlow beats Raster2Seq on a matched 503-case identity ranking.",
        "Delta-IoU is presented as defeating HouseMind.",
        "Five seeds are population-level inference.",
        "A protocol attack validates model quality rather than evaluator sensitivity.",
    ]
    sources = {
        sentence: source
        for sentence, source in (
            (
                "strict locality and no-gate counter-evidence",
                "outputs/reports/fevp_locality_claim_v2.json",
            ),
            (
                "typed mechanism Pareto",
                "outputs/reports/fevp_mechanism_pareto_v2.json",
            ),
            (
                "validity and watertight safety boundary",
                "outputs/reports/fevp_safety_baseline_v2.json",
            ),
            (
                "edit-strength fidelity",
                "outputs/reports/fevp_edit_strength_v2.json",
            ),
            (
                "graph and GED paired tests",
                "outputs/reports/fevp_graph_metrics_v2.json",
            ),
            (
                "five-seed stability",
                "outputs/reports/fevp_fiveseed_stability_v2.json",
            ),
            (
                "protocol attack sensitivity",
                "outputs/reports/fevp_protocol_attacks_compact_v2.json",
            ),
        )
    }
    checks = {
        "strict_five_layer_exact": all(
            strict[key] == 1.0
            for key in (
                "outside_pixel_exact",
                "outside_room_level_exact",
                "outside_room_label_exact",
                "outside_room_polygon_exact",
                "outside_edge_exact",
            )
        ),
        "strict_violation_zero": strict["locality_violation_rate"] == 0.0,
        "no_gate_counter_evidence": (
            no_gate["outside_room_level_exact"] < 0.5
            and no_gate["locality_violation_rate"] > 0.0
        ),
        "typed_on_pareto_frontier": pareto["checks"]["typed_on_frontier"],
        "guard_not_promoted": safety["checks"]["guard_not_promoted"],
        "edit_strength_complete": strength["status"] == "complete",
        "typed_ged_paired_tests_complete": graph["checks"][
            "paired_tests_complete"
        ],
        "five_seed_complete": stability["status"] == "complete",
        "compact_attacks_all_pass": all(
            row["sensitive"] for row in attacks["attacks"]
        ),
        "typed_node_f1_reported": 0.0 < typed["node_f1"] <= 1.0,
        "typed_edge_f1_reported": 0.0 < typed["edge_f1"] <= 1.0,
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "claim_en": claim_en,
        "claim_zh": claim_zh,
        "protocol": "FEVP v1.0-hardened",
        "evaluated_matrix_only": True,
        "checks": checks,
        "forbidden_claims": forbidden,
        "sources": sources,
        "evidence": {
            "locality": locality,
            "pareto": pareto,
            "safety": safety,
            "edit_strength": strength,
            "graph_seed_mean_typed": typed,
            "stability": stability["metrics"],
            "attacks": attacks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    doc = [
        "# Final FEVP Claim v2",
        "",
        "## Canonical Claim",
        "",
        claim_en,
        "",
        "## 中文 Claim",
        "",
        claim_zh,
        "",
        "## Forbidden Claims",
        "",
        *[f"- {claim}" for claim in forbidden],
        "",
        "## Source Discipline",
        "",
        "- Every quantitative sentence is backed by a machine-readable report.",
        "- External methods remain separate and are not ranked on missing FEVP cells.",
        "- Protocol attacks test evaluator sensitivity, not model quality.",
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
