#!/usr/bin/env python3
"""Reframe FEVP as exposing the locality-auditability gap."""

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
        default=ROOT / "outputs/reports/fevp_final_claim_v3.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fevp_final_claim_v3.md",
    )
    return parser.parse_args()


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    locality = _read("outputs/reports/fevp_locality_claim_v2.json")
    house = _read("outputs/reports/fevp_housemind_positioning_v2.json")
    pareto = _read("outputs/reports/fevp_mechanism_pareto_v3.json")
    evidence_map = _read("outputs/reports/fevp_audit_evidence_map_v2.json")
    deletion = _read("outputs/reports/fevp_room_deletion_diagnosis_v2.json")
    safety = _read("outputs/reports/fevp_safety_baseline_v2.json")
    coverage = _read("outputs/reports/fevp_coverage_matrix_v1.json")
    claim_en = (
        "Floorplan editing has lacked a reusable, attack-tested audit of whether "
        "an edit is actually local. FEVP v1.0-hardened exposes this problem by "
        "turning locality into three-layer outside persistence, continuous "
        "violation rate, negative controls, deterministic protocol attacks, and "
        "paired statistics. FloorFlow is the only complete row in the evaluated "
        f"protocol matrix: across {locality['cases']:,} five-seed rows and "
        f"{locality['single_room_negative_controls']:,} single-room controls, "
        "strict pixel/room/label/polygon/edge exactness is 1.0 and violation rate "
        f"is 0.0, whereas removing edit gating lowers room exactness to "
        f"{locality['no_edit_gating']['outside_room_level_exact']:.4f} and raises "
        f"violation rate to {locality['no_edit_gating']['locality_violation_rate']:.4f}. "
        "Typed routing is on the audited locality-quality frontier and dominates "
        "the graphless arm on Delta-IoU, Node F1, Edge F1, and fixed-evaluator "
        "GED. HouseMind addresses unified tokenization and multimodal instruction "
        "editing; its public evaluation does not report outside exactness, "
        "violation rate, negative controls, protocol attacks, or paired locality "
        "statistics. We do not claim to beat HouseMind. We claim that FEVP exposes "
        "an audit layer missing from the public comparison."
    )
    claim_zh = (
        "平面图编辑长期缺少可复用、可攻击测试的局部性审计。FEVP v1.0-hardened 把这个缺口显式化："
        "用三层 outside 持久性、连续 violation rate、负控制、确定性协议攻击和配对统计来定义"
        "“编辑是否真的局部”。FloorFlow 在所评估协议矩阵中是唯一完整行："
        f"{locality['cases']:,} 个五 seed rows 和 "
        f"{locality['single_room_negative_controls']:,} 个单房间负控制中，"
        "strict 五层 exactness 为 1.0、violation rate 为 0；移除 edit gating 后 "
        f"room exactness 降至 {locality['no_edit_gating']['outside_room_level_exact']:.4f}，"
        f"violation rate 升至 {locality['no_edit_gating']['locality_violation_rate']:.4f}。"
        "Typed routing 位于 locality-quality frontier，并在 Delta-IoU、Node F1、Edge F1 "
        "与固定 evaluator GED 上优于 graphless arm。HouseMind 解决 tokenization 与多模态指令编辑；"
        "其公开评测未报告 outside exact、violation rate、负控制、协议攻击或局部性配对统计。"
        "我们不声称击败 HouseMind，而声称 FEVP 补上了公开比较中缺失的审计层。"
    )
    forbidden = [
        "FloorFlow beats HouseMind or HouseMind-E on its complete method or benchmark.",
        "HouseMind lacks locality because its public metrics omit the FEVP audit.",
        "A dash for HouseMind is a zero or a failure.",
        "HouseMind-G is an editing baseline.",
        "The watertight guard provides a net improvement.",
        "Hardened validity is 1.0.",
        "CE2EPlan absolute GED is aligned with FEVP.",
        "Raster2Seq forms a 509-vs-503 identity ranking with FloorFlow.",
        "Delta-IoU is a defeat of HouseMind.",
        "Five seeds are population-level inference.",
    ]
    checks = {
        "claim_starts_from_auditability_gap": claim_en.startswith(
            "Floorplan editing has lacked"
        ),
        "house_mind_acknowledged": "HouseMind addresses unified tokenization" in claim_en,
        "no_house_mind_defeat": "do not claim to beat HouseMind" in claim_en,
        "locality_counter_evidence": locality["checks"]["no_gate_counter_evidence"],
        "typed_frontier": pareto["checks"]["typed_frontier"],
        "typed_dominates_graphless": pareto["checks"][
            "typed_dominates_graphless_on_all_reported_quality_metrics"
        ],
        "evidence_map_complete": evidence_map["checks"]["floorflow_complete"],
        "coverage_matrix_complete_row": coverage["claims"][
            "floorflow_complete_under_fevp"
        ],
        "house_mind_context_correct": house["checks"][
            "generation_variant_not_ranked"
        ],
        "room_deletion_diagnosed": deletion["status"] == "complete",
        "guard_not_promoted": safety["checks"]["guard_not_promoted"],
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "claim_en": claim_en,
        "claim_zh": claim_zh,
        "problem_statement": (
            "Editing quality metrics do not by themselves establish whether an "
            "edit is auditable as local."
        ),
        "contribution": (
            "FEVP exposes and operationalizes the locality-auditability gap."
        ),
        "house_mind_positioning": house["non_ranking_statement"],
        "checks": checks,
        "forbidden_claims": forbidden,
        "sources": {
            "locality": "outputs/reports/fevp_locality_claim_v2.json",
            "house_mind": "outputs/reports/fevp_housemind_positioning_v2.json",
            "pareto": "outputs/reports/fevp_mechanism_pareto_v3.json",
            "audit_map": "outputs/reports/fevp_audit_evidence_map_v2.json",
            "coverage_matrix": "outputs/reports/fevp_coverage_matrix_v1.json",
            "room_deletion": (
                "outputs/reports/fevp_room_deletion_diagnosis_v2.json"
            ),
            "safety": "outputs/reports/fevp_safety_baseline_v2.json",
        },
        "evidence": {
            "locality": locality,
            "house_mind": house,
            "pareto": pareto,
            "audit_map": evidence_map,
            "coverage_matrix": coverage,
            "room_deletion": deletion,
            "safety": safety,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    doc = [
        "# Final FEVP Claim v3",
        "",
        "## Problem",
        "",
        report["problem_statement"],
        "",
        "## Claim",
        "",
        claim_en,
        "",
        "## 中文",
        "",
        claim_zh,
        "",
        "## HouseMind Boundary",
        "",
        house["non_ranking_statement"],
        "",
        "## Forbidden Claims",
        "",
        *[f"- {claim}" for claim in forbidden],
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
