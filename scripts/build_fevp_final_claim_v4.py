#!/usr/bin/env python3
"""Add portability and adversarial-prompt robustness to the FEVP claim."""

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
        default=ROOT / "outputs/reports/fevp_final_claim_v4.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fevp_final_claim_v4.md",
    )
    return parser.parse_args()


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    base = _read("outputs/reports/fevp_final_claim_v3.json")
    ce2e = _read("outputs/reports/fevp_ce2eplan_portability_v3.json")
    prompts = _read("outputs/reports/fevp_adversarial_prompts_v3.json")
    core = _read("outputs/reports/fevp_core_audit_coverage_v3.json")
    house_gate = _read("outputs/reports/fevp_housemind_portability_gate_v1.json")
    typed = prompts["summary"]["typed_edit_graph"]
    no_gate = prompts["summary"]["no_edit_gating"]
    dense = prompts["summary"]["dense_cross_attention"]
    graphless = prompts["summary"]["no_structural_graph"]
    portability_claim = (
        f"The same FEVP evaluator was applied to {ce2e['cases']} public "
        "CE2EPlan generation rows without a FloorFlow-specific scorer. It "
        f"decomposed five-check validity ({ce2e['validity']['validity']:.4f}), "
        f"watertightness ({ce2e['watertight']['watertight_rate']:.4f}), Node F1 "
        f"({ce2e['metrics']['node_f1']:.4f}), Edge F1 "
        f"({ce2e['metrics']['edge_f1']:.4f}), and GED "
        f"({ce2e['metrics']['ged']:.4f}); outside exact is not applicable to a "
        "generation-only task."
    )
    prompt_claim = (
        f"On {prompts['prompts']} real matched rows organized as adversarial "
        "edit-prompt proxies, typed routing retained outside exactness 1.0 and "
        f"violation rate 0.0 with Delta-IoU {typed['delta_iou']:.4f}; no-gate "
        f"violation rose to {no_gate['locality_violation_rate']:.4f}. Typed GED "
        f"{typed['ged']:.4f} was lower than dense {dense['ged']:.4f}, no-gate "
        f"{no_gate['ged']:.4f}, and graphless {graphless['ged']:.4f}."
    )
    claim_en = f"{base['claim_en']} {portability_claim} {prompt_claim}"
    claim_zh = (
        f"{base['claim_zh']}"
        f"同一套 FEVP evaluator 已迁移到 {ce2e['cases']} 个 CE2EPlan 公开生成输出，"
        f"分解得到 validity {ce2e['validity']['validity']:.4f}、watertightness "
        f"{ce2e['watertight']['watertight_rate']:.4f}、Node F1 "
        f"{ce2e['metrics']['node_f1']:.4f}、Edge F1 "
        f"{ce2e['metrics']['edge_f1']:.4f}、GED {ce2e['metrics']['ged']:.4f}；"
        "生成任务不适用 outside exact。"
        f"在 {prompts['prompts']} 个真实 matched rows 构造的 adversarial prompt proxy 中，"
        f"typed routing 保持 outside exact 1.0、violation 0.0、Delta-IoU "
        f"{typed['delta_iou']:.4f}；no-gate violation 升至 "
        f"{no_gate['locality_violation_rate']:.4f}，typed GED {typed['ged']:.4f} "
        f"低于 no-gate {no_gate['ged']:.4f}。"
    )
    forbidden = list(base["forbidden_claims"])
    forbidden.extend(
        [
            "CE2EPlan portability is an editing or leaderboard comparison.",
            "Outside exact is reported for CE2EPlan generation outputs.",
            "Adversarial prompt proxies are a natural-language understanding benchmark.",
            "Prompt-proxy evidence proves robustness to arbitrary free-form text.",
        ]
    )
    checks = {
        **base["checks"],
        "core_audit_table_complete": core["checks"]["floorflow_complete"],
        "ce2eplan_100_to_200_rows": ce2e["checks"]["case_count_100_to_200"],
        "ce2eplan_shared_evaluator": ce2e["checks"]["same_evaluator"],
        "ce2eplan_outside_not_claimed": ce2e["checks"][
            "generation_outside_exact_not_claimed"
        ],
        "housemind_portability_gate_explicit": (
            house_gate["status"] == "blocked_external"
        ),
        "adversarial_prompt_count": prompts["checks"][
            "prompt_count_200_to_500"
        ],
        "adversarial_prompt_typed_locality": (
            prompts["checks"]["typed_outside_exact_one"]
            and prompts["checks"]["typed_violation_zero"]
        ),
        "adversarial_prompt_typed_ged": prompts["checks"][
            "typed_ged_lower_than_no_gate"
        ],
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "claim_en": claim_en,
        "claim_zh": claim_zh,
        "portability_claim": portability_claim,
        "prompt_robustness_claim": prompt_claim,
        "checks": checks,
        "forbidden_claims": list(dict.fromkeys(forbidden)),
        "sources": {
            **base["sources"],
            "core_audit": "outputs/reports/fevp_core_audit_coverage_v3.json",
            "ce2eplan_portability": (
                "outputs/reports/fevp_ce2eplan_portability_v3.json"
            ),
            "housemind_portability_gate": (
                "outputs/reports/fevp_housemind_portability_gate_v1.json"
            ),
            "adversarial_prompts": (
                "outputs/reports/fevp_adversarial_prompts_v3.json"
            ),
        },
        "evidence": {
            "base": base,
            "core_audit": core,
            "ce2eplan": ce2e,
            "housemind_gate": house_gate,
            "prompts": prompts,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    doc = [
        "# Final FEVP Claim v4",
        "",
        claim_en,
        "",
        "## 中文",
        "",
        claim_zh,
        "",
        "## Portability Boundary",
        "",
        "- CE2EPlan is a generation-only portability target.",
        "- HouseMind remains blocked by unreleased inference and machine-readable outputs.",
        "- Adversarial prompt proxies are not a general NLU benchmark.",
        "",
        "## Forbidden Claims",
        "",
        *[f"- {claim}" for claim in report["forbidden_claims"]],
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
