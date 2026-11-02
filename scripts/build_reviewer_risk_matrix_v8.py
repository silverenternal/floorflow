#!/usr/bin/env python3
"""Build reviewer risks for portable FEVP and prompt-proxy robustness."""

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
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v8.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = json.loads(
        (ROOT / "outputs/reports/fp_cvpr_evidence_v11.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = evidence["portability_and_robustness"]
    final = bundle["final_claim"]
    core = bundle["core_audit"]
    ce2e = bundle["ce2eplan"]
    house = bundle["house_mind_gate"]
    prompts = bundle["adversarial_prompts"]
    risks = [
        {
            "risk": "FEVP is a self-defined scorer for FloorFlow only",
            "evidence": (
                f"The same evaluator runs on {ce2e['cases']} CE2EPlan public "
                "generation rows and produces finite validity, watertight, graph, and GED metrics."
            ),
            "response": "Present this as task-appropriate portability, not model ranking.",
            "residual": "Only one non-FloorFlow output family is currently adapted.",
        },
        {
            "risk": "CE2EPlan portability is treated as editing evidence",
            "evidence": "CE2EPlan is generation-only and outside exact is N/A.",
            "response": "Keep outside/violation cells N/A and compare no editing quality.",
            "residual": "Generation audit does not validate edit locality.",
        },
        {
            "risk": "HouseMind is absent from portability experiments",
            "evidence": (
                f"Official code remains {house['official_code_status']} and "
                "machine-readable edit outputs are unavailable."
            ),
            "response": "Report the availability gate and unreported audit cells.",
            "residual": "FEVP cannot yet audit HouseMind outputs.",
        },
        {
            "risk": "HouseMind is inferred to fail locality",
            "evidence": "Missing code and missing metrics are not evidence of failure.",
            "response": "Use only not-reported language and no ranking.",
            "residual": "Qualitative HouseMind claims remain outside the audit.",
        },
        {
            "risk": "Unreported cells are read as zeros",
            "evidence": core["footnote"],
            "response": "Keep the not-reported footnote beside every coverage table.",
            "residual": "Readers may still misread missing evidence.",
        },
        {
            "risk": "Adversarial prompts are overclaimed as free-form NLU",
            "evidence": prompts["boundary"],
            "response": "Call them machine-readable prompt proxies over real matched controls.",
            "residual": "No natural-language parser is evaluated.",
        },
        {
            "risk": "Prompt proxies are cherry-picked",
            "evidence": (
                "Exactly four predeclared categories are sampled with deterministic "
                "hash ordering and equal per-category counts."
            ),
            "response": "Publish prompt rows, categories, and selection function.",
            "residual": "The categories are not exhaustive.",
        },
        {
            "risk": "Conflict instructions break locality",
            "evidence": (
                f"Typed outside exact is "
                f"{prompts['summary']['typed_edit_graph']['outside_pixel_exact']:.6f} "
                "and violation is zero on 400 proxies."
            ),
            "response": "Keep strict locality and quality metrics together.",
            "residual": "Topology quality still drops in room deletion/global-scope proxies.",
        },
        {
            "risk": "No-gate appears equivalent",
            "evidence": (
                f"No-gate violation is "
                f"{prompts['summary']['no_edit_gating']['locality_violation_rate']:.6f} "
                "and GED is higher than Typed."
            ),
            "response": "Report paired locality and GED contrasts.",
            "residual": "Some opening-only proxies show nearly zero leakage.",
        },
        {
            "risk": "Room deletion is confused with leakage",
            "evidence": "Deletion GED tail remains while outside exactness stays 1.0.",
            "response": "Keep the leakage-vs-topology decomposition.",
            "residual": "Deletion remains a hard topology mode.",
        },
        {
            "risk": "Watertight guard is reintroduced as a contribution",
            "evidence": "57 repairs versus 67 regressions remain unchanged.",
            "response": "Keep guard as safety baseline only.",
            "residual": "Selective future repair may differ.",
        },
        {
            "risk": "Human preference is missing",
            "evidence": "Human evaluation is excluded by scope.",
            "response": "Make no subjective-quality claim.",
            "residual": "Perceived design quality remains unmeasured.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v8",
        "",
        "Portable FEVP and adversarial prompt-proxy evidence.",
        "",
        "| Risk | Evidence | Response | Residual |",
        "|---|---|---|---|",
    ]
    for risk in risks:
        lines.append(
            f"| {risk['risk']} | {risk['evidence']} | "
            f"{risk['response']} | {risk['residual']} |"
        )
    lines.extend(
        [
            "",
            "## Final Claim",
            "",
            f"> {final['claim_en']}",
            "",
            "## Mandatory Boundaries",
            "",
            "- CE2EPlan is generation-only and outside exact is N/A.",
            "- HouseMind is unreported/blocked, not failed.",
            "- Prompt proxies are not a free-form NLU benchmark.",
            "- Watertight guard is not a net contribution.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "risks": len(risks)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
