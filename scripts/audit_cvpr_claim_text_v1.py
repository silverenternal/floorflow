#!/usr/bin/env python3
"""Audit canonical CVPR claim sections for unsupported or stale language."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claim-sections",
        type=Path,
        default=ROOT / "paper/cvpr_claim_sections_v1.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cvpr_claim_text_audit_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.claim_sections.read_text(encoding="utf-8")
    lower = text.lower()
    errors = []
    for phrase in (
        "beats housemind",
        "outperforms housemind",
        "better than housemind",
        "housemind fails",
        "unreported cells show inferiority",
        "validity is 1.0",
        "watertight guard improves",
        "human preference",
        "todo",
    ):
        for line in text.splitlines():
            if phrase not in line.lower():
                continue
            if re.search(
                r"\b(no|not|do not|does not|without)\b",
                line,
                re.IGNORECASE,
            ):
                continue
            errors.append(f"forbidden phrase: {phrase}")
    if "generation sota" in lower and "not a new generation sota" not in lower:
        errors.append("generation SOTA appears without explicit negation")
    significant_lines = [
        line
        for line in text.splitlines()
        if re.search(r"\bsignificant(?:ly)?\b", line, flags=re.IGNORECASE)
    ]
    unsupported_significant = [
        line
        for line in significant_lines
        if not re.search(
            r"\b(p\s*[<=>]|CI\b|confidence interval)",
            line,
            re.IGNORECASE,
        )
    ]
    errors.extend(
        f"significant wording lacks p/CI: {line}"
        for line in unsupported_significant
    )
    required_concepts = {
        "house_mind_tokenization": (
            "discrete room-instance tokens" in lower
        ),
        "house_mind_unified_multimodal": (
            "understanding, generation, and editing" in lower
        ),
        "ce2eplan_end_to_end": (
            "end-to-end boundary-conditioned vector floorplan generation" in lower
        ),
        "ce2eplan_topology_geometry": (
            "topology- and geometry-enhanced" in lower
        ),
        "generation_editing_separated": (
            "generation-only target" in lower
            and "no generation result is interpreted as an editing result" in lower
        ),
        "protocol_model_separated": (
            "contribution is protocol and evidence architecture" in lower
        ),
        "no_housemind_defeat": (
            "do not claim that floorflow beats housemind" in lower
        ),
        "raster2seq_matched_reconstruction": (
            "matched reconstruction baseline" in lower
            and "not an editing-aware model" in lower
        ),
        "unreported_not_failure": (
            "absence defines an audit boundary, not a performance ranking" in lower
        ),
    }
    failed_concepts = [
        name for name, present in required_concepts.items() if not present
    ]
    errors.extend(f"missing concept: {name}" for name in failed_concepts)
    report = {
        "status": "complete" if not errors else "failed",
        "source": str(args.claim_sections.relative_to(ROOT)),
        "errors": errors,
        "checks": {
            **required_concepts,
            "no_forbidden_claim_phrases": not any(
                error.startswith("forbidden phrase") for error in errors
            ),
            "significant_claims_have_p_or_ci": not unsupported_significant,
            "no_todo_or_internal_task_ids": not re.search(
                r"\b(TODO|FP-|EXT-|SYN-)\b",
                text,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
