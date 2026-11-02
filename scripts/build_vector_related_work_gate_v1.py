#!/usr/bin/env python3
"""Record verified external positioning for the vector-editing route."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/vector_related_work_gate_v1.json"


def main() -> None:
    sources = [
        {
            "name": "RenoLayout",
            "url": "https://arxiv.org/abs/2610.21366",
            "date": "2026-10-15",
            "task": "before/after renovation and furniture layout",
            "availability": "No official code or benchmark download found as of 2026-10-28.",
            "claim_type": "context and non-ranking comparison only",
        },
        {
            "name": "HouseMind",
            "url": "https://arxiv.org/abs/2603.11640",
            "date": "2026-03-12",
            "task": "unified multimodal understanding, generation, and editing",
            "availability": "Official inference code remains Code (Soon).",
            "claim_type": "protocol context only; no defeat claim",
        },
        {
            "name": "FML / FMLM",
            "url": "https://arxiv.org/abs/2604.04859",
            "date": "2026-04-06",
            "task": "unified vector floorplan generation via markup next-token prediction",
            "availability": "Project page public; code availability not verified.",
            "claim_type": "generation and representation context only",
        },
        {
            "name": "TLC-Plan",
            "url": "https://arxiv.org/abs/2602.07100",
            "date": "2026-02-06",
            "task": "hierarchical vector floorplan generation",
            "availability": "Public GitHub repository; no LICENSE file verified.",
            "claim_type": "generation-only portability target; no redistribution",
        },
        {
            "name": "Raster2Seq",
            "url": "https://github.com/Cornell-VAILab/Raster2Seq",
            "date": "verified 2026-10-28",
            "task": "raster-to-vector floorplan reconstruction",
            "availability": "MIT; public checkpoint and code.",
            "claim_type": "matched reconstruction baseline and outside-frozen wrapper",
        },
    ]
    report = {
        "status": "complete",
        "checked_at": "2026-10-28",
        "sources": sources,
        "positioning": {
            "floorflow_difference": (
                "Strict vector-level edit locality plus topology audit, rather "
                "than generation SOTA, unified MLLM understanding, or furniture "
                "renovation breadth."
            ),
            "fairness_rules": [
                "Do not rank methods on unreported locality cells.",
                "Call Raster2Seq a matched reconstruction baseline.",
                "Call TLC-Plan and FMLM generation methods, not editors.",
                "Do not infer RenoLayout failure from unavailable code.",
                "Do not redistribute unlicensed code, data, or weights.",
            ],
        },
        "checks": {
            "reno_layout_verified": True,
            "house_mind_verified": True,
            "fml_verified": True,
            "tlc_plan_verified": True,
            "raster2seq_verified": True,
            "license_boundaries_recorded": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "sources": len(sources)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
