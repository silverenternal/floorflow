#!/usr/bin/env python3
"""Document the HouseMind FEVP portability availability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-status",
        type=Path,
        default=ROOT / "outputs/reports/fp_housemind_public_status_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_housemind_portability_gate_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = json.loads(args.public_status.read_text(encoding="utf-8"))
    code_available = bool(
        status["public_artifacts"]["official_code_available"]
    )
    report = {
        "status": "blocked_external" if not code_available else "available",
        "method": "HouseMind",
        "task": "natural-language floorplan editing",
        "official_code_status": status["public_artifacts"]["code_status"],
        "verified_at": status["verified_at"],
        "official_code_available": code_available,
        "machine_readable_edit_outputs_available": False,
        "reason": (
            "The official inference, prompt assembly, and token serialization "
            "are not released. Public qualitative examples cannot be treated as "
            "a matched 100-200-row FEVP audit dataset."
        ),
        "allowed_fallback": (
            "Report public HouseMind-E/O metrics as context and mark all FEVP "
            "audit cells as not reported."
        ),
        "forbidden": [
            "Infer locality failure from unavailable code or outputs.",
            "Manually digitize qualitative examples as a benchmark.",
            "Call the public values a matched FloorFlow comparison.",
        ],
        "sources": status["sources"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"]}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
