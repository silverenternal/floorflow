#!/usr/bin/env python3
"""Summarize guidance-scale distribution sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("claim_role") != "conditional_distribution":
            continue
        rows.append(
            {
                "guidance_scale": payload["guidance_scale"],
                "mmd": payload["mmd"],
                "energy": payload["energy"],
                "coverage": payload["coverage"],
                "density": payload["density"],
                "diversity": payload["diversity"],
                "expected_room_macro_iou": payload.get("expected_room_macro_iou"),
                "sampling_policy": payload.get("sampling_policy"),
                "source": str(path),
            }
        )
    report = {
        "claim_role": "guidance_diversity_tradeoff",
        "best_of_n_used": False,
        "rows": sorted(rows, key=lambda row: row["guidance_scale"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
