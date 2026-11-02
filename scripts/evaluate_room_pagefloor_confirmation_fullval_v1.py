#!/usr/bin/env python3
"""Evaluate the frozen room confirmation candidates on full validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_pagefloor_confirmation_manifest_v1.json"
        ),
    )
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [
        {
            "name": "static_parent",
            "checkpoint": Path(manifest["parent_checkpoint"]),
            "query_count_gate": False,
            "output": (
                ROOT
                / "outputs/reports/room_pagefloor_static_parent_val491_v1.json"
            ),
        }
    ]
    jobs.extend(
        {
            "name": f"seed{row['seed']}",
            "checkpoint": Path(row["checkpoint"]),
            "query_count_gate": True,
            "output": (
                ROOT
                / "outputs/reports/"
                f"room_pagefloor_confirmation_val491_seed{row['seed']}_v1.json"
            ),
        }
        for row in manifest["seeds"]
    )
    rows = []
    for job in jobs:
        output = job["output"]
        if not (args.skip_existing and output.is_file()):
            command = [
                sys.executable,
                "scripts/evaluate_raster_instance_control.py",
                "--checkpoint",
                str(job["checkpoint"]),
                "--offset",
                "0",
                "--limit",
                str(args.limit),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--page-floor-crop",
                "--device",
                args.device,
                "--output",
                str(output),
            ]
            if job["query_count_gate"]:
                command.append("--query-count-gate")
            print("\n$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        rows.append(
            {
                "name": job["name"],
                "checkpoint": str(job["checkpoint"]),
                "query_count_gate": job["query_count_gate"],
                "output": str(output),
            }
        )
    manifest_out = {
        "schema_version": "room-pagefloor-confirmation-fullval-v1",
        "status": "complete",
        "split": "val",
        "offset": 0,
        "cases": args.limit,
        "rows": rows,
    }
    output = (
        ROOT
        / "outputs/reports/room_pagefloor_confirmation_fullval_manifest_v1.json"
    )
    output.write_text(
        json.dumps(manifest_out, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_out, indent=2))


if __name__ == "__main__":
    main()
