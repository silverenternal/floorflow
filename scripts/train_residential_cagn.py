"""Run the staged residential CAGN training matrix on cuda:0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/residential_cagn_joint_v1.json"
REPORT = ROOT / "outputs/reports/residential_cagn_training_screen_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-gate", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    feature_root = ROOT / config["data"]["feature_root"]
    embedding_epochs = str(int(config["embedding_stage"]["epochs"]))
    gate_epochs = str(int(config["gate_stage"]["epochs"]))
    rows = []
    for seed in seeds:
        embedding_root = ROOT / "outputs/residential_cagn_embedding_v1"
        embedding_report = (
            ROOT / f"outputs/reports/residential_constraint_embedding_seed{seed}.json"
        )
        command = [
            sys.executable,
            "scripts/train_constraint_embedding.py",
            "--seed",
            str(seed),
            "--epochs",
            embedding_epochs,
            "--device",
            args.device,
            "--feature-root",
            str(feature_root),
            "--checkpoint-root",
            str(embedding_root),
            "--report",
            str(embedding_report),
        ]
        _run(command)
        if not args.skip_gate:
            full_root = ROOT / "outputs/residential_cagn_gate_v1"
            full_report = (
                ROOT / f"outputs/reports/residential_cagn_validation_seed{seed}.json"
            )
            _run(
                [
                    sys.executable,
                    "scripts/train_constraint_aware_gate.py",
                    "--seed",
                    str(seed),
                    "--epochs",
                    gate_epochs,
                    "--device",
                    args.device,
                    "--feature-root",
                    str(feature_root),
                    "--embedding-checkpoint",
                    str(embedding_root / f"seed{seed}.pt"),
                    "--checkpoint-root",
                    str(full_root),
                    "--report",
                    str(full_report),
                ]
            )
            scratch_root = ROOT / "outputs/residential_cagn_gate_scratch_v1"
            scratch_report = (
                ROOT
                / f"outputs/reports/residential_cagn_validation_scratch_seed{seed}.json"
            )
            _run(
                [
                    sys.executable,
                    "scripts/train_constraint_aware_gate.py",
                    "--seed",
                    str(seed),
                    "--epochs",
                    gate_epochs,
                    "--device",
                    args.device,
                    "--feature-root",
                    str(feature_root),
                    "--embedding-checkpoint",
                    str(ROOT / "outputs/nonexistent_embedding.pt"),
                    "--checkpoint-root",
                    str(scratch_root),
                    "--report",
                    str(scratch_report),
                ]
            )
        embedding = json.loads(embedding_report.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "embedding_status": embedding["status"],
                "embedding_auroc": embedding["evaluation"]["auroc"],
                "embedding_ffr": embedding["evaluation"]["false_feasible_rate"],
                "embedding_ece": embedding["evaluation"]["ece"],
            }
        )
    report = {
        "schema_version": "residential-cagn-training-screen-v1",
        "status": (
            "passed"
            if all(row["embedding_status"] == "passed" for row in rows)
            else "failed"
        ),
        "scope": "residential_floorplan_only",
        "device": args.device,
        "seeds": seeds,
        "rows": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
