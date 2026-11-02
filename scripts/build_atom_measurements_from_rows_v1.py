"""Convert typed-channel validation rows into atom proxy measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.atom_measurement_builder import build_measurements
from floorflow.evaluation.atom_metric_contract import load_metric_pairs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "configs/atom_metric_pairs_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/atom_proxy_measurements_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", nargs="+", type=Path)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.suffix == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload if isinstance(payload, list) else payload.get("rows", [])
        if not isinstance(values, list):
            raise TypeError(f"{path} must contain a row list")
        rows.extend(values)
    return rows


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.rows)
    pairs = load_metric_pairs(args.pairs)
    measurements = build_measurements(rows, pairs)
    payload = {
        "schema_version": "atom-proxy-measurements-v1",
        "target_used_at_inference": False,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "measurements": len(measurements),
                "rows": sum(len(item["rows"]) for item in measurements),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
