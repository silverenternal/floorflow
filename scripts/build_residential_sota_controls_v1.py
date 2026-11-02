"""Freeze the residential leakage and stability controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "stress": ROOT / "outputs/reports/fp_strict_locality_stress_fiveseed_v1.json",
    "stability": ROOT / "outputs/reports/fevp_fiveseed_stability_v2.json",
    "hard_gate": ROOT / "outputs/reports/fp_control_gated_hard_gate_v1.json",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in SOURCES.items()
    }
    checks = {
        **{
            f"stress:{key}": bool(value)
            for key, value in payloads["stress"]["checks"].items()
        },
        **{
            f"stability:{key}": bool(value)
            for key, value in payloads["stability"]["checks"].items()
        },
        "hard_gate_passed": payloads["hard_gate"]["status"] == "passed",
    }
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "residential strict-locality leakage and stability controls",
        "checks": checks,
        "stress_suites": payloads["stress"]["aggregate"],
        "stress_cases": payloads["stress"]["cases"],
        "seeds": payloads["stability"]["seeds"],
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in SOURCES.items()
        },
        "boundary": payloads["stress"]["boundary"],
    }
    output = ROOT / "outputs/reports/residential_sota_controls_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
