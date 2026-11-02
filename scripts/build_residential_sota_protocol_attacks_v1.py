"""Freeze the existing FEVP adversarial audit for the residential package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTACKS = ROOT / "outputs/reports/fevp_protocol_attacks_v2.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    attacks = json.loads(ATTACKS.read_text(encoding="utf-8"))
    checks = dict(attacks["checks"])
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "FEVP v1.0 hardened protocol sensitivity",
        "source_report": str(ATTACKS.relative_to(ROOT)),
        "source_sha256": sha256(ATTACKS),
        "checks": checks,
        "by_metric": attacks["by_metric"],
        "boundary": attacks["boundary"],
    }
    output = ROOT / "outputs/reports/residential_sota_protocol_attacks_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
