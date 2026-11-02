import json
import subprocess
import sys
from pathlib import Path


def test_head_audit_flags_zero_loss(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    output = tmp_path / "audit.json"
    rows = [
        {"epoch": 0, "train": {"active_loss": 2.0, "dead_loss": 0.0}},
        {"epoch": 1, "train": {"active_loss": 1.0, "dead_loss": 0.0}},
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/build_head_audit.py",
            "--train-log",
            str(log),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["zero_or_invalid"] == ["dead_loss"]
