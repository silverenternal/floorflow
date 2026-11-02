#!/usr/bin/env python3
"""Re-rank historical checkpoints with the gate-balanced selector."""

from __future__ import annotations

import json
from pathlib import Path

from floorflow.training.typed_channel_eval import (
    GATE_TARGETS,
    selection_score,
)

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    rows = []
    for path in sorted((ROOT / "outputs/train").glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        history = report.get("history")
        if not isinstance(history, list) or not history:
            continue
        usable = [
            dict(row)
            for row in history
            if len(GATE_TARGETS.keys() & set(row)) >= 3
        ]
        if not usable:
            continue
        old_scores = [
            float(row.get("selection_score", float("-inf")))
            for row in usable
        ]
        new_scores = [
            selection_score(row, furniture_weight=1.0)
            for row in usable
        ]
        old_index = old_scores.index(max(old_scores))
        new_index = new_scores.index(max(new_scores))
        old_epoch = int(usable[old_index]["epoch"])
        new_epoch = int(usable[new_index]["epoch"])
        rows.append(
            {
                "report": str(path.relative_to(ROOT)),
                "epochs": len(usable),
                "old_epoch": old_epoch,
                "new_epoch": new_epoch,
                "changed": old_epoch != new_epoch,
                "old_best_score": float(max(old_scores)),
                "new_best_score": float(max(new_scores)),
                "new_score_at_old_epoch": float(new_scores[old_index]),
                "new_score_at_new_epoch": float(new_scores[new_index]),
            }
        )
    changed = [row for row in rows if row["changed"]]
    for row in changed:
        report_path = ROOT / row["report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        directory = report_path.parent
        old_snapshot = directory / f"epoch_{row['old_epoch']:03d}.pt"
        new_snapshot = directory / f"epoch_{row['new_epoch']:03d}.pt"
        selected_epoch = report.get("selected_epoch")
        row["old_epoch_checkpoint_available"] = bool(
            old_snapshot.is_file()
            or (
                selected_epoch == row["old_epoch"]
                and (directory / "best.pt").is_file()
            )
        )
        row["new_epoch_checkpoint_available"] = bool(
            new_snapshot.is_file()
            or (
                selected_epoch == row["new_epoch"]
                and (directory / "best.pt").is_file()
            )
        )
    output = {
        "schema_version": "gate-balanced-checkpoint-selection-audit-v1",
        "scope": "historical typed-channel training reports",
        "reports_checked": len(rows),
        "selection_changed": len(changed),
        "new_candidates_available": sum(
            bool(row["new_epoch_checkpoint_available"])
            for row in changed
        ),
        "new_candidates_unrecoverable": sum(
            not bool(row["new_epoch_checkpoint_available"])
            for row in changed
        ),
        "changed_reports": changed,
        "report_rows": rows,
    }
    path = (
        ROOT
        / "outputs/reports/gate_balanced_checkpoint_selection_audit_v1.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "reports_checked": len(rows),
                "selection_changed": len(changed),
                "changed_reports": [
                    {
                        "report": row["report"],
                        "old_epoch": row["old_epoch"],
                        "new_epoch": row["new_epoch"],
                    }
                    for row in changed[:20]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
