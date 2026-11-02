"""Freeze the Room-IoU fusion arm from the completed validation report."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/reports/raster2seq_polygon_fusion_val_v4.json"
MARGIN = 0.001


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    baseline = report["strategies"]["count_consensus"]
    eligible = []
    for name, values in report["strategies"].items():
        if name in {"original", "count_consensus"}:
            continue
        paired = report["paired_vs_count_consensus"][name]
        if (
            values["room_f1"] >= baseline["room_f1"] - MARGIN
            and values["room_iou"] > baseline["room_iou"]
            and values["corner_f1"] >= baseline["corner_f1"] - MARGIN
            and paired["room_iou"]["ci_low"] > 0.0
            and paired["room_f1"]["ci_low"] > -2.0 * MARGIN
        ):
            eligible.append(name)
    selected = max(
        eligible,
        key=lambda name: report["strategies"][name]["room_iou"],
        default="count_consensus",
    )
    output = {
        **report,
        "status": "frozen",
        "selection_rule": {
            "primary": "paired Room-IoU 95% CI lower bound > 0",
            "room_f1_noninferiority_margin": MARGIN,
            "corner_f1_noninferiority_margin": MARGIN,
            "paired_room_f1_ci_lower_floor": -2.0 * MARGIN,
        },
        "eligible": eligible,
        "selected": selected,
        "test_policy": (
            "Force this validation-selected arm once on test509; no further "
            "selection or threshold changes."
        ),
    }
    destination = (
        ROOT / "outputs/reports/raster2seq_polygon_fusion_val_v5.json"
    )
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    config = {
        "version": "residential_room_iou_fusion_v1",
        "scope": "residential_floorplan_only",
        "validation_source": str(SOURCE.relative_to(ROOT)),
        "selected": selected,
        "selection_rule": output["selection_rule"],
        "validation_metrics": report["strategies"][selected],
        "paired_vs_count_consensus": report["paired_vs_count_consensus"][
            selected
        ],
    }
    config_path = ROOT / "configs/residential_room_iou_fusion_v1.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "eligible": eligible}, indent=2))


if __name__ == "__main__":
    main()
