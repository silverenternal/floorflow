#!/usr/bin/env python3
"""Summarize raster-conditioned control screens and promotion state."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _metrics(path: str, arm: str = "overlay") -> dict:
    report = _read(path)
    return report["room_metrics"][arm]


def main() -> None:
    baseline = _metrics(
        "outputs/reports/v11_roomseg_w20_retr0_val491.json"
    )
    clean = _metrics(
        "outputs/reports/raster_room_control_full_model_val491_fullmetrics_v1.json"
    )
    native = _metrics(
        "outputs/reports/raster_room_control_native_full_val491_fullmetrics_v1.json"
    )
    clean_train = _read(
        "outputs/train/raster_room_control_full_v1/report.json"
    )
    native_train = _read(
        "outputs/train/raster_room_control_native_full_v1/report.json"
    )
    screen_gate = {
        "room_macro_iou_min": 0.25,
        "instance_room_f1_min": 0.25,
        "instance_geometric_iou_min": 0.60,
        "validity": 1.0,
    }
    native_pass = (
        native["room_macro_iou"] >= screen_gate["room_macro_iou_min"]
        and native["instance_room_f1"]
        >= screen_gate["instance_room_f1_min"]
        and native["instance_geometric_iou"]
        >= screen_gate["instance_geometric_iou_min"]
        and native["fpeval_validity"] == screen_gate["validity"]
    )
    report = {
        "schema_version": "raster-control-screen-report-v1",
        "scope": "CubiCasa val491; test503 remains closed",
        "baseline": baseline,
        "clean_nonroom_rendering": {
            "training": clean_train,
            "metrics": clean,
            "track": "rendered_structural_raster",
        },
        "native_cubicasa_image": {
            "training": native_train,
            "metrics": native,
            "track": "raster_conditioned_reconstruction",
        },
        "screen_gate": screen_gate,
        "native_image_screen_passed": native_pass,
        "rc02_promotion": {
            "promoted": False,
            "reason": (
                "Clean rendering is strong but native CubiCasa-image room "
                "identity remains below the clean-render ceiling. A "
                "full wall/opening/furniture control bank and matched external "
                "evaluation are not yet complete."
            ),
        },
        "deltas": {
            "clean_room_iou": clean["room_macro_iou"]
            - baseline["room_macro_iou"],
            "clean_instance_f1": clean["instance_room_f1"]
            - baseline["instance_room_f1"],
            "native_room_iou": native["room_macro_iou"]
            - baseline["room_macro_iou"],
            "native_instance_f1": native["instance_room_f1"]
            - baseline["instance_room_f1"],
        },
    }
    output = ROOT / "outputs/reports/raster_room_control_screen_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
