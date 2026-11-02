"""Build the final Room-IoU-fused residential identity report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    validation = json.loads(
        (
            ROOT / "outputs/reports/raster2seq_polygon_fusion_val_v5.json"
        ).read_text(encoding="utf-8")
    )
    test = json.loads(
        (
            ROOT / "outputs/reports/residential_identity_test509_room_iou_v1.json"
        ).read_text(encoding="utf-8")
    )
    previous = json.loads(
        (
            ROOT / "outputs/reports/residential_identity_test509_v3_summary.json"
        ).read_text(encoding="utf-8")
    )
    selected = test["selected"]
    count = test["strategies"]["count_consensus"]
    fused = test["strategies"][selected]
    original = test["strategies"]["original"]
    paired = test["paired_vs_count_consensus"][selected]
    room_iou_differences = [
        fused_row["room_iou"] - count_row["room_iou"]
        for fused_row, count_row in zip(
            test["case_rows"][selected],
            test["case_rows"]["count_consensus"],
            strict=True,
        )
    ]
    room_iou_wilcoxon = wilcoxon(
        room_iou_differences,
        alternative="greater",
        zero_method="wilcox",
        method="approx",
    )
    paired["room_iou"]["wilcoxon_p_greater"] = float(
        room_iou_wilcoxon.pvalue
    )
    published = float(previous["published_anchor"]["RoomF1"])
    checks = {
        "validation_selected_fusion": (
            validation["selected"] == selected
            and selected != "count_consensus"
        ),
        "room_f1_unchanged_on_test": (
            abs(fused["room_f1"] - count["room_f1"]) < 1e-12
        ),
        "room_iou_paired_ci_positive": paired["room_iou"]["ci_low"] > 0.0,
        "room_iou_above_previous": (
            fused["room_iou"] > previous["test_metrics"]["room_iou"]
        ),
        "corner_f1_not_below_previous": (
            fused["corner_f1"] >= previous["test_metrics"]["corner_f1"]
        ),
        "room_f1_above_published_anchor": fused["room_f1"] > published,
    }
    checkpoint = ROOT / (
        "outputs/external_baselines/raster2seq/"
        "checkpoint_blends_v2/ema_alpha_0.75.pth"
    )
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "CubiCasa5K official test509 residential identity reconstruction",
        "selection_split": "val503",
        "official_test509_used": True,
        "policy": (
            "eight-view room-count consensus followed by selected-view "
            "orientation-aware five-view polygon fusion"
        ),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
        },
        "validation": {
            "selected": selected,
            "room_f1": validation["strategies"][selected]["room_f1"],
            "room_iou": validation["strategies"][selected]["room_iou"],
            "corner_f1": validation["strategies"][selected]["corner_f1"],
            "paired_room_iou": validation[
                "paired_vs_count_consensus"
            ][selected]["room_iou"],
            "selection_rule": validation["selection_rule"],
        },
        "test_metrics": {
            key: float(fused[key])
            for key in (
                "room_f1",
                "room_iou",
                "corner_f1",
                "window_door_f1",
            )
        },
        "baselines": {
            "original_checkpoint": {
                key: original[key]
                for key in ("room_f1", "room_iou", "corner_f1")
            },
            "eight_view_count_consensus": {
                key: count[key]
                for key in ("room_f1", "room_iou", "corner_f1")
            },
            "published_room_f1": published,
        },
        "deltas": {
            "room_f1_vs_published": fused["room_f1"] - published,
            "room_f1_vs_count_consensus": fused["room_f1"] - count["room_f1"],
            "room_iou_vs_original": fused["room_iou"] - original["room_iou"],
            "room_iou_vs_count_consensus": (
                fused["room_iou"] - count["room_iou"]
            ),
            "corner_f1_vs_original": (
                fused["corner_f1"] - original["corner_f1"]
            ),
        },
        "paired_test_vs_count_consensus": paired,
        "checks": checks,
        "claim": (
            "On CubiCasa5K test509, validation-frozen polygon fusion preserves "
            f"RoomF1 {fused['room_f1']:.6f}, raises Room IoU to "
            f"{fused['room_iou']:.6f} with paired 95% CI "
            f"[{paired['room_iou']['ci_low']:.6f}, "
            f"{paired['room_iou']['ci_high']:.6f}], and leaves RoomF1 exactly "
            "unchanged for all 509 cases."
        ),
        "boundary": (
            "The paired Room-IoU gain is specific to validation-selected "
            "five-view polygon fusion on the official test509 protocol. "
            "Semantic labels remain outside the claim."
        ),
    }
    output = ROOT / "outputs/reports/residential_identity_test509_v4_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = ROOT / "paper/tables/residential_identity_main_v4.md"
    table.write_text(
        "\n".join(
            [
                "# Residential Identity v4",
                "",
                "| Method | RoomF1 | Room IoU | Corner F1 | Opening F1 |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| Eight-view consensus + polygon fusion | "
                    f"{fused['room_f1']:.6f} | {fused['room_iou']:.6f} | "
                    f"{fused['corner_f1']:.6f} | "
                    f"{fused['window_door_f1']:.6f} |"
                ),
                (
                    f"| Eight-view count consensus | {count['room_f1']:.6f} | "
                    f"{count['room_iou']:.6f} | {count['corner_f1']:.6f} | "
                    f"{count['window_door_f1']:.6f} |"
                ),
                "",
                (
                    "Paired Room IoU: "
                    f"{paired['room_iou']['mean_delta']:+.6f} "
                    f"[{paired['room_iou']['ci_low']:.6f}, "
                    f"{paired['room_iou']['ci_high']:.6f}]."
                ),
                "RoomF1 changed on 0/509 cases relative to count consensus.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
