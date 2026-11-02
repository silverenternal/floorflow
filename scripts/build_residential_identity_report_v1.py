"""Build the final residential identity report from frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "outputs/reports/residential_identity_selection_val_v1.json",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/reports/residential_sota_anchor_v1.json",
    )
    parser.add_argument(
        "--test-results",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/"
        "residential_identity_test509_alpha075_v1/results.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/residential_identity_test509_v1.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "paper/tables/residential_identity_main_v1.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    source = json.loads(args.source.read_text(encoding="utf-8"))
    test = json.loads(args.test_results.read_text(encoding="utf-8"))
    source_metrics = source["local_reproduction"]["metrics"]
    published = float(source["published_anchor"]["RoomF1"])
    checkpoint = ROOT / selection["selected"]["path"]
    checks = {
        "room_f1_above_published_anchor": float(test["room_f1"]) > published,
        "room_iou_not_below_source": (
            float(test["room_iou"]) >= float(source_metrics["room_iou"])
        ),
        "corner_f1_not_below_source": (
            float(test["corner_f1"]) >= float(source_metrics["corner_f1"])
        ),
        "room_semantic_f1_not_below_source": (
            float(test["room_sem_f1"]) >= float(source_metrics["room_sem_f1"])
        ),
    }
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "CubiCasa5K official test509 residential identity reconstruction",
        "selection_split": "val503",
        "official_test509_runs": 1,
        "best_of_n_used": False,
        "checkpoint": {
            "path": selection["selected"]["path"],
            "sha256": sha256(checkpoint),
        },
        "method": "validation-selected model/EMA interpolation without test feedback",
        "published_anchor": {"RoomF1": published},
        "test_metrics": {
            key: float(test[key])
            for key in (
                "room_f1",
                "room_iou",
                "corner_f1",
                "room_sem_f1",
                "window_door_f1",
            )
        },
        "deltas": {
            "room_f1_vs_published_anchor": float(test["room_f1"]) - published,
            "room_f1_vs_local_source": (
                float(test["room_f1"]) - float(source_metrics["room_f1"])
            ),
            "room_iou_vs_local_source": (
                float(test["room_iou"]) - float(source_metrics["room_iou"])
            ),
            "corner_f1_vs_local_source": (
                float(test["corner_f1"]) - float(source_metrics["corner_f1"])
            ),
            "room_semantic_f1_vs_local_source": (
                float(test["room_sem_f1"]) - float(source_metrics["room_sem_f1"])
            ),
            "window_door_f1_vs_local_source": (
                float(test["window_door_f1"])
                - float(source_metrics["window_door_f1"])
            ),
        },
        "validation_tradeoff": selection["tradeoff"],
        "checks": checks,
        "claim": (
            "CubiCasa5K test509 RoomF1 exceeds the reported 0.887 anchor by "
            f"{float(test['room_f1']) - published:.6f}; Room IoU and corner F1 "
            "also remain above the local source reproduction. Window/door F1 "
            f"is a diagnostic only and changes by {float(test['window_door_f1']) - float(source_metrics['window_door_f1']):+.6f}."
        ),
        "boundary": (
            "This is a reconstruction-only result on the official test509 split. "
            "It must not be pooled with test503 editing metrics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        "\n".join(
            [
                "# Residential Identity Reconstruction",
                "",
                "| Method | RoomF1 | Room IoU | Corner F1 | Semantic F1 | Opening F1 |",
                "|---|---:|---:|---:|---:|---:|",
                (
                    f"| Selected Raster2Seq interpolation | {test['room_f1']:.6f} | "
                    f"{test['room_iou']:.6f} | {test['corner_f1']:.6f} | "
                    f"{test['room_sem_f1']:.6f} | {test['window_door_f1']:.6f} |"
                ),
                (
                    f"| Local source reproduction | {source_metrics['room_f1']:.6f} | "
                    f"{source_metrics['room_iou']:.6f} | {source_metrics['corner_f1']:.6f} | "
                    f"{source_metrics['room_sem_f1']:.6f} | "
                    f"{source_metrics['window_door_f1']:.6f} |"
                ),
                "",
                f"Published RoomF1 anchor: **{published:.3f}**.",
                "",
                f"RoomF1 margin over the reported anchor: **{report['deltas']['room_f1_vs_published_anchor']:.6f}**.",
                "",
                "Official test509 was evaluated once after validation checkpoint freezing.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
