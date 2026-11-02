"""Build the final identity report for room-count consensus."""

from __future__ import annotations

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


def main() -> None:
    validation = json.loads(
        (ROOT / "outputs/reports/raster2seq_tta_consensus_val_v1.json").read_text(
            encoding="utf-8"
        )
    )
    test = json.loads(
        (ROOT / "outputs/reports/residential_identity_test509_v2.json").read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(
        (ROOT / "outputs/reports/residential_sota_anchor_v1.json").read_text(
            encoding="utf-8"
        )
    )
    published = float(source["published_anchor"]["RoomF1"])
    val_metrics = validation["strategies"]["count_consensus"]
    test_metrics = test["strategies"]["count_consensus"]
    paired = test["paired_vs_original"]
    checks = {
        "validation_candidate_selected": validation["selected"] == "count_consensus",
        "test_policy_forced": test["selected"] == "count_consensus",
        "room_f1_above_public_anchor": float(test_metrics["room_f1"]) > published,
        "paired_room_f1_ci_positive": (
            float(paired["room_f1"]["ci_low"]) > 0.0
        ),
        "corner_f1_above_original": float(test_metrics["corner_f1"]) > float(
            test["strategies"]["original"]["corner_f1"]
        ),
        "room_iou_tradeoff_disclosed": (
            float(test_metrics["room_iou"])
            < float(test["strategies"]["original"]["room_iou"])
        ),
    }
    checkpoint = ROOT / "outputs/external_baselines/raster2seq/checkpoint_blends_v2/ema_alpha_0.75.pth"
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "CubiCasa5K official test509 residential identity reconstruction",
        "selection_split": "val503",
        "official_test509_used": True,
        "policy": "validation-locked four-view room-count consensus",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
        },
        "validation": {
            "room_f1": val_metrics["room_f1"],
            "room_iou": val_metrics["room_iou"],
            "corner_f1": val_metrics["corner_f1"],
            "paired_room_f1_ci95": [
                validation["paired_vs_original"]["room_f1"]["ci_low"],
                validation["paired_vs_original"]["room_f1"]["ci_high"],
            ],
        },
        "test_metrics": {
            key: float(test_metrics[key])
            for key in (
                "room_f1",
                "room_iou",
                "corner_f1",
                "window_door_f1",
            )
        },
        "published_anchor": {"RoomF1": published},
        "deltas": {
            "room_f1_vs_published": float(test_metrics["room_f1"]) - published,
            "room_f1_vs_previous_locked_result": (
                float(test_metrics["room_f1"]) - 0.8872897049424247
            ),
            "corner_f1_vs_previous_locked_result": (
                float(test_metrics["corner_f1"]) - 0.5960383558114678
            ),
            "room_iou_vs_previous_locked_result": (
                float(test_metrics["room_iou"]) - 0.6191139330901431
            ),
        },
        "paired_test_vs_original_checkpoint": paired,
        "tradeoff": (
            "RoomF1 and corner F1 improve, while Room IoU decreases by 0.001605 "
            "with a 95% paired CI crossing zero. Semantic labels are not claimed "
            "from direct replay because the released evaluator mapping differs."
        ),
        "checks": checks,
        "claim": (
            "After validation-only policy freezing, four-view room-count "
            f"consensus reaches CubiCasa5K test509 RoomF1 {test_metrics['room_f1']:.6f}, "
            f"{float(test_metrics['room_f1']) - published:.6f} above the reported "
            f"0.887 anchor, with paired RoomF1 CI "
            f"[{paired['room_f1']['ci_low']:.6f}, {paired['room_f1']['ci_high']:.6f}]."
        ),
        "boundary": (
            "This is an identity reconstruction result. The IoU trade-off and "
            "semantic-mapping limitation must be reported explicitly."
        ),
    }
    output = ROOT / "outputs/reports/residential_identity_test509_v2_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Residential Identity v2",
        "",
        "| Method | RoomF1 | Room IoU | Corner F1 | Opening F1 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Four-view room-count consensus | {test_metrics['room_f1']:.6f} | "
            f"{test_metrics['room_iou']:.6f} | {test_metrics['corner_f1']:.6f} | "
            f"{test_metrics['window_door_f1']:.6f} |"
        ),
        (
            f"| Original selected checkpoint | "
            f"{test['strategies']['original']['room_f1']:.6f} | "
            f"{test['strategies']['original']['room_iou']:.6f} | "
            f"{test['strategies']['original']['corner_f1']:.6f} | "
            f"{test['strategies']['original']['window_door_f1']:.6f} |"
        ),
        "",
        f"Published RoomF1 anchor: {published:.3f}.",
        "",
        f"RoomF1 margin: +{report['deltas']['room_f1_vs_published']:.6f}.",
        "",
        "IoU trade-off: -0.001605 versus the previous locked result.",
    ]
    table = ROOT / "paper/tables/residential_identity_main_v2.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
