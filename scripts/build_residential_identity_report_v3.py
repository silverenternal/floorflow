"""Build the validation-frozen eight-view identity report."""

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
        (ROOT / "outputs/reports/raster2seq_dihedral_consensus_val_v1.json").read_text(
            encoding="utf-8"
        )
    )
    test = json.loads(
        (ROOT / "outputs/reports/residential_identity_test509_v3_dihedral.json").read_text(
            encoding="utf-8"
        )
    )
    previous = json.loads(
        (ROOT / "outputs/reports/residential_identity_test509_v2_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(
        (ROOT / "outputs/reports/residential_sota_anchor_v1.json").read_text(
            encoding="utf-8"
        )
    )
    published = float(source["published_anchor"]["RoomF1"])
    val_metrics = validation["strategies"]["count_consensus_8"]
    test_metrics = test["strategies"]["count_consensus_8"]
    original = test["strategies"]["original"]
    paired = test["paired_vs_original"]
    previous_metrics = previous["test_metrics"]
    checks = {
        "validation_candidate_selected": validation["selected"] == "count_consensus_8",
        "test_policy_forced": test["selected"] == "count_consensus_8",
        "room_f1_above_public_anchor": float(test_metrics["room_f1"]) > published,
        "room_f1_margin_above_0_008": (
            float(test_metrics["room_f1"]) - published >= 0.008
        ),
        "room_iou_not_below_original": (
            float(test_metrics["room_iou"]) >= float(original["room_iou"])
        ),
        "corner_f1_above_original": (
            float(test_metrics["corner_f1"]) > float(original["corner_f1"])
        ),
        "paired_room_f1_ci_positive": float(paired["room_f1"]["ci_low"]) > 0.0,
        "paired_corner_f1_ci_positive": (
            float(paired["corner_f1"]["ci_low"]) > 0.0
        ),
        "room_iou_paired_ci_tradeoff_disclosed": (
            float(paired["room_iou"]["ci_low"]) < 0.0
        ),
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
        "policy": "validation-locked eight-view room-count consensus",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
        },
        "validation": {
            "room_f1": val_metrics["room_f1"],
            "room_iou": val_metrics["room_iou"],
            "corner_f1": val_metrics["corner_f1"],
            "paired_ci95": {
                metric: [
                    validation["paired_vs_original"][metric]["ci_low"],
                    validation["paired_vs_original"][metric]["ci_high"],
                ]
                for metric in ("room_f1", "room_iou", "corner_f1")
            },
            "threshold_sensitivity": (
                "thresholds 1, 2, and 3 select identical outputs"
            ),
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
            "room_f1_vs_original": (
                float(test_metrics["room_f1"]) - float(original["room_f1"])
            ),
            "room_f1_vs_previous_four_view": (
                float(test_metrics["room_f1"])
                - float(previous_metrics["room_f1"])
            ),
            "room_iou_vs_original": (
                float(test_metrics["room_iou"]) - float(original["room_iou"])
            ),
            "corner_f1_vs_original": (
                float(test_metrics["corner_f1"]) - float(original["corner_f1"])
            ),
        },
        "paired_test_vs_original_checkpoint": paired,
        "previous_four_view_test_result": {
            **previous_metrics,
            "source": "outputs/reports/residential_identity_test509_v2_summary.json",
        },
        "tradeoff": (
            "RoomF1 and corner F1 improve with positive paired confidence "
            "intervals. Room IoU is 0.002276 above the original point estimate, "
            "but its paired interval crosses zero, so no paired IoU claim is made."
        ),
        "checks": checks,
        "claim": (
            "After validation-only policy freezing, eight-view room-count "
            f"consensus reaches CubiCasa5K test509 RoomF1 "
            f"{test_metrics['room_f1']:.6f}, "
            f"{float(test_metrics['room_f1']) - published:.6f} above the "
            f"published 0.887 anchor, with paired RoomF1 CI "
            f"[{paired['room_f1']['ci_low']:.6f}, "
            f"{paired['room_f1']['ci_high']:.6f}] and corner F1 "
            f"{test_metrics['corner_f1']:.6f}."
        ),
        "boundary": (
            "This is identity reconstruction on test509. It is not editing. "
            "Semantic labels are not claimed from direct replay because the "
            "released evaluator mapping differs from the local category map."
        ),
    }
    output = ROOT / "outputs/reports/residential_identity_test509_v3_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Residential Identity v3",
        "",
        "| Method | RoomF1 | Room IoU | Corner F1 | Opening F1 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Eight-view room-count consensus | {test_metrics['room_f1']:.6f} | "
            f"{test_metrics['room_iou']:.6f} | {test_metrics['corner_f1']:.6f} | "
            f"{test_metrics['window_door_f1']:.6f} |"
        ),
        (
            f"| Four-view consensus | {previous_metrics['room_f1']:.6f} | "
            f"{previous_metrics['room_iou']:.6f} | "
            f"{previous_metrics['corner_f1']:.6f} | "
            f"{previous_metrics['window_door_f1']:.6f} |"
        ),
        (
            f"| Original locked checkpoint | {original['room_f1']:.6f} | "
            f"{original['room_iou']:.6f} | {original['corner_f1']:.6f} | "
            f"{original['window_door_f1']:.6f} |"
        ),
        "",
        f"Published RoomF1 anchor: {published:.3f}.",
        "",
        (
            "RoomF1 margin: "
            f"+{report['deltas']['room_f1_vs_published']:.6f}."
        ),
        (
            "Paired RoomF1 and corner-F1 confidence intervals exclude zero; "
            "the paired Room-IoU interval crosses zero."
        ),
    ]
    table = ROOT / "paper/tables/residential_identity_main_v3.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
