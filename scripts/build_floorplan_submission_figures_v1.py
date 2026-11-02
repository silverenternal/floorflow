#!/usr/bin/env python3
"""Build publication-style floorplan submission figures from frozen evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper/figures"
REGISTRY = ROOT / "docs/floorplan_submission_source_registry_v2.json"
OPERATION_EVIDENCE = ROOT / "outputs/reports/predicted_operation_evidence_v2.json"
OPERATION_ROWS = ROOT / "outputs/reports/predicted_operation_success_val191_v1.json"
ORACLE_DIR = ROOT / "outputs/eval/fp_control_gated_local_test503_seed19_v1"
PREDICTED_DIR = ROOT / "outputs/eval"
CONTROL_DIR = ROOT / "outputs/cache/typed_edit_controls_v1"
DETECTION_DIR = ROOT / "outputs/cache/fp_edit_detection_v1"
CASE_MANIFEST = ROOT / "outputs/reports/floorplan_figure_case_manifest_v1.json"

COLORS = {
    "blue": "#2f6fb6",
    "teal": "#16837a",
    "green": "#3f8f5f",
    "amber": "#b77a00",
    "red": "#b84a5a",
    "purple": "#7254a7",
    "gray": "#526173",
    "light_blue": "#eaf2fb",
    "light_green": "#eaf5ee",
    "light_amber": "#fff5df",
    "light_red": "#fbecef",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.svg")
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _value(entries: dict[str, dict], key: str):
    return entries[key]["expected_value"]


def _mean(value) -> float:
    return float(value["mean"] if isinstance(value, dict) else value)


def _colorize(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int16)
    rgb = mpl.colormaps["tab20"]((labels % 20) / 20.0)[..., :3]
    rgb[labels == 0] = 1.0
    rgb[labels == 1] = np.array([0.20, 0.23, 0.27])
    return rgb


def _error_map(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    correct = target == prediction
    rgb = np.empty((*target.shape, 3), dtype=float)
    rgb[correct] = np.array([0.92, 0.95, 0.93])
    rgb[~correct] = np.array([0.73, 0.20, 0.28])
    return rgb


def _image_panel(ax: plt.Axes, image: np.ndarray, title: str, *, cmap: str | None = None) -> None:
    ax.imshow(image, interpolation="nearest", cmap=cmap)
    ax.set_title(title, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])


def _failed_predicted_case(rows: list[dict]) -> dict | None:
    priority = ("room_remove", "room_type", "opening_delete", "furniture_delete")
    by_operation = {name: [] for name in priority}
    for row in rows:
        if not row.get("success_applicable") or row.get("success"):
            continue
        operation = row.get("operation")
        if operation not in by_operation:
            continue
        path = PREDICTED_DIR / f"fp_local_edit_{operation}_val191_v1/rasters/{row['name']}.npz"
        if path.is_file():
            by_operation[operation].append((str(path), row))
    for operation in priority:
        candidates = sorted(by_operation[operation], key=lambda item: (item[1]["name"], item[1]["prompt_id"]))
        if candidates:
            path, row = candidates[0]
            return {**row, "path": path}
    return None


def _qualitative_figure(registry: dict) -> None:
    entries = {entry["key"]: entry for entry in registry["entries"]}
    oracle_name = "high_quality_architectural__1191__Floor-1"
    oracle_path = ORACLE_DIR / f"room_type/{oracle_name}.npz"
    control_path = CONTROL_DIR / f"room_type/{oracle_name}.npz"
    detection_path = DETECTION_DIR / f"room_type/{oracle_name}.npz"
    if not all(path.is_file() for path in (oracle_path, control_path, detection_path)):
        raise FileNotFoundError(f"missing oracle qualitative case: {oracle_name}")
    oracle = np.load(oracle_path)
    control = np.load(control_path)
    detection = np.load(detection_path)

    rows = _read_json(OPERATION_ROWS).get("rows", [])
    failure = _failed_predicted_case(rows)
    if failure is None:
        raise FileNotFoundError("no failed predicted case with existing val191 raster")
    predicted_path = Path(failure["path"])
    predicted = np.load(predicted_path)

    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.0))
    oracle_panels = (
        (_colorize(control["before"]), "Before control"),
        (_colorize(control["after"]), "After control"),
        (detection["changed"], "Changed support"),
        (_colorize(oracle["prediction"]), "Prediction"),
        (_colorize(oracle["target"]), "Reference"),
    )
    for ax, (image, title) in zip(axes[0], oracle_panels, strict=True):
        _image_panel(ax, image, title, cmap="gray" if image.dtype == bool else None)
    axes[0, 0].set_ylabel("Oracle control\n(strong positive)", fontsize=8)

    predicted_panels = (
        (_colorize(predicted["target"]), "Reference"),
        (_colorize(predicted["prediction"]), "Prediction"),
        (_colorize(predicted["topology_prediction"]), "Topology readout"),
        (_error_map(predicted["target"], predicted["prediction"]), "Class errors"),
        (predicted["footprint"], "Validity support"),
    )
    for ax, (image, title) in zip(axes[1], predicted_panels, strict=True):
        _image_panel(ax, image, title, cmap="gray" if image.dtype == bool or title == "Validity support" else None)
    axes[1, 0].set_ylabel("Predicted control\n(failure retained)", fontsize=8)
    fig.text(
        0.5,
        0.012,
        "Oracle: target-derived control, not deployable. Predicted: target-free val191 control; failed row shown without reclassification. "
        f"Delta-IoU oracle room type = {_mean(_value(entries, 'oracle.room_type.delta_iou')):.6f}.",
        ha="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2, w_pad=0.45)
    _save(fig, "floorplan_qualitative_cases_v1")
    manifest = {
        "schema_version": "floorplan-figure-case-manifest-v1",
        "sampling_rule": {
            "oracle_positive": "Known valid room-type oracle-control case named in the frozen evidence.",
            "predicted_failure": "First lexicographic failed supported row with an existing val191 raster, prioritizing room deletion, then room type, opening, and furniture.",
        },
        "cases": [
            {
                "role": "oracle_positive",
                "operation": "room_type",
                "name": oracle_name,
                "sources": [str(oracle_path.relative_to(ROOT)), str(control_path.relative_to(ROOT)), str(detection_path.relative_to(ROOT))],
            },
            {
                "role": "predicted_failure",
                "operation": failure["operation"],
                "category": failure["category"],
                "prompt_id": failure["prompt_id"],
                "name": failure["name"],
                "failure_reason": failure["failure_reason"],
                "source": str(predicted_path.relative_to(ROOT)),
            },
        ],
    }
    CASE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CASE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    edge: str,
    face: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        color=COLORS["gray"],
    )
    ax.text(
        x + width / 2,
        y + height * 0.32,
        body,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["gray"],
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.3,
            color=COLORS["gray"],
        )
    )


def _protocol_figure() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _card(
        ax,
        0.2,
        2.2,
        2.0,
        1.15,
        "Pre-edit plan",
        "old CAD state",
        edge=COLORS["blue"],
        face=COLORS["light_blue"],
    )
    _card(
        ax,
        2.7,
        2.2,
        2.0,
        1.15,
        "Control request",
        "region + operation",
        edge=COLORS["blue"],
        face=COLORS["light_blue"],
    )
    _card(
        ax,
        5.2,
        2.2,
        2.2,
        1.15,
        "Typed execution",
        "control delta + routing",
        edge=COLORS["purple"],
        face="#f0ecf8",
    )
    _card(
        ax,
        8.0,
        3.0,
        1.9,
        0.85,
        "Supported",
        "local edit",
        edge=COLORS["green"],
        face=COLORS["light_green"],
    )
    _card(
        ax,
        8.0,
        1.65,
        1.9,
        0.85,
        "Unsupported",
        "safe abstention",
        edge=COLORS["amber"],
        face=COLORS["light_amber"],
    )
    _card(
        ax,
        10.2,
        2.2,
        1.55,
        1.15,
        "After plan",
        "verified output",
        edge=COLORS["teal"],
        face="#e8f4f2",
    )
    _arrow(ax, (2.2, 2.78), (2.7, 2.78))
    _arrow(ax, (4.7, 2.78), (5.2, 2.78))
    _arrow(ax, (7.4, 2.78), (8.0, 3.42))
    _arrow(ax, (7.4, 2.78), (8.0, 2.08))
    _arrow(ax, (9.9, 3.42), (10.2, 2.9))
    _arrow(ax, (9.9, 2.08), (10.2, 2.65))
    ax.text(
        6.0,
        0.55,
        "Audit: exact outside content  ·  topology validity  ·  structural plausibility",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["gray"],
    )
    _save(fig, "floorplan_protocol_overview_v1")


def _predicted_evidence_figure(evidence: dict, registry: dict) -> None:
    entries = {entry["key"]: entry for entry in registry["entries"]}
    semantic = evidence["semantic_diagnostics"]["operations"]
    supported = [
        ("Room add", semantic["room_addition"]["success_rate"]),
        ("Room type", semantic["room_type_change"]["success_rate"]),
        ("Room remove", semantic["room_deletion"]["success_rate"]),
        ("Opening", semantic["opening_pressure"]["success_rate"]),
        ("Furniture", semantic["furniture_low_salience"]["success_rate"]),
    ]
    plausibility = [
        ("Raster", _mean(_value(entries, "predicted.val491.raster_plausible"))),
        ("Plan", _mean(_value(entries, "predicted.val491.plan_plausible"))),
        ("Functional\nwatertight", _mean(_value(entries, "predicted.val491.functional_watertight"))),
        ("Wall\ncontinuity", _mean(_value(entries, "predicted.val491.wall_continuity_rate"))),
        ("Raw\nwatertight", _mean(_value(entries, "predicted.val491.watertight"))),
    ]
    safety_summary = evidence["safety"]
    executed_rate = safety_summary["executed"] / (safety_summary["prompts"] - safety_summary["abstained"])
    abstained_rate = safety_summary["abstained"] / (safety_summary["prompts"] - safety_summary["executed"])
    safety = [
        ("Supported\nexecuted", executed_rate),
        ("Unsupported\nabstained", abstained_rate),
        ("Outside\nexact", safety_summary["outside_pixel_exact_min"]),
        ("Validity", safety_summary["validity_min"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))

    for ax, values, title in (
        (axes[0], safety, "Safety contract"),
        (axes[1], plausibility, "Structural plausibility"),
    ):
        labels = [item[0] for item in values]
        scores = [float(item[1]) for item in values]
        colors = [
            COLORS["green"] if score >= 0.95 else COLORS["red"]
            for score in scores
        ]
        bars = ax.bar(np.arange(len(scores)), scores, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(scores)), labels)
        ax.set_ylim(0.7, 1.02)
        ax.set_ylabel("Rate")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, score in zip(bars, scores, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.006,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    ax = axes[2]
    labels = [item[0] for item in supported]
    scores = [float(item[1]) for item in supported]
    colors = [
        COLORS["green"] if score >= 0.9 else COLORS["amber"] if score >= 0.5 else COLORS["red"]
        for score in scores
    ]
    bars = ax.bar(np.arange(len(scores)), scores, color=colors, width=0.68)
    ax.axhline(0.9, color=COLORS["red"], linestyle="--", linewidth=1.1)
    ax.text(4.35, 0.91, "target", color=COLORS["red"], ha="right", va="bottom")
    ax.set_xticks(np.arange(len(scores)), labels, rotation=28, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Semantic success")
    ax.set_title("Operation realization")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    fig.text(
        0.5,
        0.01,
        "Frozen scoped release: val491 plausibility, development-exposed safety, and a separately labeled older val191 semantic diagnostic.",
        ha="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "predicted_control_operation_panels_v1")


def _failure_figure(evidence: dict, rows: list[dict]) -> None:
    summary = evidence["semantic_diagnostics"]["operations"]
    names = [
        "room_addition",
        "room_type_change",
        "room_deletion",
        "opening_pressure",
        "furniture_low_salience",
    ]
    labels = ["Room add", "Room type", "Room remove", "Opening", "Furniture"]
    successes = [summary[name]["successful_prompts"] for name in names]
    totals = [summary[name]["prompts"] for name in names]
    failures = [total - success for total, success in zip(totals, successes, strict=True)]

    case = _failed_predicted_case(rows)
    if case is None:
        raise FileNotFoundError("no failed predicted case with existing val191 raster")
    predicted = np.load(case["path"])

    fig = plt.figure(figsize=(11.2, 3.5))
    grid = fig.add_gridspec(1, 4, width_ratios=(1.55, 1, 1, 1), wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    positions = np.arange(len(labels))
    ax.barh(positions, successes, color=COLORS["green"], label="Semantic success")
    ax.barh(
        positions,
        failures,
        left=successes,
        color=COLORS["red"],
        alpha=0.8,
        label="Failure",
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Prompts")
    ax.set_title("Supported-operation semantic outcomes")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for row, (success, total) in enumerate(zip(successes, totals, strict=True)):
        ax.text(
            total + 2,
            row,
            f"{success}/{total}",
            va="center",
            fontsize=7,
            color=COLORS["gray"],
        )
    ax.legend(frameon=False, loc="lower right")
    for column, (image, title) in enumerate(
        (
            (predicted["target"], "Reference"),
            (predicted["prediction"], "Prediction"),
            (_error_map(predicted["target"], predicted["prediction"]), "Class errors"),
        ),
        start=1,
    ):
        panel = fig.add_subplot(grid[0, column])
        _image_panel(panel, image, title, cmap="gray" if image.dtype == bool else None)
    fig.text(
        0.055,
        0.105,
        f"Failed case: {case['name']}\n{case['category']} · {case['failure_reason']}",
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["gray"],
    )
    fig.text(
        0.5,
        0.01,
        "Failures are retained; safety and semantic success are separate claims. The displayed failed case is selected by a recorded deterministic rule.",
        ha="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.30, top=0.92, wspace=0.18)
    _save(fig, "floorplan_failure_taxonomy_v1")


def main() -> None:
    _style()
    registry = _read_json(REGISTRY)
    evidence = _read_json(OPERATION_EVIDENCE)
    rows = _read_json(OPERATION_ROWS).get("rows", [])
    _protocol_figure()
    _predicted_evidence_figure(evidence, registry)
    _failure_figure(evidence, rows)
    _qualitative_figure(registry)
    print(
        json.dumps(
            {
                "figures": [
                    "paper/figures/floorplan_protocol_overview_v1.pdf",
                    "paper/figures/predicted_control_operation_panels_v1.pdf",
                    "paper/figures/floorplan_failure_taxonomy_v1.pdf",
                    "paper/figures/floorplan_qualitative_cases_v1.pdf",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
