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
OPERATION_REPORT = ROOT / "outputs/reports/predicted_operation_success_val191_v1.json"

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


def _predicted_evidence_figure(operation: dict) -> None:
    summary = operation["summary"]
    supported = [
        ("Room add", summary["room_addition"]["operation_success_rate"]),
        ("Room type", summary["room_type_change"]["operation_success_rate"]),
        ("Room remove", summary["room_deletion"]["operation_success_rate"]),
        ("Opening", summary["opening_pressure"]["operation_success_rate"]),
        ("Furniture", summary["furniture_low_salience"]["operation_success_rate"]),
    ]
    plausibility = [
        ("Raster", 0.98493),
        ("Plan", 0.98656),
        ("Functional\nwatertight", 0.99796),
        ("Wall\ncontinuity", 0.99543),
        ("Raw\nwatertight", 0.81670),
    ]
    safety = [
        ("Supported\nexecuted", 1.0),
        ("Unsupported\nabstained", 1.0),
        ("Outside\nexact", 1.0),
        ("Validity", 1.0),
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
        "Preliminary val191 audit; target-free controls; model and held-out test not frozen.",
        ha="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "predicted_control_operation_panels_v1")


def _failure_figure(operation: dict) -> None:
    summary = operation["summary"]
    names = [
        "room_addition",
        "room_type_change",
        "room_deletion",
        "opening_pressure",
        "furniture_low_salience",
    ]
    labels = ["Room add", "Room type", "Room remove", "Opening", "Furniture"]
    successes = [summary[name]["successful_prompts"] for name in names]
    totals = [summary[name]["supported_prompts"] for name in names]
    failures = [total - success for total, success in zip(totals, successes, strict=True)]

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
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
    fig.text(
        0.5,
        0.01,
        "Failures are retained; safety and semantic success are separate claims.",
        ha="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "floorplan_failure_taxonomy_v1")


def main() -> None:
    _style()
    operation = json.loads(OPERATION_REPORT.read_text(encoding="utf-8"))
    _protocol_figure()
    _predicted_evidence_figure(operation)
    _failure_figure(operation)
    print(
        json.dumps(
            {
                "figures": [
                    "paper/figures/floorplan_protocol_overview_v1.pdf",
                    "paper/figures/predicted_control_operation_panels_v1.pdf",
                    "paper/figures/floorplan_failure_taxonomy_v1.pdf",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
