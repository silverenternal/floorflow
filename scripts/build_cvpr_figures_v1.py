#!/usr/bin/env python3
"""Build the frozen CVPR figure and table skeleton from evidence reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

COLORS = {
    "blue": "#2f6fb6",
    "teal": "#16837a",
    "green": "#3f8f5f",
    "amber": "#b77a00",
    "red": "#b84a5a",
    "purple": "#7254a7",
    "gray": "#526173",
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def card(ax, xy, width, height, title, body, color) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=color,
            edgecolor=COLORS["gray"],
            linewidth=0.8,
            alpha=0.14,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        color=COLORS["gray"],
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )


def arrow(ax, start, end, color=COLORS["gray"]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
        )
    )


def architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card(ax, (0.02, 0.55), 0.18, 0.27, "Inputs", "contour + floor + style", COLORS["blue"])
    card(ax, (0.27, 0.55), 0.20, 0.27, "Shared MiTA-CFM", "contour-conditioned backbone", COLORS["purple"])
    card(ax, (0.54, 0.66), 0.18, 0.18, "Pixel head", "class distribution", COLORS["teal"])
    card(ax, (0.54, 0.40), 0.18, 0.18, "Element head", "query distribution", COLORS["green"])
    card(ax, (0.76, 0.55), 0.21, 0.28, "4-step CFM", "multi-channel plan", COLORS["amber"])
    card(ax, (0.76, 0.12), 0.21, 0.25, "Design gate", "legal + diverse + styled", COLORS["red"])
    card(ax, (0.28, 0.12), 0.20, 0.25, "Hierarchy", "wall>opening>room>furniture", COLORS["green"])
    arrow(ax, (0.20, 0.70), (0.27, 0.70), COLORS["blue"])
    arrow(ax, (0.47, 0.70), (0.54, 0.74), COLORS["purple"])
    arrow(ax, (0.47, 0.66), (0.54, 0.49), COLORS["purple"])
    arrow(ax, (0.63, 0.74), (0.76, 0.72), COLORS["teal"])
    arrow(ax, (0.63, 0.49), (0.76, 0.63), COLORS["green"])
    arrow(ax, (0.86, 0.55), (0.86, 0.37), COLORS["amber"])
    arrow(ax, (0.38, 0.37), (0.54, 0.40), COLORS["green"])
    arrow(ax, (0.38, 0.37), (0.54, 0.66), COLORS["green"])
    ax.text(
        0.50,
        0.99,
        "Style is an explicit deployment condition; target control is never used",
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["gray"],
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def qualitative(example_dir: Path, output: Path) -> None:
    paths = sorted(example_dir.glob("*.png"))[:4]
    if len(paths) < 4:
        raise FileNotFoundError("need at least four qualitative example panels")
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width * 2, height * 2), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    canvas.save(output)


def metrics_plot(gate: dict, output: Path) -> None:
    metrics = gate["metrics"]
    entries = [
        ("Room coverage", metrics["legality_room_coverage"]),
        ("Opening alignment", metrics["legality_opening_wall_alignment"]),
        ("Furniture containment", metrics["legality_furniture_room_containment"]),
        ("Conditional watertight", metrics["conditional_watertight"]),
    ]
    labels = [entry[0] for entry in entries][::-1]
    values = [entry[1] for entry in entries][::-1]
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    bars = ax.barh(labels, values, color=[COLORS["blue"], COLORS["green"], COLORS["teal"], COLORS["amber"]][::-1])
    ax.axvline(0.95, color=COLORS["red"], linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("score")
    ax.set_title("Constrained-design legality on CubiCasa test503")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(min(1.0, value + 0.01), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def tradeoff_plot(cubi: dict, skv4: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    offsets = {"CubiCasa": COLORS["blue"], "skv4": COLORS["amber"]}
    for dataset, payload in (("CubiCasa", cubi), ("skv4", skv4)):
        styles = list(payload["styles"].items())
        for index, (style_id, row) in enumerate(styles):
            quality = row.get("room_iou", row.get("room_iou_mean"))
            coverage = row.get("coverage")
            ax.scatter(coverage, quality, color=offsets[dataset], s=28, alpha=0.8)
            vertical_offset = (index - (len(styles) - 1) / 2) * 12
            horizontal_offset = 8 if index % 2 == 0 else -30
            ax.annotate(
                f"{dataset[0]}{style_id}",
                (coverage, quality),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                fontsize=6,
                color=offsets[dataset],
            )
    ax.set_xlabel("style coverage")
    ax.set_ylabel("Room IoU (diagnostic)")
    ax.set_title("Quality-coverage trade-off across styles")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.20, 0.84)
    ax.set_ylim(0.095, 0.132)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def tables(gate: dict, cubi: dict, skv4: dict) -> None:
    lines = [
        "# Main result table",
        "",
        "| Dataset | Split | Room IoU | Worst coverage | Min density |",
        "|---|---|---:|---:|---:|",
        f"| CubiCasa | test503 | {cubi['aggregate']['room_iou_mean']:.5f} | {cubi['aggregate']['coverage_worst']:.5f} | {cubi['aggregate']['density_min']:.5f} |",
        f"| skv4 | test296 | {skv4['aggregate']['room_iou_mean']:.5f} | {skv4['aggregate']['coverage_worst']:.5f} | {skv4['aggregate']['density_min']:.5f} |",
        "",
        "# Legality table",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Room coverage | {gate['metrics']['legality_room_coverage']:.5f} |",
        f"| Opening-wall alignment | {gate['metrics']['legality_opening_wall_alignment']:.5f} |",
        f"| Furniture-room containment | {gate['metrics']['legality_furniture_room_containment']:.5f} |",
        f"| Conditional watertight | {gate['metrics']['conditional_watertight']:.5f} |",
        f"| Unconditional watertight | {gate['metrics']['unconditional_watertight']:.5f} |",
    ]
    Path("paper/tables/main_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    captions = [
        "# Figure and table captions",
        "",
        (
            "**Fig. 1.** Contour and explicit style condition a shared MiTA-CFM backbone. "
            "Pixel and element distributions are coupled by a hierarchy; the final "
            "single-sample four-step flow is evaluated by a constrained-design gate."
        ),
        "",
        (
            "**Fig. 2.** Fixed qualitative examples show contour, reference, and "
            "style-conditioned generation. The claim is legal style-controlled design, "
            "not exact reconstruction of the reference interior."
        ),
        "",
        (
            "**Fig. 3.** Test503 legality metrics. The dashed line marks 0.95; "
            "conditional watertight is reported separately from unconditional watertight."
        ),
        "",
        (
            "**Fig. 4.** Quality-coverage trade-off for every frozen style. Room IoU is "
            "diagnostic only; style coverage is a primary distributional measure."
        ),
    ]
    Path("paper/tables/captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    gate = load("outputs/reports/style_design_gate_v1.json")
    cubi = load("outputs/reports/distribution_style_prompt_r94_v2_all_styles_test503.json")
    skv4 = load("outputs/reports/distribution_skv4_style_v1_calibrated_test296.json")
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    architecture(Path("paper/figures/fig1_architecture"))
    qualitative(
        Path("outputs/eval/style_design_v1/examples"),
        Path("paper/figures/fig2_qualitative.png"),
    )
    metrics_plot(gate, Path("paper/figures/fig3_legality"))
    tradeoff_plot(cubi, skv4, Path("paper/figures/fig4_tradeoff"))
    tables(gate, cubi, skv4)
    print(
        json.dumps(
            {
                "figures": sorted(str(path) for path in Path("paper/figures").glob("*")),
                "tables": [
                    "paper/tables/main_results.md",
                    "paper/tables/captions.md",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
