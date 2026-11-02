"""Align macro origins to routing-track residues from a reference DEF."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

COMPONENT = re.compile(
    r"^\s*-\s+(\S+)\s+(\S+)\s+\+.*"
    r"\(\s*([0-9-]+)\s+([0-9-]+)\s*\)\s*(\S+)\s*;\s*$"
)
DIEAREA = re.compile(
    r"^\s*DIEAREA\s+\(\s*([0-9-]+)\s+([0-9-]+)\s*\)"
    r"\s+\(\s*([0-9-]+)\s+([0-9-]+)\s*\)\s*;"
)
UNITS = re.compile(r"^\s*UNITS\s+DISTANCE\s+MICRONS\s+([0-9]+)\s*;")


def _reference_rows(path: Path, names: set[str]) -> tuple[dict, tuple[int, ...], int]:
    rows: dict[str, tuple[int, int, str]] = {}
    diearea: tuple[int, ...] | None = None
    dbu = 0
    for line in path.open(encoding="utf-8", errors="replace"):
        units = UNITS.match(line)
        if units:
            dbu = int(units.group(1))
        area = DIEAREA.match(line)
        if area:
            diearea = tuple(int(value) for value in area.groups())
        component = COMPONENT.match(line)
        if component and component.group(1) in names:
            rows[component.group(1)] = (
                int(component.group(3)),
                int(component.group(4)),
                component.group(5),
            )
    if dbu <= 0 or diearea is None:
        raise ValueError("reference DEF lacks units or die area")
    return rows, diearea, dbu


def align_boxes_to_reference(
    boxes: dict[str, dict[str, Any]],
    reference_def: Path,
    *,
    grid_dbu: int,
    dbu_per_micron: int = 2000,
    rollback_conflicts: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Move each macro to the nearest matching reference residue."""

    if grid_dbu <= 0 or dbu_per_micron <= 0:
        raise ValueError("grid and DEF units must be positive")
    references, diearea, reference_dbu = _reference_rows(
        reference_def,
        set(boxes),
    )
    if reference_dbu != dbu_per_micron:
        raise ValueError(
            f"reference DEF units {reference_dbu} != {dbu_per_micron}"
        )
    missing = sorted(set(boxes) - set(references))
    if missing:
        raise ValueError(f"reference DEF is missing macros: {missing[:5]}")
    original = {name: dict(row) for name, row in boxes.items()}
    aligned = {name: dict(row) for name, row in boxes.items()}
    candidates = []
    orientation_mismatch = []
    for name, row in original.items():
        reference_x, reference_y, reference_orientation = references[name]
        if str(row.get("orientation", "N")) != reference_orientation:
            orientation_mismatch.append(name)
            continue
        x = round(float(row["x_um"]) * dbu_per_micron)
        y = round(float(row["y_um"]) * dbu_per_micron)
        target_x = reference_x + round((x - reference_x) / grid_dbu) * grid_dbu
        target_y = reference_y + round((y - reference_y) / grid_dbu) * grid_dbu
        candidates.append(
            (
                name,
                target_x / dbu_per_micron,
                target_y / dbu_per_micron,
                abs(target_x - x) / dbu_per_micron,
                abs(target_y - y) / dbu_per_micron,
            )
        )
    if orientation_mismatch:
        raise ValueError(
            "reference orientation mismatch prevents pin alignment: "
            f"{orientation_mismatch[:5]}"
        )
    x0, y0, x1, y1 = diearea

    def valid(layout: dict[str, dict[str, Any]]) -> bool:
        items = sorted(layout.items())
        for name, row in items:
            left = float(row["x_um"]) * dbu_per_micron
            bottom = float(row["y_um"]) * dbu_per_micron
            right = left + float(row["width_um"]) * dbu_per_micron
            top = bottom + float(row["height_um"]) * dbu_per_micron
            if left < x0 or bottom < y0 or right > x1 or top > y1:
                return False
        for index, (_, left) in enumerate(items):
            lx, ly = float(left["x_um"]), float(left["y_um"])
            lr = lx + float(left["width_um"])
            lt = ly + float(left["height_um"])
            for _, right in items[index + 1 :]:
                rx, ry = float(right["x_um"]), float(right["y_um"])
                rr = rx + float(right["width_um"])
                rt = ry + float(right["height_um"])
                if (
                    lx < rr - 1.0e-9
                    and lr > rx + 1.0e-9
                    and ly < rt - 1.0e-9
                    and lt > ry + 1.0e-9
                ):
                    return False
        return True

    shifts = []
    rolled_back = []
    accepted = set()
    for name, target_x_um, target_y_um, shift_x, shift_y in sorted(
        candidates,
        key=lambda item: item[3] + item[4],
    ):
        trial = {key: dict(value) for key, value in aligned.items()}
        trial[name]["x_um"] = target_x_um
        trial[name]["y_um"] = target_y_um
        if rollback_conflicts and not valid(trial):
            rolled_back.append(name)
            continue
        aligned = trial
        accepted.add(name)
        shifts.extend([shift_x, shift_y])
    if not rollback_conflicts:
        for name in set(aligned) - accepted:
            row = aligned[name]
            reference_x, reference_y, _ = references[name]
            x = round(float(original[name]["x_um"]) * dbu_per_micron)
            y = round(float(original[name]["y_um"]) * dbu_per_micron)
            target_x = reference_x + round((x - reference_x) / grid_dbu) * grid_dbu
            target_y = reference_y + round((y - reference_y) / grid_dbu) * grid_dbu
            row["x_um"] = target_x / dbu_per_micron
            row["y_um"] = target_y / dbu_per_micron
            shifts.extend(
                [
                    abs(target_x - x) / dbu_per_micron,
                    abs(target_y - y) / dbu_per_micron,
                ]
            )
    outside = []
    items = sorted(aligned.items())
    for name, row in items:
        left = float(row["x_um"]) * dbu_per_micron
        bottom = float(row["y_um"]) * dbu_per_micron
        right = left + float(row["width_um"]) * dbu_per_micron
        top = bottom + float(row["height_um"]) * dbu_per_micron
        if left < x0 or bottom < y0 or right > x1 or top > y1:
            outside.append(name)
    overlaps = []
    for index, (left_name, left) in enumerate(items):
        lx, ly = float(left["x_um"]), float(left["y_um"])
        lr = lx + float(left["width_um"])
        lt = ly + float(left["height_um"])
        for right_name, right in items[index + 1 :]:
            rx, ry = float(right["x_um"]), float(right["y_um"])
            rr = rx + float(right["width_um"])
            rt = ry + float(right["height_um"])
            if (
                lx < rr - 1.0e-9
                and lr > rx + 1.0e-9
                and ly < rt - 1.0e-9
                and lt > ry + 1.0e-9
            ):
                overlaps.append([left_name, right_name])
    if outside or overlaps:
        raise ValueError(
            "pin alignment is infeasible: "
            f"outside={outside[:5]} overlaps={overlaps[:5]}"
        )
    return aligned, {
        "status": "pass",
        "grid_dbu": grid_dbu,
        "macro_count": len(aligned),
        "aligned_macro_count": len(accepted),
        "rolled_back_macro_count": len(rolled_back),
        "rolled_back_macros": rolled_back,
        "max_shift_um": max(shifts, default=0.0),
        "mean_shift_um": sum(shifts) / len(shifts) if shifts else 0.0,
        "overlap_pair_count": 0,
        "outside_macro_count": 0,
    }


__all__ = ["align_boxes_to_reference"]
