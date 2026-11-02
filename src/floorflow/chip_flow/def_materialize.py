"""Materialize model macro placements into a ChiPBench-style DEF."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

COMPONENT = re.compile(r"^(\s*-\s+)(\S+)(\s+)(\S+)(\s+)\+.*;\s*$")


def load_macro_boxes(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = {row["name"]: row for row in payload["macros"]}
    if len(boxes) != len(payload["macros"]):
        raise ValueError("candidate JSON contains duplicate macro names")
    return boxes


def materialize_macro_def(
    source: Path,
    destination: Path,
    boxes: dict[str, dict[str, Any]],
    *,
    dbu_per_micron: int = 2000,
    remove_blockages: bool = False,
) -> dict[str, Any]:
    """Stream a DEF file while replacing only supplied macro locations."""

    if dbu_per_micron <= 0:
        raise ValueError("dbu_per_micron must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    in_components = False
    skip_blockages = False
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as target:
        temporary = Path(target.name)
        try:
            with source.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip()
                    if remove_blockages and stripped.startswith("BLOCKAGES "):
                        skip_blockages = True
                        continue
                    if skip_blockages:
                        if stripped == "END BLOCKAGES":
                            skip_blockages = False
                        continue
                    if stripped.startswith("COMPONENTS "):
                        in_components = True
                    elif stripped == "END COMPONENTS":
                        in_components = False
                    if in_components:
                        match = COMPONENT.match(line.rstrip("\n"))
                        if match is not None and match.group(2) in boxes:
                            name = match.group(2)
                            row = boxes[name]
                            x = round(float(row["x_um"]) * dbu_per_micron)
                            y = round(float(row["y_um"]) * dbu_per_micron)
                            orientation = str(row.get("orientation", "N"))
                            line = (
                                f"{match.group(1)}{name}{match.group(3)}"
                                f"{match.group(4)} + FIXED "
                                f"( {x} {y} ) {orientation} ;\n"
                            )
                            found.add(name)
                    target.write(line)
            target.flush()
            os.fsync(target.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    missing = sorted(set(boxes) - found)
    extra = sorted(found - set(boxes))
    if missing or extra:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"macro materialization mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    temporary.replace(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "macro_count": len(found),
        "blockages_removed": remove_blockages,
        "dbu_per_micron": dbu_per_micron,
    }


__all__ = ["load_macro_boxes", "materialize_macro_def"]
