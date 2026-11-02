"""Contracts for the PlaceReasoner-Beta external routed-PPA benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlaceReasonerTask:
    """One design/aspect task in the external benchmark."""

    design: str
    top: str
    aspect_id: str
    floorplan_def: str
    macro_count: int
    source_family: str

    @property
    def task_id(self) -> str:
        return f"{self.design}/{self.aspect_id}"


def load_benchmark_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("unique_designs") != len(payload.get("designs", [])):
        raise ValueError("unique design count does not match the design list")
    if payload.get("placement_tasks") != (
        len(payload["designs"]) * len(payload["aspect_tasks"])
    ):
        raise ValueError("placement task count does not match designs x aspects")
    return payload


def benchmark_tasks(contract: dict[str, Any]) -> tuple[PlaceReasonerTask, ...]:
    tasks = []
    for design in contract["designs"]:
        for aspect in contract["aspect_tasks"]:
            tasks.append(
                PlaceReasonerTask(
                    design=design["name"],
                    top=design["top"],
                    aspect_id=aspect["id"],
                    floorplan_def=aspect["def"],
                    macro_count=int(design["macros"]),
                    source_family=design["name"],
                )
            )
    return tuple(tasks)


def validate_benchmark_root(root: Path, tasks: tuple[PlaceReasonerTask, ...]) -> list[str]:
    errors = []
    for task in tasks:
        design_dir = root / task.design
        required = (
            design_dir / f"{task.design}.v",
            design_dir / "constraint.sdc",
            design_dir / task.floorplan_def,
            design_dir / "lef",
            design_dir / "lib",
        )
        errors.extend(f"missing: {path}" for path in required if not path.exists())
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_coordinates(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("coords", payload.get("entries", payload.get("macros")))
    if not isinstance(payload, list) or not payload:
        raise ValueError("coordinates must be a non-empty list or a dict with coords")
    normalized = []
    seen = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise TypeError(f"coordinate row {index} is not an object")
        try:
            name = str(row["name"])
            x = float(row.get("x", row.get("x_um")))
            y = float(row.get("y", row.get("y_um")))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"coordinate row {index} lacks name/x/y") from error
        if not name or name in seen:
            raise ValueError(f"coordinate row {index} has an empty or duplicate name")
        seen.add(name)
        normalized.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "orient": str(row.get("orient", "R0")),
            }
        )
    return normalized


def validate_coordinate_count(
    coordinates: list[dict],
    expected: int,
) -> None:
    if len(coordinates) != expected:
        raise ValueError(
            f"coordinate count {len(coordinates)} does not match expected macros {expected}"
        )


__all__ = [
    "PlaceReasonerTask",
    "benchmark_tasks",
    "load_benchmark_contract",
    "normalize_coordinates",
    "sha256_file",
    "validate_benchmark_root",
    "validate_coordinate_count",
]
