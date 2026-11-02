from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOORFLOW = ROOT / "src" / "floorflow"


def test_flat_modules_are_removed_and_package_map_is_acyclic() -> None:
    allowed = {"__init__.py", "_compat.py", "_module_map.py"}
    flat = {path.name for path in FLOORFLOW.glob("*.py")} - allowed
    assert flat == set()

    package_map = json.loads((FLOORFLOW / "package_map.json").read_text())
    assert package_map["cycles"] == []
    assert package_map["module_count"] >= 88
    assert package_map["uncovered_legacy_aliases"] == []


def test_legacy_import_aliases_share_module_identity() -> None:
    old = importlib.import_module("floorflow.conditioning")
    new = importlib.import_module("floorflow.representation.conditioning")
    assert old is new


def test_every_floorflow_python_file_is_at_most_600_lines() -> None:
    oversized = {
        str(path.relative_to(ROOT)): len(path.read_text().splitlines())
        for path in FLOORFLOW.rglob("*.py")
        if len(path.read_text().splitlines()) > 600
    }
    assert oversized == {}
