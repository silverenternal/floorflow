"""Audit the floorflow feature-package layout and emit package_map.json."""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FLOORFLOW = SRC / "floorflow"
OUTPUT = FLOORFLOW / "package_map.json"

sys.path.insert(0, str(SRC))
from floorflow._module_map import MODULE_MAP  # noqa: E402


def module_name(path: Path) -> str:
    relative = path.relative_to(SRC).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def canonical_module(name: str, known: set[str]) -> str | None:
    if name in known:
        return name
    prefix = "floorflow."
    if name.startswith(prefix):
        short = name[len(prefix):]
        return MODULE_MAP.get(short)
    return None


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        if node in visiting:
            start = stack.index(node)
            cycles.append(stack[start:] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for dependency in sorted(graph.get(node, ())):
            visit(dependency)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)
    return cycles


def main() -> int:
    paths = sorted(path for path in FLOORFLOW.rglob("*.py") if path.name != "__init__.py")
    known = {module_name(path) for path in paths}
    graph = {
        module_name(path): {
            dependency
            for imported in resolve_imports(path)
            if (dependency := canonical_module(imported, known)) and dependency != module_name(path)
        }
        for path in paths
    }
    cycles = find_cycles(graph)
    flat = sorted(
        path.name for path in FLOORFLOW.glob("*.py")
        if path.name not in {"__init__.py", "_compat.py", "_module_map.py"}
    )
    uncovered = sorted(
        f"floorflow.{legacy} -> {target}"
        for legacy, target in MODULE_MAP.items()
        if target not in known
    )
    payload = {
        "schema_version": 1,
        "root": "src/floorflow",
        "module_count": len(paths),
        "packages": sorted({name.split(".")[1] for name in known if name.count(".") >= 2}),
        "flat_module_files": flat,
        "uncovered_legacy_aliases": uncovered,
        "cycles": cycles,
        "modules": {
            name: {
                "path": str(paths[[module_name(p) for p in paths].index(name)].relative_to(ROOT)),
                "package": name.split(".")[1] if name.count(".") >= 2 else "",
                "dependencies": sorted(graph[name]),
            }
            for name in sorted(graph)
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "module_count": len(paths),
                "packages": payload["packages"],
                "flat_module_files": flat,
                "uncovered_legacy_aliases": uncovered,
                "cycles": cycles,
            },
            indent=2,
        )
    )
    return 1 if flat or uncovered or cycles else 0


if __name__ == "__main__":
    raise SystemExit(main())
