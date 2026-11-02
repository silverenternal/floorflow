"""Parse DEF/LEF macro graphs without launching an EDA process."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from floorflow.chip_flow.external_graph_conditioning import normalize_orientation
from floorflow.chip_flow.ppa_external import ExternalMacroGraph


def _section(text: str, name: str) -> str:
    match = re.search(
        rf"^{name}\s+\d+\s*;(.*?)^END {name}\b",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"DEF section missing: {name}")
    return match.group(1)


def _lef_macro_sizes(paths: list[Path]) -> dict[str, tuple[int, int]]:
    text = "\n".join(path.read_text(errors="replace") for path in paths)
    sizes: dict[str, tuple[int, int]] = {}
    for block in re.findall(
        r"^MACRO\s+(\S+)(.*?)^END\s+\1\b",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        name, body = block
        size = re.search(
            r"^\s*SIZE\s+([0-9.eE+-]+)\s+BY\s+([0-9.eE+-]+)\s*;",
            body,
            re.MULTILINE,
        )
        if size is not None:
            sizes[name] = (float(size.group(1)), float(size.group(2)))
    return sizes


def _units(text: str) -> int:
    match = re.search(
        r"^UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("DEF distance units are missing")
    return int(match.group(1))


def _die(text: str) -> tuple[int, int, int, int]:
    match = re.search(
        r"^DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)"
        r"\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("DEF DIEAREA is missing")
    return tuple(int(value) for value in match.groups())


def _core(text: str) -> tuple[int, int, int, int]:
    rows = re.findall(
        r"^ROW\s+\S+\s+\S+\s+(-?\d+)\s+(-?\d+)\s+\S+"
        r"\s+DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(-?\d+)\s+(-?\d+)\s*;",
        text,
        re.MULTILINE,
    )
    if not rows:
        raise ValueError("DEF ROW geometry is missing")
    x0 = min(int(row[0]) for row in rows)
    y0 = min(int(row[1]) for row in rows)
    x1 = max(
        int(row[0]) + int(row[2]) * int(row[4])
        for row in rows
    )
    y1 = max(
        int(row[1]) + int(row[3]) * int(row[5])
        for row in rows
    )
    return x0, y0, x1, y1


def _placement(statement: str) -> tuple[int, int, str, str]:
    match = re.search(
        r"\+\s+(PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)"
        r"\s+(\S+)",
        statement,
    )
    if match is None:
        return 0, 0, "UNPLACED", "N"
    status, x, y, orientation = match.groups()
    return int(x), int(y), status, orientation


def _macro_components(
    section: str,
    masters: set[str],
) -> dict[str, dict[str, Any]]:
    macros = {}
    for statement in section.split(";"):
        match = re.match(r"\s*-\s+(\S+)\s+(\S+)\b", statement)
        if match is None or match.group(2) not in masters:
            continue
        name, master = match.groups()
        x, y, status, orientation = _placement(statement)
        macros[name] = {
            "name": name,
            "master": master,
            "x_dbu": x,
            "y_dbu": y,
            "placement_status": status,
            "orientation": orientation,
        }
    return macros


def _ports(
    section: str,
    dbu: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, ...]]]:
    ports = {}
    features = {}
    orientation_index = {"N": 0, "S": 1, "E": 2, "W": 3}
    for statement in section.split(";"):
        match = re.match(r"\s*-\s+(\S+)\b", statement)
        if match is None:
            continue
        x, y, status, orientation = _placement(statement)
        if status != "UNPLACED":
            ports[match.group(1)] = (x / dbu, y / dbu)
            semantic = [0.0] * 5
            normalized = normalize_orientation(orientation)
            if normalized in orientation_index:
                semantic[orientation_index[normalized]] = 1.0
            features[match.group(1)] = tuple(semantic)
    return ports, features


def _nets(
    section: str,
    macros: set[str],
    ports: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    net_macros: dict[str, set[str]] = defaultdict(set)
    net_ports: dict[str, set[str]] = defaultdict(set)
    for statement in section.split(";"):
        match = re.match(r"\s*-\s+(\S+)\b", statement)
        if match is None:
            continue
        net = match.group(1)
        for left, right in re.findall(
            r"\(\s*([^\s()]+)\s+([^\s()]+)\s*\)",
            statement,
        ):
            if left == "PIN" and right in ports:
                net_ports[net].add(right)
            elif left in macros:
                net_macros[net].add(left)
    return dict(net_macros), dict(net_ports)


def parse_def_graph(
    def_path: Path,
    macro_lef_paths: list[Path],
) -> ExternalMacroGraph:
    """Build the same graph schema used by the candidate model."""

    text = def_path.read_text(errors="replace")
    dbu = _units(text)
    sizes = _lef_macro_sizes(macro_lef_paths)
    if not sizes:
        raise ValueError("macro LEF files define no macros")
    components = _macro_components(_section(text, "COMPONENTS"), set(sizes))
    if not components:
        raise ValueError("DEF contains no instances of supplied LEF macros")
    for name, macro in components.items():
        width, height = sizes[macro["master"]]
        macro["width_dbu"] = round(width * dbu)
        macro["height_dbu"] = round(height * dbu)
        if macro["width_dbu"] <= 0 or macro["height_dbu"] <= 0:
            raise ValueError(f"non-positive LEF size for {name}")
    ports, port_features = _ports(_section(text, "PINS"), dbu)
    net_macros, net_ports = _nets(
        _section(text, "NETS"),
        set(components),
        set(ports),
    )
    die_dbu = _die(text)
    core_dbu = _core(text)
    return ExternalMacroGraph(
        dbu_per_micron=dbu,
        die=tuple(value / dbu for value in die_dbu),
        core=tuple(value / dbu for value in core_dbu),
        macros=tuple(components[name] for name in sorted(components)),
        ports=ports,
        net_macros=net_macros,
        net_ports=net_ports,
        port_features=port_features,
    )


__all__ = ["parse_def_graph"]
