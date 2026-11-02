"""Enrich external macro graphs with target-free semantic contracts."""

from __future__ import annotations

import math
from typing import Any

import torch

from floorflow.chip_flow.global_spec import global_spec_vector

ORIENTATIONS = ("N", "S", "E", "W", "FN", "FS", "FE", "FW")
ORIENTATION_ALIASES = {
    "N": "N",
    "R0": "N",
    "S": "S",
    "R180": "S",
    "E": "E",
    "R270": "E",
    "W": "W",
    "R90": "W",
    "FN": "FN",
    "MY": "FN",
    "FS": "FS",
    "MX": "FS",
    "FE": "FE",
    "MY90": "FE",
    "FW": "FW",
    "MX90": "FW",
}
NODE_SEMANTIC_DIM = len(ORIENTATIONS) + 4
PORT_SEMANTIC_DIM = 5


def normalize_orientation(value: str) -> str:
    """Canonicalize common DEF and ORFS orientation aliases."""

    normalized = ORIENTATION_ALIASES.get(str(value).upper())
    if normalized is None:
        raise ValueError(f"unsupported macro orientation: {value}")
    return normalized


def external_node_features(graph: Any) -> torch.Tensor:
    macros = tuple(graph.macros)
    if not macros:
        return torch.zeros(0, NODE_SEMANTIC_DIM)
    widths = torch.tensor(
        [macro["width_dbu"] / graph.dbu_per_micron for macro in macros],
        dtype=torch.float32,
    )
    heights = torch.tensor(
        [macro["height_dbu"] / graph.dbu_per_micron for macro in macros],
        dtype=torch.float32,
    )
    area_scale = (widths * heights).sum().sqrt().clamp_min(1.0e-6)
    orientation = torch.zeros(len(macros), len(ORIENTATIONS))
    for row, macro in enumerate(macros):
        value = normalize_orientation(macro.get("orientation", "N"))
        orientation[row, ORIENTATIONS.index(value)] = 1.0
    status = [str(macro.get("placement_status", "PLACED")) for macro in macros]
    return torch.cat(
        [
            orientation,
            torch.tensor([value == "FIXED" for value in status]).float()[:, None],
            torch.tensor([value == "UNPLACED" for value in status]).float()[:, None],
            (widths / area_scale)[:, None],
            (heights / area_scale)[:, None],
        ],
        dim=-1,
    )


def external_port_features(graph: Any, names: list[str]) -> torch.Tensor:
    if len(names) != len(set(names)):
        raise ValueError("external port names must be unique")
    missing = set(names) - set(graph.ports)
    if missing:
        raise ValueError(f"external port features reference missing ports: {missing}")
    features = torch.zeros(len(names), PORT_SEMANTIC_DIM)
    x0, y0, x1, y1 = graph.core
    tolerance = 1.0e-6 * max(x1 - x0, y1 - y0, 1.0)
    for row, name in enumerate(names):
        x, y = graph.ports[name]
        known = getattr(graph, "port_features", {}).get(name)
        if known is not None:
            values = torch.as_tensor(known, dtype=torch.float32)
            if values.numel() > PORT_SEMANTIC_DIM:
                values = values[:PORT_SEMANTIC_DIM]
            features[row, : values.numel()] = values
            continue
        if abs(x - x0) <= tolerance:
            features[row, 0] = 1.0
        elif abs(x - x1) <= tolerance:
            features[row, 1] = 1.0
        elif abs(y - y0) <= tolerance:
            features[row, 2] = 1.0
        elif abs(y - y1) <= tolerance:
            features[row, 3] = 1.0
        else:
            features[row, 4] = 1.0
    return features


def external_used_ports(graph: Any) -> list[str]:
    """Return ports on nets that also connect at least one exported macro."""

    macro_names = {macro["name"] for macro in graph.macros}
    used = {
        port
        for net, ports in graph.net_ports.items()
        if any(name in macro_names for name in graph.net_macros.get(net, set()))
        for port in ports
        if port in graph.ports
    }
    return sorted(used)


def external_netlist_hypergraph(
    graph: Any,
    port_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    index = {macro["name"]: row for row, macro in enumerate(graph.macros)}
    ports_used = (
        sorted(graph.ports)
        if port_names is None
        else list(port_names)
    )
    if len(ports_used) != len(set(ports_used)):
        raise ValueError("external hypergraph port names must be unique")
    if set(ports_used) - set(graph.ports):
        raise ValueError("external hypergraph references missing ports")
    port_index = {name: row for row, name in enumerate(ports_used)}
    nets = sorted(set(graph.net_macros) | set(graph.net_ports))
    if not nets:
        return {
            "net_features": torch.zeros(1, 0, 4),
            "net_block_incidence": torch.zeros(
                1,
                0,
                len(graph.macros),
                dtype=torch.bool,
            ),
            "net_port_incidence": torch.zeros(
                1,
                0,
                len(ports_used),
                dtype=torch.bool,
            ),
            "net_mask": torch.zeros(1, 0, dtype=torch.bool),
        }
    incidence = torch.zeros(len(nets), len(index), dtype=torch.bool)
    port_incidence = torch.zeros(len(nets), len(port_index), dtype=torch.bool)
    features = torch.zeros(len(nets), 4)
    core_area = math.prod(graph.core_size)
    macro_area = sum(
        macro["width_dbu"] * macro["height_dbu"]
        / (graph.dbu_per_micron**2)
        for macro in graph.macros
    )
    for net_row, net in enumerate(nets):
        macros = [
            index[name]
            for name in sorted(graph.net_macros.get(net, set()))
            if name in index
        ]
        ports = [
            port_index[name]
            for name in sorted(graph.net_ports.get(net, set()))
            if name in port_index
        ]
        incidence[net_row, macros] = True
        port_incidence[net_row, ports] = True
        member_area = sum(
            graph.macros[row]["width_dbu"]
            * graph.macros[row]["height_dbu"]
            / (graph.dbu_per_micron**2)
            for row in macros
        )
        features[net_row] = torch.tensor(
            [
                math.log1p(len(macros)),
                float(bool(ports)),
                math.log1p(len(ports)),
                member_area / max(core_area, macro_area, 1.0e-9),
            ]
        )
    return {
        "net_features": features[None],
        "net_block_incidence": incidence[None],
        "net_port_incidence": port_incidence[None],
        "net_mask": torch.ones(1, len(nets), dtype=torch.bool),
    }


def apply_external_conditioning(graph: Any, batch: dict) -> dict:
    """Attach only target-free design/netlist semantics to an external batch."""

    core_width, core_height = graph.core_size
    die_width = graph.die[2] - graph.die[0]
    die_height = graph.die[3] - graph.die[1]
    macro_area = torch.tensor(
        sum(
            macro["width_dbu"]
            * macro["height_dbu"]
            / (graph.dbu_per_micron**2)
            for macro in graph.macros
        ),
        dtype=torch.float32,
    )
    conditioned = dict(batch)
    conditioned["global_spec"] = global_spec_vector(
        core_width=torch.tensor(core_width),
        core_height=torch.tensor(core_height),
        die_width=torch.tensor(die_width),
        die_height=torch.tensor(die_height),
        macro_area=macro_area,
    )[None]
    conditioned["global_spec_mask"] = torch.ones(1, dtype=torch.bool)
    conditioned["node_features"] = external_node_features(graph)[None]
    conditioned["node_feature_mask"] = torch.ones(
        1,
        len(graph.macros),
        dtype=torch.bool,
    )
    ports = external_used_ports(graph)
    conditioned["port_features"] = external_port_features(graph, ports)[None]
    conditioned["port_feature_mask"] = torch.ones(
        1,
        len(ports),
        dtype=torch.bool,
    )
    conditioned.update(external_netlist_hypergraph(graph, ports))
    device = batch["areas"].device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in conditioned.items()
    }


__all__ = [
    "NODE_SEMANTIC_DIM",
    "ORIENTATIONS",
    "ORIENTATION_ALIASES",
    "PORT_SEMANTIC_DIM",
    "apply_external_conditioning",
    "external_netlist_hypergraph",
    "external_node_features",
    "external_port_features",
    "external_used_ports",
    "normalize_orientation",
]
