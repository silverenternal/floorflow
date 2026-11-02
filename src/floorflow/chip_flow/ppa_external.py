"""Reusable bridge from an OpenROAD macro graph to the trained chip model."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from floorflow.chip_flow.external_graph_conditioning import (
    apply_external_conditioning,
    external_used_ports,
)
from floorflow.chip_flow.model import RHO_CLAMP, encode_latent
from floorflow.chip_flow.proposals import shelf_pack


def parse_global_route_overflow(text: str) -> dict[str, int | None]:
    """Parse the final ``Max H / Max V / Total Congestion`` row from ORFS GRT."""

    match = re.search(
        r"^Total\s+.*?\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        return {"max_horizontal": None, "max_vertical": None, "total": None}
    return {
        "max_horizontal": int(match.group(1)),
        "max_vertical": int(match.group(2)),
        "total": int(match.group(3)),
    }


@dataclass(frozen=True)
class ExternalMacroGraph:
    """Macro geometry and macro/port hypergraph exported from ORFS."""

    dbu_per_micron: int
    die: tuple[float, float, float, float]
    core: tuple[float, float, float, float]
    macros: tuple[dict[str, Any], ...]
    ports: dict[str, tuple[float, float]]
    net_macros: dict[str, set[str]]
    net_ports: dict[str, set[str]]
    port_features: dict[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def core_size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.core
        return x1 - x0, y1 - y0

    @property
    def core_origin(self) -> tuple[float, float]:
        return self.core[0], self.core[1]


def parse_orfs_graph(text: str) -> ExternalMacroGraph:
    """Parse the stable TSV emitted by ``orfs_export_macro_graph.tcl``."""

    dbu: int | None = None
    die: tuple[float, float, float, float] | None = None
    core: tuple[float, float, float, float] | None = None
    macros: dict[str, dict[str, Any]] = {}
    ports: dict[str, tuple[float, float]] = {}
    port_features: dict[str, tuple[float, ...]] = {}
    net_macros: dict[str, set[str]] = defaultdict(set)
    net_ports: dict[str, set[str]] = defaultdict(set)
    for raw_line in text.splitlines():
        fields = raw_line.rstrip("\n").split("\t")
        if not fields or not fields[0].startswith("ORFS_"):
            continue
        kind = fields[0]
        if kind == "ORFS_DBU_PER_MICRON":
            dbu = int(fields[1])
        elif kind in {"ORFS_DIE", "ORFS_CORE"}:
            values = tuple(float(value) for value in fields[1:5])
            if kind == "ORFS_DIE":
                die = values
            else:
                core = values
        elif kind == "ORFS_MACRO":
            name, master, width, height, x, y, status, orientation = fields[1:9]
            macros[name] = {
                "name": name,
                "master": master,
                "width_dbu": int(width),
                "height_dbu": int(height),
                "x_dbu": int(x),
                "y_dbu": int(y),
                "placement_status": status,
                "orientation": orientation,
            }
        elif kind == "ORFS_PORT":
            ports[fields[1]] = (float(fields[2]), float(fields[3]))
            if len(fields) > 4:
                port_features[fields[1]] = tuple(
                    float(value) for value in fields[4:]
                )
        elif kind == "ORFS_NET_ITERM":
            net_macros[fields[1]].add(fields[2])
        elif kind == "ORFS_NET_BTERM":
            net_ports[fields[1]].add(fields[2])
    if dbu is None or die is None or core is None:
        raise ValueError("incomplete ORFS geometry header")
    if not macros:
        raise ValueError("ORFS export contains no block-level macros")
    die_microns = tuple(value / dbu for value in die)
    core_microns = tuple(value / dbu for value in core)
    ports_microns = {
        name: (position[0] / dbu, position[1] / dbu)
        for name, position in ports.items()
    }
    return ExternalMacroGraph(
        dbu_per_micron=dbu,
        die=die_microns,
        core=core_microns,
        macros=tuple(macros[name] for name in sorted(macros)),
        ports=ports_microns,
        net_macros=dict(net_macros),
        net_ports=dict(net_ports),
        port_features=port_features,
    )


def graph_edges(graph: ExternalMacroGraph) -> dict[str, Any]:
    """Aggregate hyperedges into weighted b2b and port-to-block edges."""

    index = {macro["name"]: i for i, macro in enumerate(graph.macros)}
    b2b: dict[tuple[int, int], float] = defaultdict(float)
    p2b: dict[tuple[str, int], float] = defaultdict(float)
    used_ports = set(external_used_ports(graph))
    for net, macro_names in graph.net_macros.items():
        blocks = sorted(index[name] for name in macro_names if name in index)
        port_names = sorted(graph.net_ports.get(net, set()))
        if not blocks:
            continue
        if len(blocks) > 1:
            weight = 1.0 / float(len(blocks) - 1)
            for offset, left in enumerate(blocks):
                for right in blocks[offset + 1 :]:
                    b2b[(left, right)] += weight
        if port_names:
            weight = 1.0 / float(len(blocks))
            for port in port_names:
                if port in graph.ports:
                    for block in blocks:
                        p2b[(port, block)] += weight
    return {
        "index": index,
        "b2b": b2b,
        "p2b": p2b,
        "used_ports": sorted(used_ports),
    }


def infer_hierarchy_clusters(
    graph: ExternalMacroGraph,
    cluster_slots: int = 4,
) -> dict[str, int]:
    """Map netlist instance-path prefixes to the model's four cluster slots."""

    groups: dict[str, list[str]] = defaultdict(list)
    for macro in graph.macros:
        parts = macro["name"].split("/")
        key = parts[0]
        if len(parts) > 1:
            child = re.split(r"[._]", parts[1])[0]
            if child in {"be", "fe", "me", "core", "uncore"}:
                key = f"{key}/{child}"
        groups[key].append(macro["name"])
    ranked = sorted(groups, key=lambda key: (-len(groups[key]), key))
    selected = sorted(ranked[:cluster_slots])
    return {
        macro_name: index
        for index, key in enumerate(selected, start=1)
        for macro_name in groups[key]
    }


def _shape_boxes(
    area_norm: torch.Tensor,
    rho: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    centers = boxes[:, :2] + boxes[:, 2:] * 0.5
    width = (area_norm * torch.exp(rho)).sqrt()
    height = area_norm / width.clamp_min(1.0e-12)
    return torch.stack(
        [centers[:, 0] - width / 2, centers[:, 1] - height / 2, width, height],
        dim=1,
    )


def model_batch(
    graph: ExternalMacroGraph,
    device: torch.device,
    cluster_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Create the same tensor schema used by FloorSet checkpoint inference."""

    edges = graph_edges(graph)
    count = len(graph.macros)
    widths = torch.tensor(
        [macro["width_dbu"] / graph.dbu_per_micron for macro in graph.macros],
        dtype=torch.float32,
    )
    heights = torch.tensor(
        [macro["height_dbu"] / graph.dbu_per_micron for macro in graph.macros],
        dtype=torch.float32,
    )
    physical_area = widths * heights
    core_width, core_height = graph.core_size
    area_norm = physical_area * (core_width * core_height / physical_area.sum())
    rho = torch.log(widths / heights).clamp(-RHO_CLAMP, RHO_CLAMP)
    source_square = shelf_pack(area_norm)
    source_boxes = _shape_boxes(area_norm, rho, source_square)
    target_boxes = _shape_boxes(area_norm, rho, source_square)
    source_latent = encode_latent(source_boxes, area_norm)
    target_latent = encode_latent(target_boxes, area_norm)
    b2b_pairs = sorted(edges["b2b"])
    b2b = torch.tensor(
        [[left, right, edges["b2b"][(left, right)]] for left, right in b2b_pairs],
        dtype=torch.float32,
    ).reshape(-1, 3)
    port_names = [name for name in edges["used_ports"] if name in graph.ports]
    port_lookup = {name: index for index, name in enumerate(port_names)}
    p2b = torch.tensor(
        [
            [port_lookup[port], block, weight]
            for (port, block), weight in sorted(edges["p2b"].items())
            if port in port_lookup
        ],
        dtype=torch.float32,
    ).reshape(-1, 3)
    pins_xy = torch.tensor(
        [graph.ports[name] for name in port_names],
        dtype=torch.float32,
    ).reshape(-1, 2)
    origin_x, origin_y = graph.core_origin
    scale = math.sqrt(core_width * core_height)
    pins_xy = torch.stack(
        [
            (pins_xy[:, 0] - origin_x) / scale,
            (pins_xy[:, 1] - origin_y) / scale,
        ],
        dim=1,
    )
    constraints = torch.zeros(count, 5)
    constraints[:, 0] = 1.0
    if cluster_ids:
        for index, macro in enumerate(graph.macros):
            constraints[index, 3] = float(cluster_ids.get(macro["name"], 0))
    batch = {
        "n": count,
        "areas": area_norm[None],
        "boxes": target_boxes[None],
        "constraints": constraints[None],
        "features": torch.zeros(1, count, 19),
        "source_latent": source_latent[None],
        "target_latent": target_latent[None],
        "b2b": b2b[None],
        "p2b": p2b[None],
        "pins": pins_xy[None],
        "n_b2b": torch.tensor([len(b2b)]),
        "n_p2b": torch.tensor([len(p2b)]),
        "n_pins": torch.tensor([len(pins_xy)]),
    }
    tensors = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    return apply_external_conditioning(
        graph,
        tensors,
    )


def guard_small_graph_pe(model: Any, count: int) -> bool:
    """Apply the documented zero-PE behavior for graphs smaller than ``k + 2``.

    The pinned Trinity batched SpectralDrawPE returns fewer columns than its declared
    dimension on these cases. This is a compatibility shim only; no learned parameter
    is changed.
    """

    graph_pe = getattr(model, "graph_pe", None)
    if graph_pe is None or count > int(graph_pe.k) + 1:
        return False

    def zero_pe(adjacency: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        del key_mask
        return adjacency.new_zeros(
            (adjacency.shape[0], adjacency.shape[1], graph_pe.dim)
        )

    graph_pe.batched = zero_pe
    return True


def physical_boxes(
    latent: torch.Tensor,
    graph: ExternalMacroGraph,
) -> np.ndarray:
    """Map model coordinates to physical micron boxes with fixed macro shapes."""

    core_width, core_height = graph.core_size
    scale = math.sqrt(core_width * core_height)
    center = latent[0, :, :2].detach().cpu().numpy() * scale
    center[:, 0] += graph.core[0]
    center[:, 1] += graph.core[1]
    widths = np.asarray(
        [macro["width_dbu"] / graph.dbu_per_micron for macro in graph.macros]
    )
    heights = np.asarray(
        [macro["height_dbu"] / graph.dbu_per_micron for macro in graph.macros]
    )
    return np.stack(
        [
            center[:, 0] - widths / 2,
            center[:, 1] - heights / 2,
            widths,
            heights,
        ],
        axis=1,
    )


def floorplan_instance(
    graph: ExternalMacroGraph,
    halo_um: float = 0.0,
    cluster_ids: dict[str, int] | None = None,
):
    """Build the pinned Trinity instance, optionally inflating fixed shapes."""

    from trinity.floorplan.types import FloorplanInstance

    edges = graph_edges(graph)
    widths = np.asarray(
        [
            macro["width_dbu"] / graph.dbu_per_micron + float(halo_um)
            for macro in graph.macros
        ],
        dtype=np.float64,
    )
    heights = np.asarray(
        [
            macro["height_dbu"] / graph.dbu_per_micron + float(halo_um)
            for macro in graph.macros
        ],
        dtype=np.float64,
    )
    constraints = np.zeros((len(graph.macros), 5), dtype=np.float32)
    constraints[:, 0] = 1.0
    if cluster_ids:
        for index, macro in enumerate(graph.macros):
            constraints[index, 3] = cluster_ids.get(macro["name"], 0)
    target_positions = np.full((len(graph.macros), 4), -1.0, dtype=np.float32)
    target_positions[:, 2] = widths
    target_positions[:, 3] = heights
    port_names = [name for name in edges["used_ports"] if name in graph.ports]
    port_lookup = {name: index for index, name in enumerate(port_names)}
    p2b = np.asarray(
        [
            [port_lookup[port], block, weight]
            for (port, block), weight in sorted(edges["p2b"].items())
            if port in port_lookup
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    b2b = np.asarray(
        [
            [left, right, edges["b2b"][(left, right)]]
            for left, right in sorted(edges["b2b"])
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    pins_pos = np.asarray(
        [graph.ports[name] for name in port_names],
        dtype=np.float64,
    ).reshape(-1, 2)
    return FloorplanInstance(
        block_count=len(graph.macros),
        area_targets=widths * heights,
        constraints=constraints,
        b2b=b2b,
        p2b=p2b,
        pins_pos=pins_pos,
        target_positions=target_positions,
        outline=graph.core_size,
    )


def inflate_boxes(boxes: np.ndarray, halo_um: float) -> np.ndarray:
    """Inflate axis-aligned boxes around their centers for halo-aware packing."""

    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, 0] -= halo_um / 2.0
    out[:, 1] -= halo_um / 2.0
    out[:, 2] += halo_um
    out[:, 3] += halo_um
    return out


def deflate_boxes(boxes: np.ndarray, halo_um: float) -> np.ndarray:
    """Recover the real macro geometry after halo-aware packing."""

    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, 0] += halo_um / 2.0
    out[:, 1] += halo_um / 2.0
    out[:, 2] -= halo_um
    out[:, 3] -= halo_um
    return out


def restore_fixed_dimensions(
    boxes: np.ndarray,
    graph: ExternalMacroGraph,
) -> np.ndarray:
    """Restore exact DBU-derived macro dimensions after halo deflation."""

    out = np.asarray(boxes, dtype=np.float64).copy()
    out[:, 2] = [
        macro["width_dbu"] / graph.dbu_per_micron
        for macro in graph.macros
    ]
    out[:, 3] = [
        macro["height_dbu"] / graph.dbu_per_micron
        for macro in graph.macros
    ]
    return out


def shelf_pack_fixed(
    boxes: np.ndarray,
    origin: tuple[float, float],
    outline: tuple[float, float],
    *,
    position_aware_ties: bool = False,
) -> np.ndarray | None:
    """Deterministic decreasing-height shelf fallback for fixed rectangles.

    Blocks are sorted by height and width, then placed in the first shelf that fits
    with the least remaining width. This avoids the severe row-height waste of the
    previous predicted-centre ordering. An optional predicted-position tie-break
    keeps otherwise equal-sized proposals from collapsing to one layout.
    """

    incoming = np.asarray(boxes, dtype=np.float64)
    if position_aware_ties:
        order = sorted(
            range(len(incoming)),
            key=lambda index: (
                -incoming[index, 3],
                -incoming[index, 2],
                incoming[index, 1],
                incoming[index, 0],
                index,
            ),
        )
    else:
        order = sorted(
            range(len(incoming)),
            key=lambda index: (
                -incoming[index, 3],
                -incoming[index, 2],
                index,
            ),
        )
    out = incoming.copy()
    shelves: list[dict[str, float]] = []
    used_height = 0.0
    for index in order:
        width, height = incoming[index, 2:]
        if width > outline[0] + 1.0e-9 or height > outline[1] + 1.0e-9:
            return None
        fitting = [
            shelf_index
            for shelf_index, shelf in enumerate(shelves)
            if width <= shelf["remaining"] + 1.0e-9
            and height <= shelf["height"] + 1.0e-9
        ]
        if fitting:
            shelf_index = min(
                fitting,
                key=lambda candidate: shelves[candidate]["remaining"] - width,
            )
            shelf = shelves[shelf_index]
            out[index, 0] = shelf["cursor_x"]
            out[index, 1] = shelf["y"]
            shelf["cursor_x"] += width
            shelf["remaining"] -= width
            continue
        if used_height + height > outline[1] + 1.0e-9:
            return None
        shelves.append(
            {
                "y": float(origin[1]) + used_height,
                "cursor_x": float(origin[0]) + width,
                "remaining": float(outline[0]) - width,
                "height": height,
            }
        )
        out[index, 0] = float(origin[0])
        out[index, 1] = float(origin[1]) + used_height
        used_height += height
    return out


def placement_connectivity(
    boxes: np.ndarray,
    graph: ExternalMacroGraph,
) -> dict[str, float]:
    """Weighted macro HPWL for the exported b2b and port hyperedges."""

    edges = graph_edges(graph)
    centers = np.asarray(boxes)[:, :2] + np.asarray(boxes)[:, 2:] * 0.5
    b2b_wirelength = 0.0
    for (left, right), weight in edges["b2b"].items():
        delta = np.abs(centers[left] - centers[right])
        b2b_wirelength += float(weight) * float(delta.sum())
    p2b_wirelength = 0.0
    for (port, block), weight in edges["p2b"].items():
        if port not in graph.ports:
            continue
        delta = np.abs(centers[block] - np.asarray(graph.ports[port]))
        p2b_wirelength += float(weight) * float(delta.sum())
    return {
        "b2b_hpwl_um": b2b_wirelength,
        "p2b_hpwl_um": p2b_wirelength,
    }


def export_macro_tcl(
    graph: ExternalMacroGraph,
    boxes: np.ndarray,
    *,
    preserve_reference_orientation: bool = False,
) -> str:
    """Render deterministic ORFS ``place_macro`` commands in microns."""

    lines = []
    for macro, box in zip(graph.macros, boxes):
        orientation = (
            macro.get("orientation", "R0")
            if preserve_reference_orientation
            else "R0"
        )
        lines.extend(
            [
                "place_macro \\",
                f"  -macro_name {{{macro['name']}}} \\",
                (
                    f"  -location {{{box[0]:.6f} {box[1]:.6f}}} "
                    f"-orientation {orientation}"
                ),
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "ExternalMacroGraph",
    "deflate_boxes",
    "export_macro_tcl",
    "floorplan_instance",
    "graph_edges",
    "guard_small_graph_pe",
    "infer_hierarchy_clusters",
    "inflate_boxes",
    "model_batch",
    "parse_orfs_graph",
    "physical_boxes",
    "placement_connectivity",
    "restore_fixed_dimensions",
    "shelf_pack_fixed",
]
