"""Versioned, hole-aware CAD state for saved independent O decoder planes.

IDs in this adapter are system-created runtime identities, never network or
author-created room IDs. Initialization is a readout; subsequent transactions
inherit records and record split/merge lineage explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy import ndimage
from shapely import contains_xy, normalize
from shapely.geometry import LineString, mapping, shape
from shapely.ops import unary_union

from floorflow.geometry.cell_polygons import component_cell_polygon

FAMILIES = {"rooms": "room_labels", "openings": "opening", "fixtures": "furniture"}
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
EPS = 1e-9


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def mask_geometry(mask: np.ndarray):
    """Use the existing exact cell-union primitive without filling small regions."""
    components, count = ndimage.label(mask, structure=FOUR)
    return unary_union([component_cell_polygon(components == i) for i in range(1, count + 1)])


def encoded_geometry(geometry) -> dict:
    return json.loads(json.dumps(mapping(normalize(geometry))))


def raster_geometry(geometry, frame: tuple[int, int]) -> np.ndarray:
    result = np.zeros(frame, bool)
    if geometry.is_empty:
        return result
    x0, y0, x1, y1 = geometry.bounds
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(frame[1], int(np.ceil(x1))), min(frame[0], int(np.ceil(y1)))
    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        result[y0:y1, x0:x1] = contains_xy(geometry, xx + 0.5, yy + 0.5)
    return result


def rle_encode(array: np.ndarray) -> dict:
    flat = array.ravel()
    edges = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "runs": [[int(flat[a]), int(b - a)] for a, b in pairwise(edges)],
    }


def rle_decode(value: dict) -> np.ndarray:
    return (
        np.repeat([v[0] for v in value["runs"]], [v[1] for v in value["runs"]])
        .astype(value["dtype"])
        .reshape(value["shape"])
    )


def object_record(namespace: str, family: str, label: int, geometry, *, identity: str | None = None) -> dict:
    encoded = encoded_geometry(geometry)
    token = fingerprint([namespace, family, int(label), encoded])[:20]
    return {
        "id": identity or f"runtime:{family}:{token}",
        "family": family,
        "label": int(label),
        "geometry": encoded,
        "identity_source": "system_created_at_initialization_not_authored_or_model_owner",
    }


@dataclass
class LayeredCad08State:
    """Serializable geometry and independent planes, with derived relationships."""

    document: dict

    @classmethod
    def from_decoded(cls, decoded: dict, taxonomy, namespace: str, *, hosts: list[dict] | None = None):
        frame = tuple(decoded["footprint"].shape)
        inside = decoded["footprint"].astype(bool)
        objects = []
        for family, field in FAMILIES.items():
            allowed = getattr(
                taxonomy,
                "room_ids" if family == "rooms" else "opening_ids" if family == "openings" else "furniture_ids",
            )
            for label in allowed:
                components, count = ndimage.label(decoded[field] == label, structure=FOUR)
                for component in range(1, count + 1):
                    geometry = component_cell_polygon(components == component)
                    objects.append(object_record(namespace, family, int(label), geometry))
        wall = mask_geometry(decoded["wall"].astype(bool))
        if not wall.is_empty:
            objects.append(object_record(namespace, "wall_material", 1, wall))
        doc = {
            "schema": "floorflow-layered-cad08-v1",
            "frame": list(frame),
            "namespace": namespace,
            "footprint": encoded_geometry(mask_geometry(inside)),
            "objects": objects,
            "wall_hosts": copy.deepcopy(hosts or []),
            "room_aliases": {},
            "relationships": {},
            "lineage": [],
            "visible_reference": rle_encode(decoded["prediction"]),
            "diagnostic_rasters": {
                k: rle_encode(decoded[k])
                for k in ("room_support", "topology_wall", "topology_prediction")
                if k in decoded
            },
            "provenance": {
                "decoded_fields_unchanged": True,
                "minimum_export_component_area": 1,
                "small_components_not_merged_or_filled": True,
                "hidden_plane_overlap_preserved": True,
                "room_owner_predictions_available": False,
            },
        }
        state = cls(doc)
        state.refresh_relationships()
        return state

    def to_dict(self) -> dict:
        return copy.deepcopy(self.document)

    @classmethod
    def from_dict(cls, document: dict):
        if document.get("schema") != "floorflow-layered-cad08-v1":
            raise ValueError("unsupported layered CAD schema")
        return cls(copy.deepcopy(document))

    def objects(self, family: str) -> list[dict]:
        return [v for v in self.document["objects"] if v["family"] == family]

    def render(self) -> dict[str, np.ndarray]:
        frame = tuple(self.document["frame"])
        layers = {field: np.zeros(frame, np.uint8) for field in (*FAMILIES.values(), "wall")}
        for obj in self.document["objects"]:
            field = "wall" if obj["family"] == "wall_material" else FAMILIES[obj["family"]]
            layers[field][raster_geometry(shape(obj["geometry"]), frame)] = obj["label"]
        layers["footprint"] = raster_geometry(shape(self.document["footprint"]), frame).astype(np.uint8)
        labels = layers["room_labels"].copy()
        labels = np.where(layers["furniture"] != 0, layers["furniture"], labels)
        labels = np.where(layers["wall"] != 0, 1, labels)
        layers["prediction"] = np.where(layers["opening"] != 0, layers["opening"], labels) * layers["footprint"]
        return layers

    def refresh_relationships(self) -> None:
        rooms = [(v, shape(v["geometry"])) for v in self.objects("rooms")]
        hosts = [(h, shape(h["geometry"])) for h in self.document["wall_hosts"]]
        relations = {"fixture_room": {}, "opening_host": {}, "room_adjacency": [], "wall_room_incidence": []}
        for fixture in self.objects("fixtures"):
            g = shape(fixture["geometry"])
            candidates = [
                {"id": r["id"], "overlap_area": float(g.intersection(rg).area)}
                for r, rg in rooms
                if g.intersection(rg).area > EPS
            ]
            candidates.sort(key=lambda c: (-c["overlap_area"], c["id"]))
            best = candidates[0]["overlap_area"] if candidates else 0
            unique = bool(candidates) and sum(abs(c["overlap_area"] - best) <= EPS for c in candidates) == 1
            relations["fixture_room"][fixture["id"]] = {
                "status": "unique" if unique else "ambiguous" if candidates else "missing",
                "room_id": candidates[0]["id"] if unique else None,
                "candidates": candidates,
                "source": "before_or_current_geometry_diagnostic_not_model_assignment",
            }
        for opening in self.objects("openings"):
            g = shape(opening["geometry"])
            candidates = [
                {"host_id": h["id"], "covered_fraction": float(g.intersection(hg).area / g.area)}
                for h, hg in hosts
                if g.area and g.intersection(hg).area / g.area >= 0.5
            ]
            unique = len(candidates) == 1
            relations["opening_host"][opening["id"]] = {
                "status": "unique" if unique else "ambiguous" if candidates else "missing",
                "host_id": candidates[0]["host_id"] if unique else None,
                "candidates": candidates,
                "source": "fixed_before_segment_coverage_not_network_host_ID",
            }
        for i, (a, ag) in enumerate(rooms):
            for b, bg in rooms[i + 1 :]:
                if ag.boundary.intersection(bg.boundary).length > EPS:
                    relations["room_adjacency"].append([a["id"], b["id"]])
            for host, hg in hosts:
                if hg.intersection(ag).area > EPS:
                    relations["wall_room_incidence"].append([host["id"], a["id"]])
        self.document["relationships"] = relations


def host_rows(walls: list, footprint: np.ndarray, namespace: str) -> list[dict]:
    """Host witnesses from existing segment readout; material stays independent."""
    inside = mask_geometry(footprint)
    result = []
    for i, wall in enumerate(walls):
        if isinstance(wall, dict):
            start, end, thickness = wall["start"], wall["end"], float(wall["thickness"])
        else:
            start, end, thickness = wall.start, wall.end, float(wall.thickness)
        line = LineString([start, end])
        if line.length <= 0 or thickness <= 0:
            continue
        result.append(
            {
                "id": f"runtime:host:{fingerprint([namespace, i, start, end, thickness])[:20]}",
                "geometry": encoded_geometry(line.buffer(thickness / 2, cap_style=2).intersection(inside)),
                "start": list(start),
                "end": list(end),
                "thickness": thickness,
                "source": "existing_segment_witness_not_material_or_authored_wall",
            }
        )
    return result
