"""Independent CAD08 runtime-state acceptance predicates; no neural execution.

These predicates preserve the old geometry metrics. IDs are system runtime IDs;
geometric support of serialized references is not a model-predicted owner ID.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

EPS = 1e-9
TABLES = ("opening_host", "fixture_room", "room_adjacency", "wall_room_incidence")
FAMILIES = ("rooms", "openings", "fixtures", "wall_material")
TOP_FIELDS = {
    "schema",
    "frame",
    "namespace",
    "footprint",
    "objects",
    "wall_hosts",
    "relationships",
    "lineage",
    "visible_reference",
    "diagnostic_rasters",
    "provenance",
}


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tri_and(*values: str) -> str:
    return "fail" if "fail" in values else "unknown" if "unknown" in values else "pass"


def mask_geometry(mask: np.ndarray):
    runs = []
    for y, line in enumerate(mask):
        changes = np.diff(np.pad(line.astype(np.int8), (1, 1)))
        runs.extend(
            box(int(a), y, int(b), y + 1)
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)
        )
    return unary_union(runs)


def decode_rle(record: dict) -> np.ndarray:
    result = np.repeat([r[0] for r in record["runs"]], [r[1] for r in record["runs"]])
    return result.astype(record["dtype"]).reshape(record["shape"])


class StateView:
    """Minimal data-only interface used by the exact frozen legacy G function."""

    def __init__(self, document: dict):
        self.document = document

    def objects(self, family: str) -> list:
        return [v for v in self.document["objects"] if v["family"] == family]


def state_key(doc: dict, predecessor: dict | None = None) -> str:
    fields = ("schema", "frame", "namespace", "footprint", "objects", "wall_hosts", "relationships", "lineage")
    payload = {k: doc.get(k) for k in fields}
    unchecked = set(doc) - TOP_FIELDS
    if unchecked:
        payload["unvalidated_top_fields"] = {k: doc[k] for k in sorted(unchecked)}
    return digest(payload) + (
        digest(sorted(v["id"] for v in predecessor["objects"])) if predecessor and doc.get("lineage") else ""
    )


def _issue(status: str, code: str, source: str, target=None, **evidence) -> dict:
    return {
        "status": status,
        "code": code,
        "source": source,
        "target": target,
        "key": canonical([status, code, source, target]),
        "evidence": evidence,
    }





from floorflow.evaluation.cad08_state_preservation_v1 import preservation_state
from floorflow.evaluation.cad08_state_transition_v1 import transition_audit
from floorflow.evaluation.cad08_state_validity_v1 import reference_validity, validity_state

__all__ = [
    "EPS", "FAMILIES", "TABLES", "TOP_FIELDS", "StateView", "canonical",
    "decode_rle", "digest", "mask_geometry", "preservation_state",
    "reference_validity", "state_key", "transition_audit", "tri_and",
    "validity_state",
]
