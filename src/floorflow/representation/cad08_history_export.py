"""Opt-in R2010 package export bound to separately anchored history evidence.

This does not establish permissions, semantic identity, or CAD edit safety.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from shapely import get_coordinates
from shapely.errors import GEOSException
from shapely.geometry import shape

from floorflow.representation.cad08_content_digest import content_digest
from floorflow.representation.cad08_history_archive import read_archive, verify_archive
from floorflow.representation.cad08_history_receipt import write_history_receipt


def _emit_geometry(model: Any, geometry: Any) -> dict[str, Any]:
    kind = geometry.geom_type
    supported = {"Polygon", "MultiPolygon", "GeometryCollection", "MultiLineString", "MultiPoint",
                 "LineString", "LinearRing", "Point"}
    if kind not in supported:
        raise ValueError(f"Unsupported geometry type: {kind}")
    if geometry.has_z or getattr(geometry, "has_m", False):
        raise ValueError("Only explicitly 2D geometry is supported")
    if geometry.is_empty:
        return {"type": kind, "empty": True}
    if not np.isfinite(get_coordinates(geometry)).all():
        raise ValueError("Geometry coordinates must be finite")
    if kind == "Polygon":
        entity = model.add_hatch(color=7)
        entity.paths.add_polyline_path(list(geometry.exterior.coords)[:-1], is_closed=True, flags=1)
        for ring in geometry.interiors:
            entity.paths.add_polyline_path(list(ring.coords)[:-1], is_closed=True, flags=0)
        return {"type": kind, "handle": entity.dxf.handle}
    if kind in {"MultiPolygon", "GeometryCollection", "MultiLineString", "MultiPoint"}:
        return {"type": kind, "children": [_emit_geometry(model, child) for child in geometry.geoms]}
    if kind in {"LineString", "LinearRing"}:
        entity = model.add_lwpolyline(list(geometry.coords), format="xy", close=kind == "LinearRing")
        return {"type": kind, "handle": entity.dxf.handle}
    entity = model.add_point((geometry.x, geometry.y))
    return {"type": kind, "handle": entity.dxf.handle}


def export_history_package(
    output_dir: Path, *, cad: dict[str, Any], history_path: Path,
    trusted_history_sha256: str | None, branch: str, trusted_previous_head: str | None,
    trusted_head: str | None, snapshots: list[dict[str, Any] | None],
    action_evidence: list[dict[str, Any] | None] | None = None, stage: str,
) -> dict[str, Any]:
    """Produce a new file package only from supplied, locally linked evidence.

    Caller-held hashes are not authorship or safety proofs. An ``exported``
    status means files were created, not that history, semantics or scores pass.
    """
    if trusted_history_sha256 is None:
        return {"status": "unknown", "reason": "trusted_history_hash_missing"}
    try:
        archive, file_check = read_archive(history_path, trusted_file_sha256=trusted_history_sha256)
    except FileNotFoundError:
        return {"status": "unknown", "reason": "history_file_missing"}
    if file_check["file_integrity"] != "pass" or archive is None:
        return {"status": file_check["file_integrity"], "reason": "history_file_not_verified"}
    try:
        result = verify_archive(archive, snapshots, branch=branch, trusted_previous_head=trusted_previous_head,
                                trusted_head=trusted_head, action_evidence=action_evidence)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "fail", "reason": f"history_unrepresentable:{type(exc).__name__}"}
    if result["link_integrity"] != "pass":
        return {"status": result["link_integrity"], "reason": "history_links_not_verified", "history": result}
    if any(record.get("kind") in {"event", "no_emission"} and not record.get("action_content_sha256")
           for record in archive["records"]):
        return {"status": "unknown", "reason": "action_evidence_not_bound_to_history", "history": result}
    source_cad = copy.deepcopy(cad)
    cad_hash = content_digest(source_cad)
    if archive["records"][-1]["after_cad"] != cad_hash:
        return {"status": "fail", "reason": "terminal_cad_content_mismatch", "history": result}
    history_bytes = history_path.read_bytes()
    if hashlib.sha256(history_bytes).hexdigest() != trusted_history_sha256:
        return {"status": "fail", "reason": "history_changed_during_export", "history": result}
    try:
        import ezdxf
    except ModuleNotFoundError as exc:
        if exc.name != "ezdxf":
            raise
        return {"status": "unknown", "reason": "optional_ezdxf_unavailable", "history": result}
    try:
        metadata = source_cad
        document = ezdxf.new("R2010")
        model = document.modelspace()
        manifest = {"footprint": _emit_geometry(model, shape(metadata.pop("footprint"))), "objects": [], "hosts": []}
        for collection, key in (("objects", "objects"), ("wall_hosts", "hosts")):
            items = metadata[collection]
            if not isinstance(items, list):
                raise TypeError(f"{collection} must be a list")
            for item in items:
                manifest[key].append(_emit_geometry(model, shape(item.pop("geometry"))))
        text = io.StringIO()
        document.write(text)
        dxf_bytes = text.getvalue().encode("utf-8")
        metadata_bytes = (json.dumps({"metadata": metadata, "geometry_manifest": manifest},
                                     sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (KeyError, TypeError, ValueError, GEOSException) as exc:
        return {"status": "fail", "reason": f"unexportable_cad:{type(exc).__name__}", "history": result}
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        return {"status": "fail", "reason": "output_directory_exists", "history": result}
    except OSError as exc:
        return {"status": "fail", "reason": f"output_directory_failed:{type(exc).__name__}", "history": result}
    dxf_path, metadata_path, history_copy = (output_dir / "plan.dxf", output_dir / "metadata.json",
                                             output_dir / "history.json")
    try:
        dxf_path.write_bytes(dxf_bytes)
        metadata_path.write_bytes(metadata_bytes)
        history_copy.write_bytes(history_bytes)
        receipt_hash = write_history_receipt(output_dir / "receipt.json", dxf=dxf_path, metadata=metadata_path,
                                             history=history_copy, before_export_cad_content_sha256=cad_hash,
                                             branch=branch, head=trusted_head, stage=stage)
    except (OSError, TypeError, ValueError) as exc:
        return {"status": "fail", "reason": f"package_write_failed:{type(exc).__name__}", "history": result,
                "partial_output_directory": str(output_dir)}
    return {"status": "exported_unvalidated", "receipt_sha256": receipt_hash, "history": result,
            "permissions": "unknown", "semantic_identity": "unknown", "original_epgj": "not_evaluated"}
