"""Opt-in file-bound R2010 CAD readback; topology is not edit safety."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from shapely.errors import GEOSException
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)

from floorflow.representation.cad08_content_digest import content_digest
from floorflow.representation.cad08_history_receipt import verify_history_receipt

_EMPTY = {
    "Polygon": Polygon,
    "MultiPolygon": MultiPolygon,
    "GeometryCollection": GeometryCollection,
    "MultiLineString": MultiLineString,
    "MultiPoint": MultiPoint,
    "LineString": LineString,
    "LinearRing": LinearRing,
    "Point": Point,
}
_MULTI = {
    "MultiPolygon": MultiPolygon,
    "GeometryCollection": GeometryCollection,
    "MultiLineString": MultiLineString,
    "MultiPoint": MultiPoint,
}


def _geometry(document: Any, tree: dict[str, Any], used: set[str]) -> Any:
    kind = tree["type"]
    if kind not in _EMPTY:
        raise ValueError(f"Unsupported geometry type: {kind}")
    if tree.get("empty") is True:
        if set(tree) != {"type", "empty"}:
            raise ValueError("Empty geometry has unexpected manifest fields")
        return _EMPTY[kind]()
    if kind in _MULTI:
        if set(tree) != {"type", "children"}:
            raise ValueError("Geometry collection has unexpected manifest fields")
        children = tree["children"]
        if not isinstance(children, list):
            raise ValueError("Geometry children must be a list")
        return _MULTI[kind]([_geometry(document, child, used) for child in children])
    if set(tree) != {"type", "handle"}:
        raise ValueError("Geometry entity has unexpected manifest fields")
    handle = tree["handle"]
    if not isinstance(handle, str) or handle in used:
        raise ValueError("DXF handle is missing or repeated")
    used.add(handle)
    entity = document.entitydb.get(handle)
    expected = "HATCH" if kind == "Polygon" else "POINT" if kind == "Point" else "LWPOLYLINE"
    if entity is None or entity.dxftype() != expected:
        raise ValueError(f"Missing or wrong DXF entity for {kind}")
    thickness = entity.dxf.get("thickness", 0) if entity.dxf.is_supported("thickness") else 0
    if tuple(entity.dxf.get("extrusion", (0, 0, 1))) != (0, 0, 1) or thickness != 0:
        raise ValueError("Only flat XY entities are supported")
    if kind == "Polygon":
        if tuple(entity.dxf.elevation) != (0, 0, 0):
            raise ValueError("Only zero-elevation hatches are supported")
        paths = list(entity.paths)
        if not paths:
            raise ValueError("Polygon hatch has no rings")
        rings = []
        for index, path in enumerate(paths):
            if not hasattr(path, "vertices") or not path.is_closed:
                raise ValueError("Hatch must use closed polyline rings")
            if bool(path.path_type_flags & 1) != (index == 0):
                raise ValueError("Hatch exterior/interior ring flags do not match manifest")
            if any(len(vertex) > 2 and float(vertex[2]) != 0.0 for vertex in path.vertices):
                raise ValueError("Curved hatch boundaries are not represented by this format")
            rings.append([(float(vertex[0]), float(vertex[1])) for vertex in path.vertices])
        return Polygon(rings[0], rings[1:])
    if kind in ("LineString", "LinearRing"):
        if entity.dxf.get("elevation", 0) != 0:
            raise ValueError("Only zero-elevation polylines are supported")
        if bool(entity.closed) != (kind == "LinearRing"):
            raise ValueError("Polyline closed flag disagrees with manifest")
        points = list(entity.get_points("xyb"))
        if any(float(point[2]) != 0.0 for point in points):
            raise ValueError("Curved polylines are not represented by this format")
        coordinates = [(float(point[0]), float(point[1])) for point in points]
        return LinearRing(coordinates) if kind == "LinearRing" else LineString(coordinates)
    if entity.dxf.location.z != 0:
        raise ValueError("Only zero-elevation points are supported")
    return Point(float(entity.dxf.location.x), float(entity.dxf.location.y))


def _read_document(dxf_bytes: bytes, sidecar_bytes: bytes) -> dict[str, Any]:
    # The optional DXF dependency is loaded only on the opt-in readback path.
    import ezdxf

    document = ezdxf.read(io.StringIO(dxf_bytes.decode("utf-8")))
    if document.dxfversion != "AC1024":
        raise ValueError("Expected the historical R2010 DXF format")
    payload = json.loads(sidecar_bytes)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "geometry_manifest"}:
        raise ValueError("Expected the historical metadata/geometry manifest sidecar")
    metadata, manifest = copy.deepcopy(payload["metadata"]), payload["geometry_manifest"]
    if not isinstance(metadata, dict) or not isinstance(manifest, dict):
        raise TypeError("Invalid metadata or geometry manifest")
    if "footprint" in metadata or set(manifest) != {"footprint", "objects", "hosts"}:
        raise ValueError("Sidecar contains an unexpected geometry field")
    used: set[str] = set()
    metadata["footprint"] = mapping(_geometry(document, manifest["footprint"], used))
    for collection, key in (("objects", "objects"), ("wall_hosts", "hosts")):
        items, trees = metadata[collection], manifest[key]
        if not isinstance(items, list) or not isinstance(trees, list) or len(items) != len(trees):
            raise ValueError(f"{collection} and geometry manifest lengths differ")
        for item, tree in zip(items, trees, strict=True):
            if not isinstance(item, dict) or "geometry" in item:
                raise ValueError(f"Invalid {collection} metadata")
            item["geometry"] = mapping(_geometry(document, tree, used))
    if {entity.dxf.handle for entity in document.modelspace()} != used:
        raise ValueError("DXF modelspace entities and geometry manifest differ")
    return metadata


def _metadata_without_geometry(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result.pop("footprint")
    for collection in ("objects", "wall_hosts"):
        for item in result[collection]:
            item.pop("geometry")
    # The archive is JSON-based; normalize tuples introduced by DXF geometry.
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def _geometry_comparison(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    differences = []
    if not shape(reference["footprint"]).equals(shape(actual["footprint"])):
        differences.append("footprint")
    for collection in ("objects", "wall_hosts"):
        before, after = reference[collection], actual[collection]
        if len(before) != len(after):
            differences.append(collection + ":length")
            continue
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            if not shape(left["geometry"]).equals(shape(right["geometry"])):
                differences.append(f"{collection}:{index}")
    return {"status": "fail" if differences else "pass", "differences": differences}


def verify_history_dxf_readback(
    receipt_path: Path, *, dxf: Path, metadata: Path, history: Path,
    trusted_receipt_sha256: str | None, branch: str, trusted_previous_head: str | None,
    trusted_head: str | None, snapshots: list[dict[str, Any] | None],
    action_evidence: list[dict[str, Any] | None] | None = None,
    reference_cad: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check receipt then compare R2010 readback to the supplied source document.

    File, history, topology and metadata statuses are separate. This does not
    assess authorship, permissions, edit success or CAD safety.
    """
    receipt_check = verify_history_receipt(
        receipt_path, dxf=dxf, metadata=metadata, history=history,
        trusted_receipt_sha256=trusted_receipt_sha256, branch=branch,
        trusted_previous_head=trusted_previous_head, trusted_head=trusted_head,
        snapshots=snapshots, action_evidence=action_evidence,
    )
    result: dict[str, Any] = {
        "receipt": receipt_check, "readback": "not_evaluated", "reference_content": "unknown",
        "geometry": "unknown", "metadata": "unknown", "permissions": "unknown",
        "cross_rebuild_identity": "unknown", "original_epgj": "not_evaluated",
    }
    if receipt_check["file_integrity"] != "pass":
        return result
    # Use the same bytes whose digests are authenticated by the caller's receipt.
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != trusted_receipt_sha256:
        result["readback"] = "fail"
        result["reason"] = "receipt_changed_during_readback"
        return result
    receipt = json.loads(receipt_bytes)
    dxf_bytes, metadata_bytes = dxf.read_bytes(), metadata.read_bytes()
    if any(hashlib.sha256(data).hexdigest() != receipt["files"][key]
           for key, data in (("dxf", dxf_bytes), ("metadata", metadata_bytes))):
        result["readback"] = "fail"
        result["reason"] = "artifact_changed_during_readback"
        return result
    try:
        import ezdxf
    except ModuleNotFoundError as exc:
        if exc.name != "ezdxf":
            raise
        result["readback"] = "unknown"
        result["reason"] = "optional_ezdxf_unavailable"
        return result
    try:
        actual = _read_document(dxf_bytes, metadata_bytes)
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, GEOSException, ezdxf.DXFError) as exc:
        result["readback"] = "fail"
        result["reason"] = f"readback_error:{type(exc).__name__}"
        return result
    result["readback"] = "pass"
    result["readback_document"] = actual
    if reference_cad is None:
        return result
    if content_digest(reference_cad) != receipt.get("before_export_cad_content_sha256"):
        result["reference_content"] = "fail"
        return result
    result["reference_content"] = "pass"
    try:
        result["geometry_detail"] = _geometry_comparison(reference_cad, actual)
        result["geometry"] = result["geometry_detail"]["status"]
        result["metadata"] = "pass" if _metadata_without_geometry(reference_cad) == _metadata_without_geometry(actual) else "fail"
    except (KeyError, TypeError, ValueError, GEOSException) as exc:
        result["geometry"] = result["metadata"] = "fail"
        result["reason"] = f"comparison_error:{type(exc).__name__}"
    return result
