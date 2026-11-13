import copy
import json

import numpy as np
import pytest
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
from floorflow.representation.cad08_history_archive import make_archive, write_archive
from floorflow.representation.cad08_history_dxf_readback import verify_history_dxf_readback
from floorflow.representation.cad08_history_receipt import write_history_receipt
from floorflow.representation.cad08_transition_history import make_transition

ezdxf = pytest.importorskip("ezdxf")


def _write_geometry(model, geometry):
    kind = geometry.geom_type
    if geometry.is_empty:
        return {"type": kind, "empty": True}
    if kind == "Polygon":
        entity = model.add_hatch()
        entity.paths.add_polyline_path(list(geometry.exterior.coords)[:-1], is_closed=True, flags=1)
        for ring in geometry.interiors:
            entity.paths.add_polyline_path(list(ring.coords)[:-1], is_closed=True, flags=0)
        return {"type": kind, "handle": entity.dxf.handle}
    if kind in ("MultiPolygon", "GeometryCollection", "MultiLineString", "MultiPoint"):
        return {"type": kind, "children": [_write_geometry(model, child) for child in geometry.geoms]}
    if kind in ("LineString", "LinearRing"):
        entity = model.add_lwpolyline(list(geometry.coords), format="xy", close=kind == "LinearRing")
        return {"type": kind, "handle": entity.dxf.handle}
    entity = model.add_point((geometry.x, geometry.y))
    return {"type": kind, "handle": entity.dxf.handle}


def _export(document, dxf_path, metadata_path):
    dxf = ezdxf.new("R2010")
    model = dxf.modelspace()
    metadata = copy.deepcopy(document)
    manifest = {"footprint": _write_geometry(model, shape(metadata.pop("footprint"))), "objects": [], "hosts": []}
    for collection, key in (("objects", "objects"), ("wall_hosts", "hosts")):
        for item in metadata[collection]:
            manifest[key].append(_write_geometry(model, shape(item.pop("geometry"))))
    dxf.saveas(dxf_path)
    metadata_path.write_text(json.dumps({"metadata": metadata, "geometry_manifest": manifest}, sort_keys=True))


def test_real_dxf_readback_separates_file_history_geometry_and_metadata(tmp_path):
    footprint = Polygon([(0, 0), (12, 0), (12, 12), (0, 12)], holes=[[(4, 4), (4, 6), (6, 6), (6, 4)]])
    geometries = [
        footprint, MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]),
        GeometryCollection([Point(3, 3), LineString([(2, 3), (3, 4)])]),
        MultiLineString([[(1, 1), (2, 1)], [(3, 1), (4, 1)]]), MultiPoint([(1, 2), (2, 2)]),
        Point(), LineString([(7, 1), (8, 1)]), Point(9, 1), LinearRing([(7, 2), (8, 2), (8, 3)]),
    ]
    reference = {
        "footprint": mapping(footprint), "objects": [
            {"id": f"o{i}", "family": "rooms", "geometry": mapping(geometry)}
            for i, geometry in enumerate(geometries)
        ],
        "wall_hosts": [{"id": "h1", "geometry": mapping(LineString([(1, 0), (5, 0)]))}],
        "lineage": [], "provenance": {"source": "synthetic"},
    }
    reference = json.loads(json.dumps(reference))
    native = {"program": np.array([True, False])}
    snapshot = {"before_cad": reference, "after_cad": reference, "before_native": native, "after_native": native}
    record = make_transition(branch="a", sequence=0, previous_head="anchor", kind="no_emission",
                             operation="room_type", trace={"accepted": False}, **snapshot)
    history = tmp_path / "history.json"
    write_archive(history, make_archive("a", [record]))
    dxf, metadata, receipt = tmp_path / "plan.dxf", tmp_path / "metadata.json", tmp_path / "receipt.json"
    _export(reference, dxf, metadata)
    paths = {"dxf": dxf, "metadata": metadata, "history": history}
    receipt_hash = write_history_receipt(receipt, **paths, before_export_cad_content_sha256=content_digest(reference),
                                         branch="a", head=record["record_hash"], stage="after")
    args = dict(**paths, trusted_receipt_sha256=receipt_hash, branch="a", trusted_previous_head="anchor",
                trusted_head=record["record_hash"], snapshots=[snapshot], reference_cad=reference)
    result = verify_history_dxf_readback(receipt, **args)
    assert result["receipt"]["file_integrity"] == result["receipt"]["history_integrity"] == "pass"
    assert result["reference_content"] == result["readback"] == result["geometry"] == result["metadata"] == "pass"
    assert result["permissions"] == "unknown" and result["original_epgj"] == "not_evaluated"
    assert verify_history_dxf_readback(receipt, **{**args, "reference_cad": None})["geometry"] == "unknown"
    missing_history = verify_history_dxf_readback(receipt, **{**args, "snapshots": [None]})
    assert missing_history["receipt"]["history_integrity"] == "unknown" and missing_history["geometry"] == "pass"

    changed = copy.deepcopy(reference)
    changed["objects"][0]["geometry"] = mapping(Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]))
    changed_dxf, changed_meta = tmp_path / "changed.dxf", tmp_path / "changed.json"
    _export(changed, changed_dxf, changed_meta)
    changed_receipt = tmp_path / "changed-receipt.json"
    changed_paths = {"dxf": changed_dxf, "metadata": changed_meta, "history": history}
    changed_hash = write_history_receipt(changed_receipt, **changed_paths,
                                         before_export_cad_content_sha256=content_digest(reference),
                                         branch="a", head=record["record_hash"], stage="after")
    changed_args = {**args, **changed_paths, "trusted_receipt_sha256": changed_hash}
    assert verify_history_dxf_readback(changed_receipt, **changed_args)["geometry"] == "fail"
    changed_metadata = copy.deepcopy(reference)
    changed_metadata["objects"][0]["id"] = "different-id"
    metadata_dxf, metadata_file = tmp_path / "metadata-changed.dxf", tmp_path / "metadata-changed.json"
    _export(changed_metadata, metadata_dxf, metadata_file)
    metadata_receipt = tmp_path / "metadata-receipt.json"
    metadata_paths = {"dxf": metadata_dxf, "metadata": metadata_file, "history": history}
    metadata_hash = write_history_receipt(metadata_receipt, **metadata_paths,
                                          before_export_cad_content_sha256=content_digest(reference),
                                          branch="a", head=record["record_hash"], stage="after")
    metadata_args = {**args, **metadata_paths, "trusted_receipt_sha256": metadata_hash}
    assert verify_history_dxf_readback(metadata_receipt, **metadata_args)["metadata"] == "fail"
    wrong_reference = copy.deepcopy(reference)
    wrong_reference["provenance"]["source"] = "other"
    wrong = verify_history_dxf_readback(receipt, **{**args, "reference_cad": wrong_reference})
    assert wrong["reference_content"] == "fail" and wrong["geometry"] == "unknown"
    invalid_dxf, invalid_receipt = tmp_path / "not-a-dxf", tmp_path / "invalid-receipt.json"
    invalid_dxf.write_bytes(b"not a DXF")
    invalid_paths = {"dxf": invalid_dxf, "metadata": metadata, "history": history}
    invalid_hash = write_history_receipt(invalid_receipt, **invalid_paths,
                                         before_export_cad_content_sha256=content_digest(reference),
                                         branch="a", head=record["record_hash"], stage="after")
    invalid = verify_history_dxf_readback(invalid_receipt, **{**args, **invalid_paths,
                                                               "trusted_receipt_sha256": invalid_hash})
    assert invalid["receipt"]["file_integrity"] == "pass" and invalid["readback"] == "fail"
    elevated = ezdxf.readfile(dxf)
    point = next(entity for entity in elevated.modelspace() if entity.dxftype() == "POINT")
    point.dxf.location = (point.dxf.location.x, point.dxf.location.y, 1.0)
    elevated_path, elevated_receipt = tmp_path / "elevated.dxf", tmp_path / "elevated-receipt.json"
    elevated.saveas(elevated_path)
    elevated_paths = {"dxf": elevated_path, "metadata": metadata, "history": history}
    elevated_hash = write_history_receipt(elevated_receipt, **elevated_paths,
                                          before_export_cad_content_sha256=content_digest(reference),
                                          branch="a", head=record["record_hash"], stage="after")
    elevated_result = verify_history_dxf_readback(elevated_receipt, **{**args, **elevated_paths,
                                                                     "trusted_receipt_sha256": elevated_hash})
    assert elevated_result["receipt"]["file_integrity"] == "pass" and elevated_result["readback"] == "fail"
    metadata.write_text("{}")
    assert verify_history_dxf_readback(receipt, **args)["receipt"]["file_integrity"] == "fail"
