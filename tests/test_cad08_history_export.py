import copy
import json

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, mapping

pytest.importorskip("ezdxf")

from floorflow.representation.cad08_content_digest import content_digest
from floorflow.representation.cad08_history_archive import make_archive, write_archive
from floorflow.representation.cad08_history_dxf_readback import verify_history_dxf_readback
from floorflow.representation.cad08_history_export import export_history_package
from floorflow.representation.cad08_transition_history import make_transition


def _history(tmp_path, before, after, event):
    native = {"program": np.array([True, False]), "valid_input": np.array([1.0], dtype=np.float16)}
    snapshot = {"before_cad": before, "after_cad": after, "before_native": native, "after_native": native}
    trace = {"operation": "room_type", "accepted": True,
             "selected_before_ids": ["room"], "selected_after_ids": ["room"]}
    action = {"operation": "room_type", "target_id": "room", "label": 2}
    action_sha = content_digest(action)
    record = make_transition(branch="direct", sequence=0, previous_head="previous-anchor", kind="event",
                             operation="room_type", event=event, trace=trace, action_content_sha256=action_sha,
                             **snapshot)
    path = tmp_path / "source-history.json"
    file_hash = write_archive(path, make_archive("direct", [record]))
    return path, file_hash, record["record_hash"], snapshot, [{"action": action, "trusted_sha256": action_sha}]


def test_export_event_archive_to_real_dxf_and_independently_read_back(tmp_path):
    footprint = Polygon([(0, 0), (12, 0), (12, 12), (0, 12)], holes=[[(4, 4), (4, 6), (6, 6), (6, 4)]])
    event = {"operation": "room_type", "parents": ["room"], "children": ["room"]}
    before = json.loads(json.dumps({
        "footprint": mapping(footprint),
        "objects": [{"id": "room", "family": "rooms", "label": 1,
                     "geometry": mapping(MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]))}],
        "wall_hosts": [{"id": "host", "geometry": mapping(GeometryCollection([Point(3, 3),
                                                                                  LineString([(2, 3), (3, 4)])]))}],
        "lineage": [], "provenance": {"source": "synthetic"},
    }))
    after = copy.deepcopy(before)
    after["objects"][0]["label"] = 2
    after["lineage"].append(event)
    history, history_sha, head, snapshot, action_evidence = _history(tmp_path, before, after, event)
    args = {"cad": after, "history_path": history, "trusted_history_sha256": history_sha, "branch": "direct",
            "trusted_previous_head": "previous-anchor", "trusted_head": head, "snapshots": [snapshot],
            "stage": "after-event", "action_evidence": action_evidence}

    missing = export_history_package(tmp_path / "missing", **{**args, "snapshots": [None]})
    assert missing["status"] == "unknown" and not (tmp_path / "missing").exists()
    untrusted = export_history_package(tmp_path / "untrusted", **{**args, "trusted_history_sha256": None})
    assert untrusted["status"] == "unknown" and not (tmp_path / "untrusted").exists()
    no_action = export_history_package(tmp_path / "no-action", **{**args, "action_evidence": None})
    assert no_action["status"] == "unknown" and not (tmp_path / "no-action").exists()
    out = tmp_path / "package"
    result = export_history_package(out, **args)
    assert result["status"] == "exported_unvalidated"
    assert result["history"]["link_integrity"] == "pass" and result["history"]["status"] == "unknown"
    assert result["permissions"] == "unknown" and result["original_epgj"] == "not_evaluated"
    checked = verify_history_dxf_readback(
        out / "receipt.json", dxf=out / "plan.dxf", metadata=out / "metadata.json", history=out / "history.json",
        trusted_receipt_sha256=result["receipt_sha256"], branch="direct", trusted_previous_head="previous-anchor",
        trusted_head=head, snapshots=[snapshot], action_evidence=action_evidence, reference_cad=after,
    )
    assert checked["receipt"]["file_integrity"] == checked["receipt"]["history_integrity"] == "pass"
    assert checked["reference_content"] == checked["geometry"] == checked["metadata"] == "pass"
    assert checked["permissions"] == "unknown" and checked["original_epgj"] == "not_evaluated"
    assert history.read_bytes() == (out / "history.json").read_bytes()
    assert export_history_package(out, **args)["reason"] == "output_directory_exists"


def test_export_refuses_to_flatten_3d_geometry(tmp_path):
    event = {"operation": "room_type", "parents": ["room"], "children": ["room"]}
    before = json.loads(json.dumps({"footprint": mapping(Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])),
                                    "objects": [{"id": "room", "geometry": mapping(Point(1, 1, 2))}],
                                    "wall_hosts": [], "lineage": []}))
    after = copy.deepcopy(before)
    after["lineage"].append(event)
    history, history_sha, head, snapshot, action_evidence = _history(tmp_path, before, after, event)
    output = tmp_path / "unsupported"
    result = export_history_package(output, cad=after, history_path=history, trusted_history_sha256=history_sha,
                                    branch="direct", trusted_previous_head="previous-anchor", trusted_head=head,
                                    snapshots=[snapshot], action_evidence=action_evidence, stage="after")
    assert result["status"] == "fail" and result["reason"].startswith("unexportable_cad:")
    assert not output.exists()
