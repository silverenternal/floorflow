import numpy as np
import pytest

from floorflow.representation.cad08_content_digest import content_digest
from floorflow.representation.cad08_history_archive import make_archive, write_archive
from floorflow.representation.cad08_history_receipt import verify_history_receipt, write_history_receipt
from floorflow.representation.cad08_transition_history import make_transition


def test_receipt_checks_separate_files_and_anchored_history_without_claiming_geometry(tmp_path):
    before = {"objects": [{"id": "room-a", "label": 1}], "lineage": []}
    event = {"operation": "room_type", "parents": ["room-a"], "children": ["room-a"]}
    after = {"objects": [{"id": "room-a", "label": 2}], "lineage": [event]}
    native_before = {"program": np.array([True, False])}
    native_after = {"program": np.array([False, True])}
    snapshot = {"before_cad": before, "after_cad": after,
                "before_native": native_before, "after_native": native_after}
    trace = {"accepted": True,
             "selected_before_ids": ["room-a"], "selected_after_ids": ["room-a"]}
    action = {"operation": "room_type", "source_mask": np.array([[True]])}
    action_sha = content_digest(action)
    transition = make_transition(branch="a", sequence=0, previous_head="external-anchor", kind="event",
                                 operation="room_type", event=event, trace=trace,
                                 action_content_sha256=action_sha, **snapshot)
    history = tmp_path / "history.json"
    write_archive(history, make_archive("a", [transition]))
    dxf, metadata = tmp_path / "dummy.dxf", tmp_path / "metadata.json"
    dxf.write_bytes(b"dummy bytes: not a DXF file")
    metadata.write_bytes(b'{"example":true}\n')
    path = tmp_path / "receipt.json"
    kwargs = {"dxf": dxf, "metadata": metadata, "history": history}
    receipt_sha = write_history_receipt(path, **kwargs, before_export_cad_content_sha256=transition["after_cad"],
                                        branch="a", head=transition["record_hash"], stage="after")
    verify_args = dict(**kwargs, branch="a", trusted_previous_head="external-anchor",
                       trusted_head=transition["record_hash"], snapshots=[snapshot],
                       action_evidence=[{"action": action, "trusted_sha256": action_sha}])
    result = verify_history_receipt(path, trusted_receipt_sha256=receipt_sha, **verify_args)
    assert result["receipt_integrity"] == result["file_integrity"] == result["history_integrity"] == "pass"
    assert result["declared_terminal_content_match"] == "pass"
    assert result["geometry"] == "not_evaluated" and result["permissions"] == "unknown"
    assert result["history_verification"]["status"] == "unknown"
    assert verify_history_receipt(path, trusted_receipt_sha256=None, **verify_args)["receipt_integrity"] == "unknown"
    missing = {**verify_args, "snapshots": [None]}
    assert verify_history_receipt(path, trusted_receipt_sha256=receipt_sha, **missing)["history_integrity"] == "unknown"
    missing_action = {**verify_args, "action_evidence": None}
    assert verify_history_receipt(path, trusted_receipt_sha256=receipt_sha, **missing_action)["history_integrity"] == "unknown"
    with pytest.raises(FileExistsError):
        write_history_receipt(path, **kwargs, before_export_cad_content_sha256=transition["after_cad"],
                              branch="a", head=transition["record_hash"], stage="after")
    dxf.write_bytes(b"changed")
    assert verify_history_receipt(path, trusted_receipt_sha256=receipt_sha, **verify_args)["file_integrity"] == "fail"
