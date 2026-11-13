import copy
import hashlib

import numpy as np
import pytest

from floorflow.representation.cad08_history_archive import (
    append_transition,
    make_archive,
    read_archive,
    verify_archive,
    write_archive,
)
from floorflow.representation.cad08_transition_history import make_transition


def test_archive_read_append_and_verify_actual_stage_snapshots(tmp_path):
    room_event = {"operation": "room_type", "parents": ["room-a"], "children": ["room-a"]}
    opening_event = {"operation": "opening_delete", "parents": ["opening-b"], "children": []}
    room_trace = {"operation": "room_type", "accepted": True,
                  "selected_before_ids": ["room-a"], "selected_after_ids": ["room-a"]}
    opening_trace = {"operation": "opening_delete", "accepted": True,
                     "selected_before_ids": ["opening-b"], "selected_after_ids": []}
    documents = [
        {"objects": [{"id": "room-a", "label": 1}, {"id": "opening-a"}], "lineage": []},
        {"objects": [{"id": "room-a", "label": 2}, {"id": "opening-a"}], "lineage": [room_event]},
        {"objects": [{"id": "room-b", "label": 2}, {"id": "opening-b"}], "lineage": []},
        {"objects": [{"id": "room-b", "label": 2}], "lineage": [opening_event]},
    ]
    native = [{"room_logits": np.array([i], dtype=np.float32)} for i in range(4)]
    snapshots = [{"before_cad": documents[i], "after_cad": documents[i + 1],
                  "before_native": native[i], "after_native": native[i + 1]} for i in range(3)]
    first = make_transition(branch="direct", sequence=0, previous_head="anchor", kind="event",
                            operation="room_type", event=room_event, trace=room_trace, **snapshots[0])
    initial = make_archive("direct", [first])
    path1 = tmp_path / "history1.json"
    file_hash1 = write_archive(path1, initial)
    loaded, receipt = read_archive(path1, trusted_file_sha256=file_hash1)
    assert receipt["file_integrity"] == "pass" and loaded == initial

    rebuild = make_transition(branch="direct", sequence=1, previous_head=first["record_hash"], kind="rebuild",
                              operation=None, **snapshots[1])
    second = append_transition(loaded, rebuild, trusted_current_head=first["record_hash"])
    deletion = make_transition(branch="direct", sequence=2, previous_head=rebuild["record_hash"], kind="event",
                               operation="opening_delete", event=opening_event, trace=opening_trace, **snapshots[2])
    full = append_transition(second, deletion, trusted_current_head=rebuild["record_hash"])
    path2 = tmp_path / "history2.json"
    file_hash2 = write_archive(path2, full)
    assert hashlib.sha256(path1.read_bytes()).hexdigest() == file_hash1
    with pytest.raises(FileExistsError):
        write_archive(path1, full)
    reread, receipt2 = read_archive(path2, trusted_file_sha256=file_hash2)
    assert receipt2["file_integrity"] == "pass"
    result = verify_archive(reread, snapshots, branch="direct", trusted_previous_head="anchor",
                            trusted_head=deletion["record_hash"])
    assert result["link_integrity"] == "pass" and result["status"] == "unknown"
    assert result["cross_rebuild_identity"] == "unknown"
    missing = verify_archive(reread, [snapshots[0], None, snapshots[2]], branch="direct",
                             trusted_previous_head="anchor", trusted_head=deletion["record_hash"])
    assert missing["link_integrity"] == "unknown"
    altered = copy.deepcopy(snapshots)
    altered[2]["before_cad"]["objects"][1]["id"] = "other-opening"
    assert verify_archive(reread, altered, branch="direct", trusted_previous_head="anchor",
                          trusted_head=deletion["record_hash"])["status"] == "fail"
    path2.write_bytes(path2.read_bytes() + b" ")
    assert read_archive(path2, trusted_file_sha256=file_hash2)[1]["file_integrity"] == "fail"
    assert read_archive(path1, trusted_file_sha256=None)[1]["file_integrity"] == "unknown"
