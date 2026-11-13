"""Run two direct room edits and verify an R2010 history package on CPU."""

import argparse
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.cad08_adapter import compose_visible
from floorflow.inference.cad08_transaction import apply_transaction
from floorflow.representation.cad08_content_digest import content_digest
from floorflow.representation.cad08_history_archive import make_archive, write_archive
from floorflow.representation.cad08_history_dxf_readback import verify_history_dxf_readback
from floorflow.representation.cad08_history_export import export_history_package
from floorflow.representation.cad08_state import LayeredCad08State
from floorflow.representation.cad08_transition_history import make_transition


def run(output_dir: Path) -> Path:
    taxonomy = FloorplanTaxonomy(
        ["background", "wall", "room:Living", "room:Bedroom", "room:Study"], {}, {}, {}, {}, {}
    )
    footprint = np.ones((48, 48), dtype=np.uint8)
    empty = np.zeros_like(footprint)
    wall = empty.copy()
    wall[:2] = 1
    decoded = {
        "footprint": footprint,
        "room_labels": np.full_like(footprint, 2),
        "wall": wall,
        "opening": empty,
        "furniture": empty,
    }
    decoded["prediction"] = compose_visible(decoded)
    initial = LayeredCad08State.from_decoded(decoded, taxonomy, "quickstart")
    parent_id = initial.objects("rooms")[0]["id"]
    right_half = np.zeros_like(footprint, dtype=bool)
    right_half[:, 24:] = True
    split_request = {
        "index": 1,
        "operation": "room_add",
        "source_room_id": parent_id,
        "source_mask": footprint.astype(bool),
        "split_mask": right_half,
        "destination_mask": np.zeros_like(right_half),
        "source_class_id": 2,
        "target_class_id": 3,
        "destination_ambiguous": False,
    }
    split, split_trace = apply_transaction(initial, split_request)
    if not split_trace["accepted"]:
        raise RuntimeError(f"Room split rejected: {split_trace['reasons']}")
    split_rooms = {room["id"]: room for room in split.objects("rooms")}
    (child_id,) = split_rooms.keys() - {parent_id}

    retype_request = {
        **split_request,
        "index": 2,
        "operation": "room_type",
        "source_room_id": child_id,
        "source_mask": right_half,
        "split_mask": np.zeros_like(right_half),
        "source_class_id": 3,
        "target_class_id": 4,
    }
    final, retype_trace = apply_transaction(split, retype_request)
    if not retype_trace["accepted"]:
        raise RuntimeError(f"Room reclassification rejected: {retype_trace['reasons']}")
    final_rooms = {room["id"]: room for room in final.objects("rooms")}
    expected_rooms = {**split_rooms, child_id: {**split_rooms[child_id], "label": 4}}
    if final_rooms != expected_rooms:
        raise RuntimeError("Reclassification changed an unexpected room field")
    print("Room split and reclassification passed (child ID and geometry preserved).")

    # The example supplies its own snapshots and trust anchors for this run.
    branch = "quickstart"
    anchor = head = content_digest(initial.to_dict())
    records, snapshots, action_evidence = [], [], []
    transitions = [(initial, split, split_request, split_trace), (split, final, retype_request, retype_trace)]
    for sequence, (before, after, action, trace) in enumerate(transitions):
        snapshot = {
            "before_cad": before.to_dict(),
            "after_cad": after.to_dict(),
            "before_native": before.render(),
            "after_native": after.render(),
        }
        action_hash = content_digest(action)
        record = make_transition(
            branch=branch,
            sequence=sequence,
            previous_head=head,
            kind="event",
            operation=action["operation"],
            event=after.document["lineage"][-1],
            trace=trace,
            action_content_sha256=action_hash,
            **snapshot,
        )
        head = record["record_hash"]
        records.append(record)
        snapshots.append(snapshot)
        action_evidence.append({"action": action, "trusted_sha256": action_hash})

    archive_path = output_dir / "source-history.json"
    archive_hash = write_archive(archive_path, make_archive(branch, records))
    package = output_dir / "package"
    evidence = {
        "branch": branch,
        "trusted_previous_head": anchor,
        "trusted_head": head,
        "snapshots": snapshots,
        "action_evidence": action_evidence,
    }
    final_cad = final.to_dict()
    exported = export_history_package(
        package,
        cad=final_cad,
        history_path=archive_path,
        trusted_history_sha256=archive_hash,
        stage="after-event",
        **evidence,
    )
    if exported["status"] != "exported_unvalidated":
        raise RuntimeError(f"History package export failed: {exported['reason']}")
    readback = verify_history_dxf_readback(
        package / "receipt.json",
        dxf=package / "plan.dxf",
        metadata=package / "metadata.json",
        history=package / "history.json",
        trusted_receipt_sha256=exported["receipt_sha256"],
        reference_cad=final_cad,
        **evidence,
    )
    statuses = {
        "files": readback["receipt"]["file_integrity"],
        "history": readback["receipt"]["history_integrity"],
        "reference": readback["reference_content"],
        "geometry": readback["geometry"],
        "metadata": readback["metadata"],
    }
    if any(status != "pass" for status in statuses.values()):
        raise RuntimeError(f"R2010 readback failed: {statuses}")
    print("R2010 readback passed (files, history, reference, geometry, metadata).")
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Existing empty directory for the example output")
    args = parser.parse_args()
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        parser.error("output_dir must be an existing empty directory")
    print(f"Package: {run(args.output_dir)}")
