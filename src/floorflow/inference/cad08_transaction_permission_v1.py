"""Versioned CAD08 add adapter: bound source AND input permission AND proposal.

This is a CPU execution-interface correction, not a neural model. The original
unbounded transaction remains available, unchanged, for exact historical replay.
"""

from __future__ import annotations

from shapely.geometry import shape

from floorflow.inference.cad08_transaction import apply_transaction, create_binding
from floorflow.representation.cad08_state import (
    LayeredCad08State,
    encoded_geometry,
    fingerprint,
    mask_geometry,
    object_record,
    raster_geometry,
    rle_encode,
)


def apply_permission_transaction(
    before: LayeredCad08State, action: dict, *, proposal: dict | None = None
) -> tuple[LayeredCad08State, dict]:
    """Restrict cached neural add cells to the original input split permission.

    No reference target or control-after image is accepted. Within permission,
    every new target cell must already carry that class in the saved proposal.
    Other operation families and the explicitly direct baseline delegate to the
    unchanged historical CAD08 transaction.
    """
    if action["operation"] != "room_add" or proposal is None:
        return apply_transaction(before, action, proposal=proposal)
    state = LayeredCad08State.from_dict(before.to_dict())
    binding = create_binding(before, action)
    trace = {
        "implementation": "new_typed_CAD08_transaction_permission_v1_not_historical_network",
        "mode": "neural_proposal",
        "binding": binding,
        "proposal_consumed": False,
        "accepted": False,
        "reasons": [],
        "selected_before_ids": [],
        "selected_after_ids": [],
    }
    if binding["no_spatial_change"]:
        trace["reasons"].append("request_has_no_spatial_split_support")
        return state, trace
    if binding["source"]["status"] != "unique":
        trace["reasons"].append(
            "initial_source_instance_missing"
            if binding["source"]["status"] == "no_source_instance"
            else "source_instance_binding_ambiguous"
        )
        return state, trace
    source_id = binding["source"]["id"]
    source = next(v for v in state.document["objects"] if v["id"] == source_id)
    geometry = shape(source["geometry"])
    source_mask = raster_geometry(geometry, tuple(state.document["frame"]))
    trace["selected_before_ids"] = [source_id]
    trace["proposal_consumed"] = True
    trace["proposal_selected_values_sha256"] = fingerprint(proposal["room_labels"][source_mask].tolist())
    # split_mask authorizes writes; it never supplies target-class fill values.
    new_mask = source_mask & action["split_mask"] & (proposal["room_labels"] == action["target_class_id"])
    new_geometry = mask_geometry(new_mask)
    remainder = geometry.difference(new_geometry)
    if new_geometry.area < 8 or remainder.area < 8:
        trace["reasons"].append("proposal_or_direct_split_has_empty_or_small_part")
        return state, trace
    child_id = f"{source_id}:split:{action['index']}"
    source["geometry"] = encoded_geometry(remainder)
    child = object_record(
        state.document["namespace"], "rooms", action["target_class_id"], new_geometry, identity=child_id
    )
    state.document["objects"].append(child)
    trace["selected_after_ids"] = [source_id, child_id]
    state.document["lineage"].append(
        {
            "operation": "room_add",
            "parents": [source_id],
            "children": [source_id, child_id],
            "split_geometry_source": "saved_neural_proposal_inside_bound_source_and_original_input_split_permission",
        }
    )
    trace["accepted"] = True
    state.refresh_relationships()
    state.document["visible_reference"] = rle_encode(state.render()["prediction"])
    state.document["provenance"]["decoded_fields_unchanged"] = False
    state.document["provenance"]["transaction"] = trace["mode"]
    return state, trace
