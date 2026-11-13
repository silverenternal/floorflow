"""State-bound transaction entry for persisted proposals and request permissions."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from shapely.geometry import shape

from floorflow.inference.cad08_transaction import apply_transaction
from floorflow.representation.cad08_state import LayeredCad08State, fingerprint, raster_geometry


def apply_transaction_with_state_guard(
    before: LayeredCad08State,
    action: dict[str, Any],
    *,
    proposal: dict[str, Any] | None = None,
) -> tuple[LayeredCad08State, dict[str, Any]]:
    """Fail closed for stale proposals and intersect split output with split permission.

    Legacy calls remain on ``cad08_transaction.apply_transaction``. This entry
    requires a proposal fingerprint and leaves the historical transaction source
    untouched for reproducible baseline runs.
    """
    if proposal is None:
        return apply_transaction(before, action)

    actual_state_sha = fingerprint(before.document)
    expected_state_sha = proposal.get("_source_state_sha256")
    if not isinstance(expected_state_sha, str) or not expected_state_sha:
        return LayeredCad08State.from_dict(before.to_dict()), {
            "implementation": "state_guarded_typed_CAD08_transaction_v1",
            "mode": "neural_proposal",
            "binding": None,
            "proposal_consumed": False,
            "accepted": False,
            "reasons": ["proposal_source_state_fingerprint_missing"],
            "actual_source_state_sha256": actual_state_sha,
        }
    if expected_state_sha != actual_state_sha:
        return LayeredCad08State.from_dict(before.to_dict()), {
            "implementation": "state_guarded_typed_CAD08_transaction_v1",
            "mode": "neural_proposal",
            "binding": None,
            "proposal_consumed": False,
            "accepted": False,
            "reasons": ["stale_proposal_source_state_mismatch"],
            "expected_source_state_sha256": expected_state_sha,
            "actual_source_state_sha256": actual_state_sha,
        }

    validation_state, validation_trace = apply_transaction(before, action, proposal=proposal)
    if not validation_trace.get("accepted") or action.get("operation") != "room_add":
        validation_trace["implementation"] = "state_guarded_typed_CAD08_transaction_v1"
        validation_trace["source_state_sha256"] = actual_state_sha
        validation_trace["expected_source_state_sha256"] = expected_state_sha
        return validation_state, validation_trace

    source_id = validation_trace["selected_before_ids"][0]
    source = next(room for room in before.objects("rooms") if room["id"] == source_id)
    source_mask = raster_geometry(shape(source["geometry"]), tuple(before.document["frame"]))
    requested_permission = np.asarray(action["split_mask"], dtype=bool)
    proposal_labels = np.asarray(proposal["room_labels"])
    if requested_permission.shape != source_mask.shape or proposal_labels.shape != source_mask.shape:
        return LayeredCad08State.from_dict(before.to_dict()), {
            **validation_trace,
            "implementation": "state_guarded_typed_CAD08_transaction_v1",
            "proposal_consumed": True,
            "accepted": False,
            "reasons": ["split_permission_or_proposal_frame_mismatch"],
            "source_state_sha256": actual_state_sha,
            "expected_source_state_sha256": expected_state_sha,
        }

    authorized_predicted_split = source_mask & requested_permission & (proposal_labels == action["target_class_id"])
    authorized_action = copy.deepcopy(action)
    authorized_action["split_mask"] = authorized_predicted_split
    after, trace = apply_transaction(before, authorized_action)
    trace.update({
        "implementation": "state_guarded_typed_CAD08_transaction_v1",
        "mode": "neural_proposal",
        "proposal_consumed": True,
        "proposal_selected_values_sha256": validation_trace.get("proposal_selected_values_sha256"),
        "proposal_target_fraction": validation_trace.get("proposal_target_fraction"),
        "authorized_split_pixels": int(authorized_predicted_split.sum()),
        "split_geometry_source": "saved_neural_proposal_intersect_original_request_split_mask_and_bound_source",
        "source_state_sha256": actual_state_sha,
        "expected_source_state_sha256": expected_state_sha,
    })
    if trace.get("accepted"):
        after.document["lineage"][-1]["split_geometry_source"] = trace["split_geometry_source"]
        after.document["provenance"]["transaction"] = "neural_proposal"
        return after, trace
    trace["proposal_consumed"] = True
    return LayeredCad08State.from_dict(before.to_dict()), trace
