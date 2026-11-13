"""Fail-closed CAD08 opening add/replace transactions for registered hosts."""

from __future__ import annotations

import copy

from shapely.geometry import shape

from floorflow.representation.cad08_state import (
    EPS,
    LayeredCad08State,
    encoded_geometry,
    fingerprint,
    object_record,
    rle_encode,
)


def _rejected(before: LayeredCad08State, reason: str, binding: dict | None = None):
    return LayeredCad08State.from_dict(before.to_dict()), {
        "implementation": "new_typed_CAD08_transaction_not_historical_network",
        "mode": "direct_action",
        "binding": binding or {"status": "unknown", "binding_uses_after": False},
        "accepted": False,
        "reasons": [reason],
        "selected_before_ids": [],
        "selected_after_ids": [],
    }


def apply_opening_lifecycle(
    before: LayeredCad08State,
    action: dict,
    *,
    proposal: dict | None = None,
) -> tuple[LayeredCad08State, dict]:
    """Add or replace one opening on an explicit, uniquely identified wall host.

    The supplied geometry is the decoded CAD aperture polygon. This routine
    never chooses a host from the after-state or from a spatial nearest match.
    It changes wall material only inside the old/new aperture polygons.
    """
    if proposal is not None:
        return _rejected(before, "opening_lifecycle_proposal_contract_not_defined")
    operation = action.get("operation")
    if operation not in {"opening_add", "opening_replace"}:
        return _rejected(before, "unsupported_opening_lifecycle_operation")

    host_id = action.get("host_id")
    hosts = [host for host in before.document.get("wall_hosts", []) if host.get("id") == host_id]
    binding = {
        "status": "unique" if len(hosts) == 1 else "ambiguous" if len(hosts) > 1 else "no_source_instance",
        "host_id": host_id if len(hosts) == 1 else None,
        "candidate_count": len(hosts),
        "binding_uses_after": False,
        "source": "explicit_host_id_from_committed_before_state",
    }
    if len(hosts) != 1:
        return _rejected(before, "input_host_id_not_unique", binding)
    host_geometry = shape(hosts[0]["geometry"])
    try:
        new_geometry = shape(action["opening_geometry"])
    except (KeyError, TypeError, ValueError):
        return _rejected(before, "opening_geometry_missing_or_invalid", binding)
    if new_geometry.is_empty or not new_geometry.is_valid or new_geometry.area <= EPS:
        return _rejected(before, "opening_geometry_empty_invalid_or_zero_area", binding)

    footprint = shape(before.document["footprint"])
    if new_geometry.difference(footprint).area > EPS:
        return _rejected(before, "opening_geometry_outside_footprint", binding)
    host_coverage = new_geometry.intersection(host_geometry).area / new_geometry.area
    if host_coverage < 0.999:
        return _rejected(before, "opening_geometry_not_covered_by_selected_host", binding)

    opening_id = action.get("opening_id")
    openings = before.objects("openings")
    replacement = None
    if operation == "opening_add":
        if not isinstance(opening_id, str) or not opening_id:
            return _rejected(before, "opening_add_requires_new_runtime_id", binding)
        if any(opening["id"] == opening_id for opening in before.document["objects"]):
            return _rejected(before, "opening_runtime_id_already_exists", binding)
    else:
        source_id = action.get("source_opening_id")
        matches = [opening for opening in openings if opening.get("id") == source_id]
        if len(matches) != 1:
            return _rejected(before, "source_opening_id_not_unique", binding)
        replacement = matches[0]
        opening_id = source_id
        existing_host = before.document["relationships"].get("opening_host", {}).get(source_id, {})
        if existing_host.get("status") != "unique" or existing_host.get("host_id") != host_id:
            return _rejected(before, "source_opening_not_uniquely_bound_to_selected_host", binding)
    try:
        opening_class_id = int(action["opening_class_id"])
    except (KeyError, TypeError, ValueError):
        return _rejected(before, "opening_class_id_missing_or_invalid", binding)
    if opening_class_id <= 0:
        return _rejected(before, "opening_class_id_must_be_positive", binding)

    for opening in openings:
        if replacement is not None and opening["id"] == replacement["id"]:
            continue
        if shape(opening["geometry"]).intersection(new_geometry).area > EPS:
            return _rejected(before, "opening_geometry_overlaps_another_opening", binding)

    wall_objects = before.objects("wall_material")
    if len(wall_objects) != 1:
        return _rejected(before, "requires_exactly_one_wall_material_object", binding)
    wall = wall_objects[0]
    wall_before = shape(wall["geometry"])
    wall_restored = wall_before
    old_geometry = None
    if replacement is not None:
        old_geometry = shape(replacement["geometry"])
        wall_restored = wall_restored.union(old_geometry)
    if wall_restored.intersection(new_geometry).area / new_geometry.area < 0.999:
        return _rejected(before, "wall_material_does_not_cover_new_aperture", binding)
    wall_after = wall_restored.difference(new_geometry)
    if wall_after.is_empty or not wall_after.is_valid:
        return _rejected(before, "wall_material_invalid_after_aperture_cut", binding)

    state = LayeredCad08State.from_dict(before.to_dict())
    state_wall = next(obj for obj in state.objects("wall_material") if obj["id"] == wall["id"])
    state_wall["geometry"] = encoded_geometry(wall_after)
    if replacement is not None:
        state.document["objects"] = [
            obj for obj in state.document["objects"] if obj["id"] != replacement["id"]
        ]
    opening = object_record(
        state.document["namespace"],
        "openings",
        opening_class_id,
        new_geometry,
        identity=opening_id,
    )
    state.document["objects"].append(opening)
    state.document["lineage"].append(
        {
            "operation": operation,
            "parents": [] if replacement is None else [replacement["id"]],
            "children": [opening_id],
            "identity_preserved": replacement is not None,
            "host_id": host_id,
            "old_geometry_sha256": None if old_geometry is None else fingerprint(encoded_geometry(old_geometry)),
            "new_geometry_sha256": fingerprint(encoded_geometry(new_geometry)),
            "wall_delta_scope": "old_and_new_aperture_geometry_only",
        }
    )
    state.refresh_relationships()
    relation = state.document["relationships"]["opening_host"].get(opening_id, {})
    if relation.get("status") != "unique" or relation.get("host_id") != host_id:
        return _rejected(before, "post_transaction_opening_host_relation_not_unique", binding)
    relation["source"] = "explicit_request_host_id_verified_by_unique_geometry_coverage"

    unaffected_opening_relations_equal = all(
        before.document["relationships"]["opening_host"].get(item["id"])
        == state.document["relationships"]["opening_host"].get(item["id"])
        for item in openings
        if replacement is None or item["id"] != replacement["id"]
    )
    if not unaffected_opening_relations_equal:
        return _rejected(before, "unselected_opening_host_relations_changed", binding)
    state.document["visible_reference"] = rle_encode(state.render()["prediction"])
    state.document["provenance"]["decoded_fields_unchanged"] = False
    state.document["provenance"]["transaction"] = "direct_action"
    trace = {
        "implementation": "new_typed_CAD08_transaction_not_historical_network",
        "mode": "direct_action",
        "binding": binding,
        "proposal_consumed": False,
        "accepted": True,
        "reasons": [],
        "selected_before_ids": [] if replacement is None else [replacement["id"]],
        "selected_after_ids": [opening_id],
        "opening_host_before": None
        if replacement is None
        else copy.deepcopy(before.document["relationships"]["opening_host"][replacement["id"]]),
        "opening_host_after": copy.deepcopy(relation),
        "wall_material_id": wall["id"],
        "wall_material_delta_area": float(wall_before.symmetric_difference(wall_after).area),
        "host_coverage": float(host_coverage),
    }
    return state, trace
