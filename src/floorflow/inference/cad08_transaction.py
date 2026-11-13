"""Explicit new CPU typed CAD transactions with proposal and direct-action modes."""

from __future__ import annotations

import copy

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union

from floorflow.inference.cad08_opening_lifecycle import apply_opening_lifecycle
from floorflow.representation.cad08_state import (
    EPS,
    LayeredCad08State,
    encoded_geometry,
    fingerprint,
    mask_geometry,
    object_record,
    raster_geometry,
    rle_encode,
)


def bind_before(state: LayeredCad08State, support: np.ndarray, family: str, label: int | None = None) -> dict:
    """Bind only input support to before geometry at frozen IoU >= .5."""
    source = mask_geometry(support)
    minimum = 2 if family == "openings" else 8
    candidates = []
    best = 0.0
    for obj in state.objects(family):
        g = shape(obj["geometry"])
        if g.area < minimum or (label is not None and obj["label"] != label):
            continue
        union = source.union(g).area
        iou = source.intersection(g).area / union if union else 0.0
        best = max(best, iou)
        if iou >= 0.5:
            candidates.append({"id": obj["id"], "iou": float(iou)})
    return {
        "status": "unique" if len(candidates) == 1 else "ambiguous" if candidates else "no_source_instance",
        "id": candidates[0]["id"] if len(candidates) == 1 else None,
        "candidates": candidates,
        "best_iou": float(best),
        "binding_uses_after": False,
    }


def bind_room_reference(
    state: LayeredCad08State,
    support: np.ndarray,
    label: int | None = None,
    room_id: str | None = None,
) -> dict:
    """Bind by geometry by default, or resolve an explicit current/aliased runtime room ID."""
    if room_id is None:
        return bind_before(state, support, "rooms", label)
    resolution = resolve_room_alias(state, room_id)
    if resolution["status"] not in ("current", "alias_resolved"):
        return {
            "status": "no_source_instance",
            "id": None,
            "candidates": [],
            "best_iou": 0.0,
            "binding_uses_after": False,
            "reason": "explicit_room_id_has_no_live_alias_target",
            "alias_resolution": resolution,
        }
    target_id = resolution["room_id"]
    target = next(obj for obj in state.objects("rooms") if obj["id"] == target_id)
    if label is not None and target["label"] != label:
        return {
            "status": "no_source_instance",
            "id": None,
            "candidates": [],
            "best_iou": 0.0,
            "binding_uses_after": False,
            "reason": "explicit_room_id_label_mismatch",
            "alias_resolution": resolution,
        }
    support_geometry = mask_geometry(support)
    support_area = float(support_geometry.area)
    overlap = float(support_geometry.intersection(shape(target["geometry"])).area)
    coverage = overlap / support_area if support_area > EPS else 0.0
    if coverage < 0.5:
        return {
            "status": "no_source_instance",
            "id": None,
            "candidates": [],
            "best_iou": 0.0,
            "binding_uses_after": False,
            "reason": "explicit_room_id_does_not_cover_input_support",
            "alias_resolution": resolution,
            "support_coverage": coverage,
        }
    target_area = float(shape(target["geometry"]).area)
    union = support_area + target_area - overlap
    return {
        "status": "unique",
        "id": target_id,
        "candidates": [{"id": target_id, "iou": overlap / union if union else 0.0}],
        "best_iou": overlap / union if union else 0.0,
        "binding_uses_after": False,
        "source": "explicit_room_id_or_lineage_alias_from_input_state",
        "requested_room_id": room_id,
        "alias_resolution": resolution,
        "support_coverage": coverage,
    }


def create_binding(state: LayeredCad08State, action: dict) -> dict:
    family = (
        "rooms"
        if action["operation"].startswith("room_")
        else "openings"
        if action["operation"] == "opening_delete"
        else "fixtures"
    )
    source = (
        bind_room_reference(
            state,
            action["source_mask"],
            action.get("source_class_id"),
            action.get("source_room_id"),
        )
        if family == "rooms"
        else bind_before(state, action["source_mask"], family)
    )
    destination = (
        bind_room_reference(
            state,
            action["destination_mask"],
            action.get("target_class_id"),
            action.get("destination_room_id"),
        )
        if action["operation"] == "room_remove" and not action["destination_ambiguous"]
        else None
    )
    return {
        "source": source,
        "destination": destination,
        "destination_ambiguous": action["destination_ambiguous"],
        "no_spatial_change": action["operation"] == "room_add" and not action["split_mask"].any(),
        "binding_uses_after": False,
    }


def resolve_room_alias(state: LayeredCad08State, room_id: str) -> dict:
    """Resolve retired room IDs through explicit merge aliases, without spatial guessing."""
    live_ids = {obj["id"] for obj in state.objects("rooms")}
    if room_id in live_ids:
        return {"status": "current", "room_id": room_id, "path": [room_id]}
    aliases = state.document.get("room_aliases", {})
    path = [room_id]
    seen = {room_id}
    current = room_id
    while current in aliases:
        current = aliases[current]
        path.append(current)
        if current in seen:
            return {"status": "invalid_cycle", "room_id": None, "path": path}
        seen.add(current)
        if current in live_ids:
            return {"status": "alias_resolved", "room_id": current, "path": path}
    return {"status": "unknown", "room_id": None, "path": path, "reason": "no_live_room_alias_target"}


def _record_room_alias(state: LayeredCad08State, retired_id: str, retained_id: str) -> dict:
    live_ids = {obj["id"] for obj in state.objects("rooms")}
    if retired_id in live_ids or retained_id not in live_ids or retired_id == retained_id:
        raise ValueError("room alias must map a retired ID to a different live room ID")
    aliases = state.document.setdefault("room_aliases", {})
    # Keep aliases flat so every historical ID points directly to a live room.
    for alias in list(aliases):
        current = alias
        seen = set()
        while current in aliases and current not in seen:
            seen.add(current)
            current = aliases[current]
        if current == retired_id:
            aliases[alias] = retained_id
    aliases[retired_id] = retained_id
    return {retired_id: retained_id}


def _revalidate_room_relations(
    before: LayeredCad08State,
    after: LayeredCad08State,
    affected_geometry,
    changed_room_ids: set[str],
) -> dict:
    before_relations = before.document["relationships"]["fixture_room"]
    after_relations = after.document["relationships"]["fixture_room"]
    touched: dict[str, dict] = {}
    unaffected_equal = True
    after_fixtures = {obj["id"]: obj for obj in after.objects("fixtures")}
    rooms = [(obj, shape(obj["geometry"])) for obj in after.objects("rooms")]
    for fixture in before.objects("fixtures"):
        fixture_id = fixture["id"]
        geometry = shape(fixture["geometry"])
        old_relation = before_relations.get(fixture_id, {})
        old_candidates = {item.get("room_id", item.get("id")) for item in old_relation.get("candidates", [])}
        is_touched = (
            old_relation.get("room_id") in changed_room_ids
            or bool(old_candidates & changed_room_ids)
            or geometry.intersection(affected_geometry).area > EPS
        )
        if not is_touched:
            unaffected_equal &= old_relation == after_relations.get(fixture_id, {})
            continue
        if fixture_id not in after_fixtures:
            touched[fixture_id] = {
                "before": old_relation,
                "after": None,
                "status": "unknown",
                "reason": "fixture_record_missing_after_room_transaction",
            }
            continue
        overlap_candidates = []
        containing_ids = []
        for room, room_geometry in rooms:
            overlap = float(geometry.intersection(room_geometry).area)
            if overlap > EPS:
                overlap_candidates.append({"room_id": room["id"], "overlap_area": overlap})
            if geometry.difference(room_geometry).area <= EPS:
                containing_ids.append(room["id"])
        overlap_candidates.sort(key=lambda item: (-item["overlap_area"], item["room_id"]))
        if len(containing_ids) == 1:
            validated = {
                "status": "unique",
                "room_id": containing_ids[0],
                "candidates": overlap_candidates,
                "source": "post_room_transaction_full_fixture_containment",
            }
            after_relations[fixture_id] = validated
            status, reason = "unique", None
        else:
            reason = (
                "multiple_room_faces_contain_fixture"
                if len(containing_ids) > 1
                else "no_single_room_contains_full_fixture_geometry"
            )
            validated = {
                "status": "unknown",
                "room_id": None,
                "candidates": overlap_candidates,
                "source": "post_room_transaction_full_fixture_containment",
                "reason": reason,
            }
            after_relations[fixture_id] = validated
            status = "unknown"
        touched[fixture_id] = {"before": old_relation, "after": validated, "status": status, "reason": reason}

    return {
        "fixture_room": {
            "touched_count": len(touched),
            "touched_status_counts": {
                status: sum(row["status"] == status for row in touched.values())
                for status in ("unique", "unknown")
            },
            "unaffected_relations_exact": unaffected_equal,
            "rows": touched,
        },
        "opening_host": {
            "status": "unchanged"
            if before.document["relationships"]["opening_host"] == after.document["relationships"]["opening_host"]
            else "changed",
            "checked_opening_count": len(before.document["relationships"]["opening_host"]),
            "rows_exact": before.document["relationships"]["opening_host"] == after.document["relationships"]["opening_host"],
        },
    }


def apply_transaction(
    before: LayeredCad08State, action: dict, *, proposal: dict | None = None
) -> tuple[LayeredCad08State, dict]:
    """Consume saved neural proposal or execute the separately named direct baseline.

    Proposal add geometry uses only after room labels inside the bound source;
    the original requested split support is used only by direct-action mode.
    No control-after rendering is accepted by this API.
    """
    if action.get("operation") in {"opening_add", "opening_replace"}:
        return apply_opening_lifecycle(before, action, proposal=proposal)
    state = LayeredCad08State.from_dict(before.to_dict())
    binding = create_binding(before, action)
    op = action["operation"]
    if op.startswith("room_"):
        state.document.setdefault("room_aliases", {})
    trace = {
        "implementation": "new_typed_CAD08_transaction_not_historical_network",
        "mode": "neural_proposal" if proposal is not None else "direct_action",
        "binding": binding,
        "proposal_consumed": False,
        "accepted": False,
        "reasons": [],
        "selected_before_ids": [],
        "selected_after_ids": [],
    }
    if op == "room_remove" and binding["destination_ambiguous"]:
        trace["reasons"].append("input_destination_instance_not_unique")
        return state, trace
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
    objects = {v["id"]: v for v in state.document["objects"]}
    source_id = binding["source"]["id"]
    source = objects[source_id]
    g = shape(source["geometry"])
    mask = raster_geometry(g, tuple(state.document["frame"]))
    affected_room_geometry = None
    changed_room_ids: set[str] = set()
    trace["selected_before_ids"] = [source_id]
    if op == "room_remove":
        if not binding["destination"] or binding["destination"]["status"] != "unique":
            trace["reasons"].append("initial_destination_instance_missing_or_ambiguous")
            return state, trace
        trace["selected_before_ids"].append(binding["destination"]["id"])
    if op == "opening_delete":
        host = before.document["relationships"]["opening_host"][source_id]
        trace["opening_host_before"] = copy.deepcopy(host)
        if host["status"] != "unique":
            trace["reasons"].append("before_opening_host_missing_or_ambiguous")
            return state, trace
        host_geometry = next(shape(h["geometry"]) for h in before.document["wall_hosts"] if h["id"] == host["host_id"])
        aperture = g.intersection(host_geometry)
        trace["host_aperture"] = encoded_geometry(aperture)
        if aperture.is_empty:
            trace["reasons"].append("empty_host_aperture")
            return state, trace
    if proposal is not None:
        field = "room_labels" if op.startswith("room_") else "opening" if op == "opening_delete" else "furniture"
        values = proposal[field][mask]
        trace["proposal_consumed"] = True
        trace["proposal_selected_values_sha256"] = fingerprint(values.tolist())
        if op in ("room_type", "room_remove"):
            labels, counts = np.unique(values, return_counts=True)
            winners = labels[counts == counts.max()] if len(counts) else []
            accepted = (
                len(winners) == 1
                and int(winners[0]) == action["target_class_id"]
                and counts.max() / max(1, len(values)) > 0.5
            )
            trace["proposal_target_fraction"] = (
                float(np.mean(values == action["target_class_id"])) if len(values) else 0
            )
            if not accepted:
                trace["reasons"].append("proposal_did_not_support_requested_object_class")
                return state, trace
        elif op.endswith("delete") and np.any(values != 0):
            trace["reasons"].append("proposal_retained_selected_family_geometry")
            return state, trace
    if op == "room_type":
        old_label = source["label"]
        geometry_hash = fingerprint(source["geometry"])
        source["label"] = int(action["target_class_id"])
        trace["selected_after_ids"] = [source_id]
        state.document["lineage"].append(
            {
                "operation": op,
                "parents": [source_id],
                "children": [source_id],
                "identity_preserved": True,
                "label_before": old_label,
                "label_after": int(action["target_class_id"]),
                "geometry_inherited_exactly": True,
                "geometry_sha256": geometry_hash,
            }
        )
    elif op == "room_add":
        new_mask = (
            mask & (proposal["room_labels"] == action["target_class_id"])
            if proposal is not None
            else mask & action["split_mask"]
        )
        new_geometry = mask_geometry(new_mask)
        remainder = g.difference(new_geometry)
        if new_geometry.area < 8 or remainder.area < 8:
            trace["reasons"].append("proposal_or_direct_split_has_empty_or_small_part")
            return state, trace
        child_id = f"{source_id}:split:{action['index']}"
        if child_id in objects:
            trace["reasons"].append("split_child_runtime_id_already_exists")
            return LayeredCad08State.from_dict(before.to_dict()), trace
        parent_geometry_hash = fingerprint(source["geometry"])
        affected_room_geometry = g
        changed_room_ids = {source_id, child_id}
        source["geometry"] = encoded_geometry(remainder)
        child = object_record(
            state.document["namespace"], "rooms", action["target_class_id"], new_geometry, identity=child_id
        )
        state.document["objects"].append(child)
        trace["selected_after_ids"] = [source_id, child_id]
        state.document["lineage"].append(
            {
                "operation": op,
                "parents": [source_id],
                "children": [source_id, child_id],
                "parent_geometry_sha256_before": parent_geometry_hash,
                "child_faces": {
                    source_id: {
                        "role": "retained_remainder",
                        "geometry_sha256": fingerprint(source["geometry"]),
                    },
                    child_id: {"role": "requested_split", "geometry_sha256": fingerprint(child["geometry"])},
                },
                "split_geometry_source": "saved_neural_proposal_inside_bound_source"
                if proposal is not None
                else "original_request_split_geometry_direct_action",
            }
        )
    elif op == "room_remove":
        destination_id = binding["destination"]["id"]
        if destination_id == source_id:
            trace["reasons"].append("source_and_destination_are_same_runtime_object")
            return state, trace
        destination = objects[destination_id]
        destination_before_geometry = shape(destination["geometry"])
        affected_room_geometry = unary_union([g, destination_before_geometry])
        changed_room_ids = {source_id, destination_id}
        merged_geometry = g.union(destination_before_geometry)
        destination["geometry"] = encoded_geometry(merged_geometry)
        state.document["objects"] = [v for v in state.document["objects"] if v["id"] != source_id]
        aliases_added = _record_room_alias(state, source_id, destination_id)
        trace["selected_after_ids"] = [destination_id]
        state.document["lineage"].append(
            {
                "operation": op,
                "parents": [source_id, destination_id],
                "children": [destination_id],
                "retained_destination_id": destination_id,
                "retired_source_id": source_id,
                "room_aliases_added": aliases_added,
                "merged_geometry_sha256": fingerprint(encoded_geometry(merged_geometry)),
                "no_corridor_or_wall_synthesis": True,
            }
        )
    else:
        state.document["objects"] = [v for v in state.document["objects"] if v["id"] != source_id]
        if op == "opening_delete":
            fill = aperture if proposal is None else aperture.intersection(mask_geometry(proposal["wall"].astype(bool)))
            walls = state.objects("wall_material")
            if walls:
                walls[0]["geometry"] = encoded_geometry(shape(walls[0]["geometry"]).union(fill))
                trace["selected_wall_material_id"] = walls[0]["id"]
            elif not fill.is_empty:
                wall = object_record(state.document["namespace"], "wall_material", 1, fill)
                state.document["objects"].append(wall)
                trace["selected_wall_material_id"] = wall["id"]
            trace["wall_fill_source"] = (
                "fixed_before_host_aperture_intersect_saved_proposal_wall"
                if proposal is not None
                else "direct_fixed_before_host_aperture"
            )
        state.document["lineage"].append(
            {"operation": op, "parents": [source_id], "children": [], "nonselected_objects_inherited": True}
        )
    trace["accepted"] = True
    if op == "room_type":
        state.document["relationships"] = copy.deepcopy(before.document["relationships"])
    else:
        state.refresh_relationships()
    if op == "opening_delete":
        live_opening_ids = {obj["id"] for obj in state.objects("openings")}
        state.document["relationships"]["fixture_room"] = copy.deepcopy(
            before.document["relationships"]["fixture_room"]
        )
        state.document["relationships"]["opening_host"] = {
            opening_id: copy.deepcopy(relation)
            for opening_id, relation in before.document["relationships"]["opening_host"].items()
            if opening_id in live_opening_ids
        }
    elif op == "furniture_delete":
        live_fixture_ids = {obj["id"] for obj in state.objects("fixtures")}
        state.document["relationships"]["fixture_room"] = {
            fixture_id: copy.deepcopy(relation)
            for fixture_id, relation in before.document["relationships"]["fixture_room"].items()
            if fixture_id in live_fixture_ids
        }
        state.document["relationships"]["opening_host"] = copy.deepcopy(
            before.document["relationships"]["opening_host"]
        )
    if op.startswith("room_"):
        before_non_room = [obj for obj in before.document["objects"] if obj["family"] != "rooms"]
        after_non_room = [obj for obj in state.document["objects"] if obj["family"] != "rooms"]
        opening_host_exact = (
            before.document["relationships"]["opening_host"] == state.document["relationships"]["opening_host"]
        )
        if before_non_room != after_non_room or not opening_host_exact:
            trace["accepted"] = False
            trace["selected_after_ids"] = []
            trace["reasons"].append("room_transaction_changed_unselected_records_or_opening_host_relations")
            return LayeredCad08State.from_dict(before.to_dict()), trace
        if op in ("room_add", "room_remove"):
            relation_report = _revalidate_room_relations(
                before, state, affected_room_geometry, changed_room_ids
            )
            if not relation_report["fixture_room"]["unaffected_relations_exact"]:
                trace["accepted"] = False
                trace["selected_after_ids"] = []
                trace["reasons"].append("room_relation_revalidation_guard_failed")
                return LayeredCad08State.from_dict(before.to_dict()), trace
            state.document["lineage"][-1]["relation_revalidation"] = relation_report
            trace["room_relation_revalidation"] = relation_report
        elif (
            before.document["relationships"]["fixture_room"] != state.document["relationships"]["fixture_room"]
        ):
            trace["accepted"] = False
            trace["selected_after_ids"] = []
            trace["reasons"].append("room_retype_changed_fixture_room_relations")
            return LayeredCad08State.from_dict(before.to_dict()), trace
        else:
            trace["room_relation_revalidation"] = {
                "fixture_room": {"status": "unchanged", "rows_exact": True},
                "opening_host": {"status": "unchanged", "rows_exact": True},
            }
            state.document["lineage"][-1]["relation_revalidation"] = trace["room_relation_revalidation"]
    state.document["visible_reference"] = rle_encode(state.render()["prediction"])
    state.document["provenance"]["decoded_fields_unchanged"] = False
    state.document["provenance"]["transaction"] = trace["mode"]
    return state, trace
