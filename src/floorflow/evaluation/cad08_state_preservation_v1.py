"""Selection, preservation, and identity checks for CAD08 transitions."""

from __future__ import annotations

from collections import Counter

import numpy as np
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

from floorflow.evaluation.cad08_state_metrics_v1 import (
    EPS,
    FAMILIES,
    canonical,
    decode_rle,
    tri_and,
)


def _selection(before: dict, after: dict, row: dict) -> tuple[set, set, bool]:
    binding = row.get("effect", {}).get("binding", {})
    before_ids = {v["id"] for v in before["objects"]}
    selected = {
        v["id"]
        for name in ("source", "destination")
        if (v := binding.get(name)) and v.get("status") == "unique" and v.get("id") in before_ids
    }
    known = (binding.get("source") or {}).get("status") == "unique" and (
        row["operation"] != "room_remove"
        or not binding.get("destination_ambiguous")
        and (binding.get("destination") or {}).get("status") == "unique"
    )
    after_selected = set(row.get("effect", {}).get("selected_after_ids", []))
    trace = row.get("transaction")
    if trace:
        after_selected.update(trace.get("selected_after_ids", []))
        if not trace.get("accepted"):
            after_selected.update(selected)
    after_selected.update(v["id"] for v in after["objects"] if v["id"] in selected)
    return selected, after_selected, known

def preservation_state(
    before: dict, after: dict, row: dict, after_g: dict, split_mask: np.ndarray | None = None
) -> dict:
    """Preserve all unselected object fields and references, not just geometry."""
    selected, selected_after, known = _selection(before, after, row)
    op = row["operation"]
    bo = {v["id"]: v for v in before["objects"]}
    ao = {v["id"]: v for v in after["objects"]}
    checks, reasons, changes = [], [], []
    per_family = {}
    for family in FAMILIES:
        if family == "wall_material":
            continue
        left = {k: v for k, v in bo.items() if v["family"] == family and k not in selected}
        right = {k: v for k, v in ao.items() if v["family"] == family and k not in selected_after}
        exact = Counter(canonical(v) for v in left.values()) == Counter(canonical(v) for v in right.values())
        if not known and family == (
            "rooms" if op.startswith("room_") else "openings" if op == "opening_delete" else "fixtures"
        ):
            field_status = "pass" if exact else "unknown"
        else:
            field_status = "pass" if exact else "fail"
        checks.append(field_status)
        per_family[family] = {
            "status": field_status,
            "before_records": len(left),
            "after_records": len(right),
            "missing_IDs": sorted(set(left) - set(right)),
            "extra_IDs": sorted(set(right) - set(left)),
            "changed_fields": {
                key: sorted(k for k in set(left[key]) | set(right[key]) if left[key].get(k) != right[key].get(k))
                for key in set(left) & set(right)
                if left[key] != right[key]
            },
        }
        if field_status != "pass":
            reasons.append("unselected_" + family + "_records_" + field_status)
    for name in ("schema", "frame", "namespace", "footprint", "wall_hosts"):
        if before.get(name) != after.get(name):
            checks.append("fail")
            reasons.append("unselected_state_field_changed:" + name)
    bw, aw = (
        [v for v in before["objects"] if v["family"] == "wall_material"],
        [v for v in after["objects"] if v["family"] == "wall_material"],
    )
    if op != "opening_delete":
        wall_ok = Counter(canonical(v) for v in bw) == Counter(canonical(v) for v in aw)
    elif known:
        wall_ok = len(bw) == len(aw) and all(
            {k: v for k, v in b.items() if k != "geometry"} == {k: v for k, v in a.items() if k != "geometry"}
            for b, a in zip(bw, aw, strict=True)
        )
        if wall_ok:
            allowed = unary_union([shape(bo[k]["geometry"]) for k in selected if bo[k]["family"] == "openings"])
            wall_ok = (
                unary_union([shape(v["geometry"]) for v in bw])
                .symmetric_difference(unary_union([shape(v["geometry"]) for v in aw]))
                .difference(allowed)
                .area
                <= EPS
            )
    else:
        wall_ok = bw == aw
    if not wall_ok:
        checks.append("fail" if known or op != "opening_delete" else "unknown")
        reasons.append("wall_material_identity_or_nonselected_fields_changed")
    br, ar = before.get("relationships", {}), after.get("relationships", {})
    allowed_rooms_before = {k for k in selected if bo[k]["family"] == "rooms"}
    allowed_rooms_after = {k for k in selected_after if k in ao and ao[k]["family"] == "rooms"}
    for table, family in (("opening_host", "openings"), ("fixture_room", "fixtures")):
        left, right = br.get(table, {}), ar.get(table, {})
        for identity in set(left) | set(right):
            if identity in selected and (
                op == "opening_delete" and family == "openings" or op == "furniture_delete" and family == "fixtures"
            ):
                continue
            if left.get(identity) == right.get(identity):
                continue
            authorized = False
            if (
                table == "fixture_room"
                and op in ("room_add", "room_remove")
                and known
                and identity in left
                and identity in right
                and identity in bo
                and identity in ao
                and bo[identity] == ao[identity]
            ):
                a, b = left[identity], right[identity]
                lmap = {v["id"]: v for v in a.get("candidates", [])}
                rmap = {v["id"]: v for v in b.get("candidates", [])}
                stable = set(lmap) | set(rmap)
                stable -= allowed_rooms_before | allowed_rooms_after
                endpoint_ok = (
                    a.get("room_id") == b.get("room_id")
                    or a.get("room_id") in allowed_rooms_before
                    and b.get("room_id") in allowed_rooms_after
                )
                extra_ok = {k: v for k, v in a.items() if k not in ("room_id", "status", "candidates")} == {
                    k: v for k, v in b.items() if k not in ("room_id", "status", "candidates")
                }
                authorized = endpoint_ok and extra_ok and all(lmap.get(k) == rmap.get(k) for k in stable)
            observed = "authorized_selected_room_update" if authorized else "unauthorized_relationship_change"
            changes.append(
                {
                    "table": table,
                    "source": identity,
                    "category": observed,
                    "before": left.get(identity),
                    "after": right.get(identity),
                }
            )
            if not authorized:
                checks.append("fail" if known or table == "opening_host" else "unknown")
                reasons.append(observed + ":" + table)
    for table in ("room_adjacency", "wall_room_incidence"):
        allowance = allowed_rooms_before | allowed_rooms_after if op in ("room_add", "room_remove") and known else set()
        left = Counter(canonical(v) for v in br.get(table, []) if not any(k in allowance for k in v))
        right = Counter(canonical(v) for v in ar.get(table, []) if not any(k in allowance for k in v))
        if left != right:
            checks.append("fail" if known else "unknown")
            reasons.append("unselected_graph_edges_changed:" + table)
    prefix = before.get("lineage", [])
    events = after.get("lineage", [])
    lineage_ok = events[: len(prefix)] == prefix
    for event in events[len(prefix) :]:
        lineage_ok &= (
            event.get("operation") == op
            and set(event.get("parents", [])) <= selected
            and bool(event.get("parents"))
            and set(event.get("children", [])) <= selected_after
            and all(k in ao for k in event.get("children", []))
        )
    if not lineage_ok:
        checks.append("fail")
        reasons.append("unauthorized_or_dangling_lineage")
    mutable = {"decoded_fields_unchanged", "transaction", "saved_visible_only_evaluated_before"}
    provenance_ok = {k: v for k, v in before.get("provenance", {}).items() if k not in mutable} == {
        k: v for k, v in after.get("provenance", {}).items() if k not in mutable
    }
    if not provenance_ok:
        checks.append("fail")
        reasons.append("unselected_provenance_fields_changed")
    diagnostic = {
        k: before.get("diagnostic_rasters", {}).get(k) == after.get("diagnostic_rasters", {}).get(k)
        for k in set(before.get("diagnostic_rasters", {})) | set(after.get("diagnostic_rasters", {}))
    }
    if diagnostic.get("room_support") is False:
        checks.append("fail")
        reasons.append("nonselected_model_auxiliary_support_changed")
    view = "unknown"
    if before.get("visible_reference") == after.get("visible_reference"):
        view = "pass"
    elif known:
        a, b = decode_rle(before["visible_reference"]), decode_rle(after["visible_reference"])
        allowed = unary_union(
            [
                shape(bo[k]["geometry"])
                for k in selected
                if bo[k]["family"] != "rooms"
                or op != "room_remove"
                or k == (row["effect"]["binding"].get("source") or {}).get("id")
            ]
        )
        yy, xx = np.indices(a.shape)
        mask = contains_xy(allowed, xx + 0.5, yy + 0.5)
        if op == "room_add":
            if split_mask is None:
                mask[:] = False
                view = "unknown"
            else:
                mask &= split_mask
        inside = contains_xy(shape(before["footprint"]), xx + 0.5, yy + 0.5)
        if op != "room_add" or split_mask is not None:
            view = "fail" if np.any((a != b) & inside & ~mask) else "pass"
    checks.append(view)
    if view != "pass":
        reasons.append("unselected_visible_reference_" + view)
    # Whole-state reference integrity is in G_state; P also prevents a new
    # dangling pointer from passing through an otherwise stable object record.
    if after_g["reference_validity"]["status"] == "fail":
        checks.append("fail")
        reasons.append("after_reference_integrity_failed")
    identity_status = tri_and(*checks)
    return {
        "P_state": tri_and(row["P"], identity_status),
        "P_legacy": row["P"],
        "identity_state_component": identity_status,
        "selection_unique_from_input_before": known,
        "selected_before_IDs": sorted(selected),
        "selected_after_IDs_from_frozen_effect_or_trace": sorted(selected_after),
        "families": per_family,
        "wall_material_records_preserved_in_authorized_domain": wall_ok,
        "relationship_changes": changes,
        "lineage_valid": bool(lineage_ok),
        "visible_nonselected_status": view,
        "auxiliary_and_unused_scaffold_exact": diagnostic,
        "unused_scaffold_is_not_CAD_wall_or_identity": True,
        "reasons": sorted(set(reasons)),
    }

__all__ = ["preservation_state"]
