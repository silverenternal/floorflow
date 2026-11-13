"""Strict operation effects and preservation for the new CAD08 consumer."""

from __future__ import annotations

from collections import Counter

import numpy as np
from shapely import normalize
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from floorflow.inference.cad08_transaction import bind_before, create_binding
from floorflow.representation.cad08_state import EPS, canonical, mask_geometry, raster_geometry


def tri_and(*values: str) -> str:
    return "fail" if "fail" in values else "unknown" if "unknown" in values else "pass"


def status(value: bool) -> str:
    return "pass" if value else "fail"


def geometry_key(obj: dict) -> str:
    return canonical([obj["family"], obj["label"], mapping(normalize(shape(obj["geometry"])))])


def geometry_normalized_record_key(record: dict) -> str:
    """Compare a full CAD record while ignoring only lossless geometry ordering."""
    normalized = dict(record)
    normalized["geometry"] = mapping(normalize(shape(record["geometry"])))
    return canonical(normalized)


def _relationship_rows(rows: dict, excluded_ids: set[str]) -> Counter:
    return Counter(
        canonical([identity, relation])
        for identity, relation in rows.items()
        if identity not in excluded_ids
    )


def effect(before, after, action: dict) -> dict:
    binding = create_binding(before, action)
    op = action["operation"]
    result = {"E": "unknown", "reasons": [], "binding": binding, "selected_before_ids": [], "selected_after_ids": []}
    if op == "room_remove" and binding["destination_ambiguous"]:
        result["reasons"].append("input_destination_instance_not_unique")
        return result
    if binding["no_spatial_change"]:
        result["reasons"].append("request_has_no_spatial_split_support")
        return result
    source_binding = binding["source"]
    if source_binding["status"] != "unique":
        result["E"] = "fail" if source_binding["status"] == "no_source_instance" else "unknown"
        result["reasons"].append(
            "initial_source_instance_missing" if result["E"] == "fail" else "source_instance_binding_ambiguous"
        )
        return result
    before_objects = {v["id"]: v for v in before.document["objects"]}
    after_objects = {v["id"]: v for v in after.document["objects"]}
    source_id = source_binding["id"]
    source = before_objects[source_id]
    g = shape(source["geometry"])
    result["selected_before_ids"] = [source_id]
    frame = tuple(before.document["frame"])

    def match(geometry, label=None):
        return bind_before(after, raster_geometry(geometry, frame), "rooms", label)

    if op == "room_type":
        found = match(g)
        ok = found["status"] == "unique"
        if ok:
            obj = after_objects[found["id"]]
            ok = obj["label"] == action["target_class_id"] and shape(obj["geometry"]).equals(g)
            result["selected_after_ids"] = [found["id"]]
        result.update(E=status(ok), after_correspondence=found)
        if not ok:
            result["reasons"].append("retype_class_or_geometry_not_satisfied")
    elif op == "room_add":
        expected = g.intersection(mask_geometry(action["split_mask"]))
        remainder = g.difference(expected)
        if (
            expected.geom_type != "Polygon"
            or remainder.geom_type != "Polygon"
            or min(expected.area, remainder.area) < 8
        ):
            result["reasons"].append("split_or_remainder_not_one_nonempty_face")
            return result
        new = match(expected, action["target_class_id"])
        old = match(remainder, source["label"])
        preexisting = bind_before(before, raster_geometry(expected, frame), "rooms", action["target_class_id"])
        ok = new["status"] == old["status"] == "unique" and new["id"] != old["id"] and not preexisting["candidates"]
        if ok:
            result["selected_after_ids"] = [new["id"], old["id"]]
            ok = (
                shape(after_objects[new["id"]]["geometry"]).union(shape(after_objects[old["id"]]["geometry"])).equals(g)
            )
        result.update(E=status(ok), new_binding=new, remainder_binding=old)
        if not ok:
            result["reasons"].append("new_room_split_or_retained_remainder_not_realized")
    elif op == "room_remove":
        dest_binding = binding["destination"]
        if not dest_binding or dest_binding["status"] != "unique":
            result["E"] = "fail" if dest_binding and dest_binding["status"] == "no_source_instance" else "unknown"
            result["reasons"].append("initial_destination_instance_missing_or_ambiguous")
            return result
        destination = before_objects[dest_binding["id"]]
        result["selected_before_ids"].append(destination["id"])
        expected = g.union(shape(destination["geometry"]))
        found = match(expected, action["target_class_id"])
        ok = found["status"] == "unique" and source_id != destination["id"]
        if ok:
            ag = shape(after_objects[found["id"]]["geometry"])
            remaining = sum(
                shape(v["geometry"]).intersection(g).area
                for v in after.objects("rooms")
                if v["label"] == source["label"]
            )
            ok = (
                expected.difference(ag).area <= EPS
                and ag.geom_type == "Polygon"
                and ag.difference(expected.buffer(4 * 256 / 224)).area <= EPS
                and remaining <= EPS
            )
            result["selected_after_ids"] = [found["id"]]
        result.update(E=status(ok), merged_binding=found)
        if not ok:
            result["reasons"].append("source_not_merged_into_bound_destination")
    else:
        family = "openings" if op == "opening_delete" else "fixtures"
        overlap = sum(shape(v["geometry"]).intersection(g).area for v in after.objects(family))
        result["selected_before_nonempty_area"] = float(g.area)
        result["after_selected_geometry_overlap"] = float(overlap)
        if op == "opening_delete":
            host = before.document["relationships"]["opening_host"][source_id]
            result["host_before"] = host
            if host["status"] != "unique":
                result["reasons"].append("before_opening_host_missing_or_ambiguous")
                return result
            host_geometry = next(
                shape(v["geometry"]) for v in before.document["wall_hosts"] if v["id"] == host["host_id"]
            )
            aperture = g.intersection(host_geometry)
            material = unary_union([shape(v["geometry"]) for v in after.objects("wall_material")])
            missing = g.difference(material).area
            result.update(
                host_aperture_area=float(aperture.area),
                missing_material_on_full_old_opening=float(missing),
                host_material_restored=status(not aperture.is_empty and aperture.difference(material).area <= EPS),
            )
            ok = overlap <= EPS and missing <= EPS and not aperture.is_empty
        else:
            ok = overlap <= EPS
        result["E"] = status(ok)
        if not ok:
            result["reasons"].append(
                "selected_opening_or_host_material_not_restored"
                if family == "openings"
                else "selected_fixture_geometry_remains"
            )
    return result


def geometry_validity(state) -> dict:
    footprint = shape(state.document["footprint"])
    invalid, leaks, overlaps = [], [], []
    for obj in state.document["objects"]:
        g = shape(obj["geometry"])
        if not g.is_valid or g.is_empty or g.area <= 0:
            invalid.append(obj["id"])
        elif g.difference(footprint).area > EPS:
            leaks.append(obj["id"])
    for family in ("rooms", "openings", "fixtures"):
        objects = state.objects(family)
        for i, a in enumerate(objects):
            for b in objects[i + 1 :]:
                if shape(a["geometry"]).intersection(shape(b["geometry"])).area > EPS:
                    overlaps.append([a["id"], b["id"]])
    nonempty = bool(state.objects("rooms")) and bool(state.objects("wall_material"))
    geometry = status(nonempty and not invalid and not leaks and not overlaps)
    objects = {v["id"]: v for v in state.document["objects"]}
    unresolved_hosts = [
        key for key, value in state.document["relationships"]["opening_host"].items() if value["status"] != "unique"
    ]
    unresolved_owners = []
    for key, value in state.document["relationships"]["fixture_room"].items():
        owner = objects.get(value["room_id"])
        if (
            value["status"] != "unique"
            or owner is None
            or shape(objects[key]["geometry"]).difference(shape(owner["geometry"])).area > EPS
        ):
            unresolved_owners.append(key)
    relationships = "unknown" if unresolved_hosts or unresolved_owners else "pass"
    return {
        "G": tri_and(geometry, relationships),
        "geometry_status": geometry,
        "relationship_status": relationships,
        "nonempty_room_and_wall_material": nonempty,
        "invalid_objects": invalid,
        "footprint_leak_objects": leaks,
        "same_plane_overlaps": overlaps,
        "unresolved_opening_host_ids": unresolved_hosts,
        "unresolved_fixture_room_ids": unresolved_owners,
        "cross_layer_overlap_allowed": True,
        "no_authored_or_network_owner_identity_claim": True,
    }


def vector_preservation(before, after, action: dict, effect_result: dict, trace: dict | None = None) -> dict:
    source = effect_result["binding"]["source"]
    if source["status"] != "unique" or (
        action["operation"] == "room_remove"
        and (
            effect_result["binding"]["destination_ambiguous"]
            or (effect_result["binding"]["destination"] or {}).get("status") != "unique"
        )
    ):
        return {"status": "unknown", "reason": "selected_object_or_destination_not_uniquely_bound"}
    selected_before = set(effect_result["selected_before_ids"])
    selected_after = set(effect_result["selected_after_ids"])
    if trace:
        selected_before.update(trace["selected_before_ids"])
        selected_after.update(trace["selected_after_ids"])
        if not trace["accepted"]:
            selected_after.update(selected_before)
    else:
        selected_after.update(v["id"] for v in after.document["objects"] if v["id"] in selected_before)
    before_objects = [
        v for v in before.document["objects"] if v["id"] not in selected_before and v["family"] != "wall_material"
    ]
    after_objects = [
        v for v in after.document["objects"] if v["id"] not in selected_after and v["family"] != "wall_material"
    ]
    geometry_exact = Counter(geometry_key(v) for v in before_objects) == Counter(
        geometry_key(v) for v in after_objects
    )
    records_geometry_normalized_exact = Counter(
        geometry_normalized_record_key(v) for v in before_objects
    ) == Counter(geometry_normalized_record_key(v) for v in after_objects)
    bytes_exact = Counter(canonical(v) for v in before_objects) == Counter(canonical(v) for v in after_objects)
    bw = unary_union([shape(v["geometry"]) for v in before.objects("wall_material")])
    aw = unary_union([shape(v["geometry"]) for v in after.objects("wall_material")])
    changed = bw.symmetric_difference(aw)
    allowed = (
        shape(next(v["geometry"] for v in before.document["objects"] if v["id"] == source["id"]))
        if action["operation"] == "opening_delete"
        else mask_geometry(np.zeros(tuple(before.document["frame"]), bool))
    )
    wall_outside = changed.difference(allowed).area
    host_rows_equal = Counter(
        geometry_normalized_record_key(v) for v in before.document["wall_hosts"]
    ) == Counter(geometry_normalized_record_key(v) for v in after.document["wall_hosts"])
    before_opening_hosts = before.document.get("relationships", {}).get("opening_host", {})
    after_opening_hosts = after.document.get("relationships", {}).get("opening_host", {})
    opening_hosts_equal = _relationship_rows(
        before_opening_hosts, selected_before
    ) == _relationship_rows(after_opening_hosts, selected_after)
    fixture_relations_equal = True
    if action["operation"] == "furniture_delete":
        br = {k: v for k, v in before.document["relationships"]["fixture_room"].items() if k not in selected_before}
        ar = {k: v for k, v in after.document["relationships"]["fixture_room"].items() if k not in selected_after}
        fixture_relations_equal = br == ar
    return {
        "status": status(
            records_geometry_normalized_exact
            and wall_outside <= EPS
            and host_rows_equal
            and opening_hosts_equal
            and fixture_relations_equal
        ),
        "unselected_geometry_and_class_exact": geometry_exact,
        "unselected_object_records_geometry_normalized_exact": records_geometry_normalized_exact,
        "unselected_object_records_byte_exact": bytes_exact,
        "actual_wall_material_changed_area": float(changed.area),
        "unauthorized_wall_material_changed_area": float(wall_outside),
        "wall_host_witness_rows_exact": host_rows_equal,
        "unselected_opening_host_relationships_exact": opening_hosts_equal,
        "unselected_fixture_room_assignments_exact": fixture_relations_equal,
        "wall_room_incidence_changed": before.document["relationships"]["wall_room_incidence"]
        != after.document["relationships"]["wall_room_incidence"],
    }
