"""Reference and aggregate validity checks for serialized CAD08 state."""

from __future__ import annotations

import json
import math

from shapely.geometry import LineString, shape

from floorflow.evaluation.cad08_state_metrics_v1 import (
    EPS,
    FAMILIES,
    TABLES,
    TOP_FIELDS,
    _issue,
    digest,
    state_key,
    tri_and,
)


def reference_validity(doc: dict, predecessor: dict | None = None) -> dict:
    """Check actual nodes, families, all foreign keys, graph edges and support."""
    issues, references = [], {}
    objects, hosts, geometries = {}, {}, {}
    for row in doc.get("objects", []):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in objects:
            issues.append(_issue("fail", "invalid_or_duplicate_object_ID", str(identity)))
            continue
        objects[identity] = row
        extra_fields = set(row) - {"id", "family", "label", "geometry", "identity_source"}
        if extra_fields:
            issues.append(_issue("unknown", "unchecked_extra_object_fields", identity, fields=sorted(extra_fields)))
        if row.get("family") not in FAMILIES:
            issues.append(_issue("fail", "wrong_object_family", identity))
        try:
            geometry = shape(row["geometry"])
            if not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
                raise ValueError("invalid or empty geometry")
            geometries[identity] = geometry
        except (KeyError, TypeError, ValueError):
            issues.append(_issue("fail", "invalid_object_geometry", identity))
    for row in doc.get("wall_hosts", []):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in hosts or identity in objects:
            issues.append(_issue("fail", "invalid_or_duplicate_host_ID", str(identity)))
            continue
        hosts[identity] = row
        if set(row) - {"id", "geometry", "start", "end", "thickness", "source"}:
            issues.append(_issue("unknown", "unchecked_extra_host_fields", identity))
        try:
            geometry = shape(row["geometry"])
            if (
                not geometry.is_valid
                or geometry.is_empty
                or geometry.area <= 0
                or not math.isfinite(row["thickness"])
                or row["thickness"] <= 0
            ):
                raise ValueError("invalid host")
            geometries[identity] = geometry
            expected = (
                LineString([row["start"], row["end"]])
                .buffer(row["thickness"] / 2, cap_style=2)
                .intersection(shape(doc["footprint"]))
            )
            if expected.is_empty or geometry.symmetric_difference(expected).area > EPS:
                issues.append(_issue("fail", "host_geometry_inconsistent_with_segment_fields", identity))
        except (KeyError, TypeError, ValueError):
            issues.append(_issue("fail", "invalid_host_geometry", identity))
    if doc.get("schema") != "floorflow-layered-cad08-v1":
        issues.append(_issue("unknown", "unsupported_state_schema", "document"))
    extra = sorted(set(doc) - TOP_FIELDS)
    if extra:
        issues.append(_issue("unknown", "unchecked_extra_state_fields", "document", fields=extra))
    relations = doc.get("relationships", {})
    if not isinstance(relations, dict):
        relations = {}
        issues.append(_issue("fail", "relationships_not_a_map", "document"))
    if set(relations) - set(TABLES):
        issues.append(
            _issue("unknown", "unchecked_relationship_tables", "document", fields=sorted(set(relations) - set(TABLES)))
        )
    rooms = {i: v for i, v in objects.items() if v.get("family") == "rooms" and i in geometries}

    def pointer(identity, expected, source, table):
        if not isinstance(identity, str):
            issues.append(_issue("fail", "missing_or_malformed_claimed_reference", source, identity, table=table))
            return False
        pool = hosts if expected == "wall_host" else objects
        if identity not in pool:
            code = (
                "wrong_reference_family"
                if identity in (objects if expected == "wall_host" else hosts)
                else "dangling_reference"
            )
            issues.append(_issue("fail", code, source, identity, table=table, expected_family=expected))
            return False
        if expected != "wall_host" and pool[identity].get("family") != expected:
            issues.append(
                _issue("fail", "wrong_reference_family", source, identity, table=table, expected_family=expected)
            )
            return False
        return identity in geometries

    for table, source_family, target_family in (
        ("opening_host", "openings", "wall_host"),
        ("fixture_room", "fixtures", "rooms"),
    ):
        rows = relations.get(table)
        if not isinstance(rows, dict):
            issues.append(_issue("unknown", "missing_relationship_table", table))
            rows = {}
        sources = {i for i, v in objects.items() if v.get("family") == source_family}
        for source in sorted(set(rows) - sources):
            issues.append(_issue("fail", "orphan_or_wrong_family_relation_source", source, table=table))
        for source in sorted(sources):
            record = rows.get(source)
            target_key = "host_id" if table == "opening_host" else "room_id"
            if not isinstance(record, dict):
                issues.append(_issue("unknown", "missing_relationship_record", source, table=table))
                references[table + "/" + source] = {
                    "status": "unknown",
                    "source": source,
                    "target": None,
                    "table": table,
                }
                continue
            statuses = []
            if set(record) - {"status", target_key, "candidates", "source"}:
                statuses.append("unknown")
                issues.append(_issue("unknown", "unchecked_extra_reference_fields", source, table=table))
            geometry = geometries.get(source)
            possible = {}
            if geometry is not None:
                pool = hosts if table == "opening_host" else rooms
                for target in pool:
                    tg = geometries.get(target)
                    if tg is None:
                        continue
                    overlap = geometry.intersection(tg).area
                    value = overlap / geometry.area if table == "opening_host" else overlap
                    if value >= 0.5 if table == "opening_host" else value > EPS:
                        possible[target] = float(value)
            candidates = record.get("candidates")
            listed = {}
            if not isinstance(candidates, list):
                statuses.append("unknown")
                issues.append(_issue("unknown", "missing_reference_candidates", source, table=table))
            else:
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        statuses.append("fail")
                        issues.append(_issue("fail", "malformed_reference_candidate", source, table=table))
                        continue
                    target = candidate.get("host_id" if table == "opening_host" else "id")
                    if not pointer(target, target_family, source, table):
                        statuses.append("fail")
                        continue
                    if target in listed:
                        statuses.append("fail")
                        issues.append(_issue("fail", "duplicate_reference_candidate", source, target, table=table))
                    value = candidate.get("covered_fraction" if table == "opening_host" else "overlap_area")
                    listed[target] = value
                    if (
                        target not in possible
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or abs(value - possible[target]) > EPS
                    ):
                        statuses.append("fail")
                        issues.append(
                            _issue(
                                "fail",
                                "candidate_not_supported_by_geometry",
                                source,
                                target,
                                table=table,
                                stored=value,
                                computed=possible.get(target),
                            )
                        )
                if set(listed) != set(possible):
                    statuses.append("fail")
                    issues.append(
                        _issue(
                            "fail",
                            "candidate_set_incomplete_or_stale",
                            source,
                            table=table,
                            missing=sorted(set(possible) - set(listed)),
                            extra=sorted(set(listed) - set(possible)),
                        )
                    )
            claim = record.get("status")
            target = record.get(target_key)
            supported = None
            if target is not None or claim == "unique":
                if not pointer(target, target_family, source, table):
                    statuses.append("fail")
                elif geometry is not None:
                    tg = geometries[target]
                    fraction = float(geometry.intersection(tg).area / geometry.area)
                    supported = fraction
                    if table == "opening_host" and fraction < 0.5 or table == "fixture_room" and fraction <= EPS:
                        statuses.append("fail")
                        issues.append(
                            _issue(
                                "fail",
                                "chosen_reference_has_no_required_geometric_support",
                                source,
                                target,
                                table=table,
                                fraction=fraction,
                            )
                        )
                    elif table == "fixture_room" and geometry.difference(tg).area > EPS:
                        # Original cached assignment is a best-overlap diagnostic.
                        # Partial containment is unresolved, never a valid owner claim.
                        statuses.append("unknown")
                        issues.append(
                            _issue(
                                "unknown",
                                "fixture_not_fully_contained_in_claimed_room",
                                source,
                                target,
                                fraction=fraction,
                            )
                        )
            if claim == "unique":
                if table == "opening_host":
                    unique = len(possible) == 1 and target in possible
                else:
                    best = max(possible.values(), default=0)
                    winners = [k for k, v in possible.items() if abs(v - best) <= EPS]
                    unique = len(winners) == 1 and target in winners
                if not unique:
                    statuses.append("fail")
                    issues.append(
                        _issue("fail", "unique_status_not_supported_by_actual_candidates", source, target, table=table)
                    )
            elif claim in ("missing", "ambiguous"):
                statuses.append("unknown")
                issues.append(
                    _issue(
                        "unknown",
                        "relationship_not_uniquely_annotated",
                        source,
                        target,
                        table=table,
                        claimed_status=claim,
                    )
                )
            else:
                statuses.append("fail")
                issues.append(_issue("fail", "invalid_relationship_status", source, target, table=table, value=claim))
            references[table + "/" + source] = {
                "status": tri_and(*statuses),
                "source": source,
                "target": target,
                "table": table,
                "geometric_support_fraction": supported,
                "record_sha256": digest(record),
                "source_geometry_sha256": digest(objects[source]["geometry"]),
                "target_geometry_sha256": digest((hosts if table == "opening_host" else objects)[target]["geometry"])
                if isinstance(target, str) and target in (hosts if table == "opening_host" else objects)
                else None,
            }

    for table in ("room_adjacency", "wall_room_incidence"):
        records = relations.get(table)
        if not isinstance(records, list):
            issues.append(_issue("unknown", "missing_relationship_table", table))
            continue
        expected = set()
        ids = sorted(rooms)
        if table == "room_adjacency":
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    if geometries[a].boundary.intersection(geometries[b].boundary).length > EPS:
                        expected.add((a, b))
        else:
            for h in hosts:
                if h not in geometries:
                    continue
                for r in rooms:
                    if geometries[h].intersection(geometries[r]).area > EPS:
                        expected.add((h, r))
        observed = []
        for edge in records:
            if not isinstance(edge, list) or len(edge) != 2:
                issues.append(_issue("fail", "malformed_graph_edge", table))
                continue
            a, b = edge
            ok = pointer(a, "rooms" if table == "room_adjacency" else "wall_host", str(a), table)
            ok = pointer(b, "rooms", str(a), table) and ok
            if not ok:
                continue
            key = tuple(sorted(edge)) if table == "room_adjacency" else tuple(edge)
            observed.append(key)
            if key not in expected:
                issues.append(_issue("fail", "graph_edge_has_no_geometric_support", str(a), b, table=table))
        if len(observed) != len(set(observed)):
            issues.append(_issue("fail", "duplicate_graph_edge", table))
        if expected - set(observed):
            issues.append(
                _issue(
                    "fail",
                    "missing_supported_graph_edges",
                    table,
                    edges=[list(v) for v in sorted(expected - set(observed))],
                )
            )
    available = set(objects) | ({v["id"] for v in predecessor["objects"]} if predecessor else set())
    node_families = {v["id"]: v.get("family") for v in (predecessor or {}).get("objects", [])}
    node_families.update({k: v.get("family") for k, v in objects.items()})
    for event in doc.get("lineage", []):
        if not isinstance(event, dict):
            issues.append(_issue("fail", "malformed_lineage_event", "lineage"))
            continue
        operation = event.get("operation", "")
        expected_family = (
            "rooms"
            if operation in ("room_type", "room_add", "room_remove")
            else "openings"
            if operation == "opening_delete"
            else "fixtures"
            if operation == "furniture_delete"
            else None
        )
        if expected_family is None:
            issues.append(_issue("fail", "invalid_lineage_operation", "lineage"))
        for parent in event.get("parents", []):
            if parent not in available:
                issues.append(_issue("fail" if predecessor else "unknown", "unresolved_lineage_parent", str(parent)))
            elif expected_family and node_families.get(parent) != expected_family:
                issues.append(
                    _issue("fail", "wrong_lineage_parent_family", str(parent), expected_family=expected_family)
                )
        for child in event.get("children", []):
            if child not in objects:
                issues.append(_issue("fail", "dangling_lineage_child", str(child)))
            elif expected_family and objects[child].get("family") != expected_family:
                issues.append(_issue("fail", "wrong_lineage_child_family", str(child), expected_family=expected_family))
        available.update(event.get("children", []))
    return {
        "status": tri_and(*(v["status"] for v in issues)),
        "issues": issues,
        "references": references,
        "counts": {"objects": len(objects), "hosts": len(hosts), "checked_reference_records": len(references)},
        "geometry_support_is_not_model_owner_prediction": True,
    }

def validity_state(doc: dict, old_g: dict, predecessor: dict | None = None) -> dict:
    refs = reference_validity(doc, predecessor)
    issues = list(refs["issues"])
    by_id = {v["id"]: v for v in doc.get("objects", [])}
    footprint = shape(doc["footprint"])
    for identity in old_g.get("invalid_objects", []):
        issues.append(_issue("fail", "invalid_or_empty_geometry", identity))
    for identity in old_g.get("footprint_leak_objects", []):
        leak = shape(by_id[identity]["geometry"]).difference(footprint)
        issues.append(
            _issue(
                "fail",
                "footprint_leak",
                identity,
                violation_geometry=json.loads(json.dumps(leak.__geo_interface__)),
                area=float(leak.area),
            )
        )
    for a, b in old_g.get("same_plane_overlaps", []):
        overlap = shape(by_id[a]["geometry"]).intersection(shape(by_id[b]["geometry"]))
        issues.append(
            _issue(
                "fail",
                "same_plane_overlap",
                a,
                b,
                violation_geometry=json.loads(json.dumps(overlap.__geo_interface__)),
                area=float(overlap.area),
            )
        )
    if not old_g.get("nonempty_room_and_wall_material", False):
        issues.append(_issue("fail", "empty_room_or_wall_layer", "document"))
    # Keep every old fail/unknown; strengthening never promotes the old G.
    return {
        "G_state": tri_and(old_g["G"], refs["status"]),
        "G_legacy_recomputed": old_g,
        "reference_validity": refs,
        "issues": issues,
        "key": state_key(doc, predecessor),
    }

__all__ = ["reference_validity", "validity_state"]
