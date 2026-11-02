"""Layered CAD metrics and auditable data-link attribution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LEVEL_METRICS = {
    "L0_legal": (
        "fpeval_validity",
        "room_target_recall",
        "outside_footprint_pixels",
        "conditional_watertight",
    ),
    "L1_topology": (
        "fpeval_watertight",
        "wall_graph_f1",
        "opening_wall_alignment",
        "furniture_room_containment",
        "room_component_gap",
    ),
    "L2_objects": (
        "wall_iou",
        "opening_iou",
        "opening_f1",
        "furniture_iou",
        "furniture_f1",
    ),
    "L3_instances": (
        "instance_geometric_iou",
        "instance_room_f1",
        "instance_type_accuracy",
        "room_group_macro_iou",
    ),
    "L4_semantics": (
        "room_macro_iou",
        "room_dataset_macro_iou",
        "room_dataset_micro_iou",
        "rare_class_recall",
        "boundary_f1",
    ),
    "L5_generation": (
        "style_separation",
        "seed_iou_gap",
        "diversity",
        "runtime_seconds",
    ),
}

LINK_OWNERS = {
    "style": "S2 style-conditioned room graph",
    "room_segmenter": "S2 dynamic room hierarchy",
    "recursion": "S2 recursive partition graph",
    "coarse_layout": "S2 coarse-to-fine room tokens",
    "objects": "S3 dynamic object queries",
    "composition": "S3 layer-aware loss and legalization",
    "topology": "S3 dynamic wall topology",
    "vectorization": "S5 vector-readout audit",
    "identifiability": "diagnostic ceiling only",
}

MODULE_LEVELS = {
    "floorflow.geometry.wall_graph_readout": ("L1", "L3"),
    "floorflow.geometry.opening_readout": ("L1", "L2"),
    "floorflow.geometry.reconciliation": ("L1",),
    "floorflow.geometry.ip_reconcile": ("L1", "L3"),
    "floorflow.geometry.diff_ras": ("L3",),
    "floorflow.distribution.hub": ("L2", "L3", "L4"),
    "floorflow.distribution.pixel_head": ("L4",),
    "floorflow.distribution.element_head": ("L2", "L3"),
    "floorflow.distribution.query_budget": ("L2",),
    "floorflow.heads.polygon_loop": ("L3",),
    "floorflow.heads.dynamic_object_instance": ("L2", "L3"),
    "floorflow.heads.opening_interval_head": ("L1",),
    "floorflow.flow.cfm_backend": ("L5",),
    "floorflow.objectives.token_optimal_transport": ("L5",),
    "floorflow.objectives.furniture_vector_refine": ("L2",),
    "floorflow.evaluation.fpeval_bridge": ("L1", "L2", "L3"),
    "floorflow.evaluation.metric_hierarchy": (
        "L0", "L1", "L2", "L3", "L4", "L5",
    ),
    "floorflow.data.geometry_cache": ("L0", "L2"),
}


@dataclass(frozen=True)
class LinkObservation:
    name: str
    metric: str
    predicted: float
    ceiling: float | None
    higher_is_better: bool
    source: str

    def gap(self) -> float:
        if self.ceiling is None:
            return 1.0 - self.predicted if self.higher_is_better else self.predicted
        if self.higher_is_better:
            return max(0.0, float(self.ceiling) - self.predicted)
        return max(0.0, self.predicted - float(self.ceiling))

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "predicted": self.predicted,
            "ceiling": self.ceiling,
            "gap": self.gap(),
            "higher_is_better": self.higher_is_better,
            "source": self.source,
            "owner": LINK_OWNERS[self.name],
        }


def metric_values(report: Mapping[str, object]) -> dict[str, float]:
    cad = report.get("cad_metrics", {})
    summary = cad.get("summary", {}) if isinstance(cad, Mapping) else {}
    objects = report.get("object_metrics", {}).get("overlay", {})
    room = report.get("room_metrics", {}).get("overlay", {})
    constraints = report.get("constraint_metrics", {})
    values = {}
    for source in (summary, objects, room, constraints):
        if isinstance(source, Mapping):
            for key, value in source.items():
                if isinstance(value, (int, float)):
                    values[str(key)] = float(value)
    aliases = {
        "fpeval_watertight": "unconditional_watertight",
    }
    for source, target in aliases.items():
        if source in values:
            values[target] = values[source]
    return values


def audit_module_ownership(
    registry: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    """Fail when a registered model-path module has no metric level."""

    registry = registry or MODULE_LEVELS
    missing = sorted(name for name, levels in registry.items() if not levels)
    return {
        "complete": not missing,
        "module_count": len(registry),
        "missing_levels": missing,
        "modules": {
            name: list(levels) for name, levels in registry.items()
        },
    }


def build_metric_hierarchy(
    report: Mapping[str, object],
    *,
    style_report: Mapping[str, object] | None = None,
    recursive_report: Mapping[str, object] | None = None,
    decision_audit: Mapping[str, object] | None = None,
    identifiability: Mapping[str, object] | None = None,
    diversity_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    values = metric_values(report)
    if style_report is not None:
        values["style_separation"] = float(
            style_report.get("style_separation", 0.0)
        )
        values["seed_iou_gap"] = float(
            style_report.get("seed_iou_gap", 0.0)
        )
        values["worst_style_room_iou"] = float(
            style_report.get("worst_style_room_iou", 0.0)
        )
        values["all_style_mean_room_iou"] = float(
            style_report.get("all_style_mean_room_iou", 0.0)
        )
        values["runtime_seconds"] = float(
            style_report.get(
                "runtime_seconds_warm_mean",
                style_report.get("runtime_seconds_mean", 0.0),
            )
        )
    if diversity_report is not None:
        styles = diversity_report.get("styles", [])
        if isinstance(styles, Mapping):
            styles = styles.values()
        distances = [
            float(style["distribution"]["mean_within_condition_feature_distance"])
            for style in styles
            if isinstance(style, Mapping)
            and isinstance(style.get("distribution"), Mapping)
            and "mean_within_condition_feature_distance"
            in style["distribution"]
        ]
        if distances:
            values["diversity"] = sum(distances) / len(distances)
    levels = {
        level: {
            name: values.get(name)
            for name in names
        }
        for level, names in LEVEL_METRICS.items()
    }
    missing = {
        level: [name for name, value in metrics.items() if value is None]
        for level, metrics in levels.items()
    }
    room_direct = None
    if recursive_report is not None:
        direct = recursive_report.get("direct", {})
        room_direct = direct.get("room_macro_iou")
    if room_direct is None:
        raw_room = report.get("room_metrics", {}).get("raw", {})
        if isinstance(raw_room, Mapping):
            room_direct = raw_room.get("room_macro_iou")
    recursion_gain = None
    if decision_audit is not None:
        by_action = decision_audit.get("by_action", {})
        recursive_room = by_action.get("recursive", {}).get("room_macro_iou")
        direct_room = by_action.get("direct", {}).get("room_macro_iou")
        if recursive_room is not None and direct_room is not None:
            recursion_gain = float(recursive_room) - float(direct_room)
    exact_ceiling = None
    if identifiability is not None:
        exact_ceiling = identifiability.get(
            "strict_0_95_identifiable_fraction"
        )
    links = {
        "style": LinkObservation(
            "style",
            "worst_style_room_iou",
            float(values.get("worst_style_room_iou", 0.0)),
            1.0,
            True,
            "recursive style/stability report",
        ),
        "room_segmenter": LinkObservation(
            "room_segmenter",
            "direct_segmenter_room_iou",
            float(room_direct or 0.0),
            1.0,
            True,
            "recursive room segmenter report",
        ),
        "recursion": LinkObservation(
            "recursion",
            "recursion_room_gain",
            float(recursion_gain or 0.0),
            0.0,
            False,
            "test decision audit",
        ),
        "coarse_layout": LinkObservation(
            "coarse_layout",
            "room_macro_iou",
            float(values.get("room_macro_iou", 0.0)),
            1.0,
            True,
            "predicted CAD chain",
        ),
        "objects": LinkObservation(
            "objects",
            "furniture_iou",
            float(values.get("furniture_iou", 0.0)),
            1.0,
            True,
            "predicted CAD chain",
        ),
        "composition": LinkObservation(
            "composition",
            "furniture_room_containment",
            float(values.get("furniture_room_containment", 0.0)),
            1.0,
            True,
            "predicted CAD chain",
        ),
        "topology": LinkObservation(
            "topology",
            "unconditional_watertight",
            float(values.get("unconditional_watertight", 0.0)),
            1.0,
            True,
            "predicted CAD chain",
        ),
        "vectorization": LinkObservation(
            "vectorization",
            "fpeval_room_polygon_f1",
            float(values.get("fpeval_room_polygon_f1", 0.0)),
            1.0,
            True,
            "predicted CAD chain",
        ),
        "identifiability": LinkObservation(
            "identifiability",
            "strict_0_95_identifiable_fraction",
            float(exact_ceiling or 0.0),
            1.0,
            True,
            "contour neighbour ceiling",
        ),
    }
    ordered_links = sorted(
        links.items(),
        key=lambda item: item[1].gap(),
        reverse=True,
    )
    ownership = audit_module_ownership()
    complete = (
        ownership["complete"]
        and not any(missing.values())
    )
    return {
        "status": "complete" if complete else "incomplete",
        "metrics": {
            level: {
                name: value
                for name, value in metrics.items()
                if value is not None
            }
            for level, metrics in levels.items()
        },
        "missing": missing,
        "links": {
            name: observation.to_dict()
            for name, observation in links.items()
        },
        "dominant_link": ordered_links[0][0] if ordered_links else None,
        "ranked_links": [
            {"name": name, "gap": observation.gap()}
            for name, observation in ordered_links
        ],
        "module_ownership": ownership,
        "predicted_chain": True,
        "oracle_or_ceiling_values_separated": True,
    }


def hierarchy_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Metric hierarchy and link budget",
        "",
        "| Level | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for level, metrics in report["metrics"].items():
        for name, value in metrics.items():
            lines.append(f"| {level} | {name} | {float(value):.5f} |")
    lines.extend(
        [
            "",
            "| Link | Metric | Gap | Owner |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, row in report["links"].items():
        lines.append(
            f"| {name} | {row['metric']} | {row['gap']:.5f} | "
            f"{row['owner']} |"
        )
    lines.extend(
        [
            "",
            f"Dominant failing link: **{report['dominant_link']}**.",
            (
                "Oracle and identifiability ceilings are kept separate from "
                "predicted-chain metrics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "LEVEL_METRICS",
    "LINK_OWNERS",
    "MODULE_LEVELS",
    "LinkObservation",
    "audit_module_ownership",
    "build_metric_hierarchy",
    "hierarchy_markdown",
    "metric_values",
]
