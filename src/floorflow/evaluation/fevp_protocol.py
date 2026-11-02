"""FloorPlan Editing Verification Protocol (FEVP) v1.0 validation helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "fevp-report-v1"
CASE_SCHEMA_VERSION = "fevp-case-row-v1"
PROTOCOL_VERSION = "1.0"
MANDATORY_METRICS = (
    "outside_exact",
    "validity",
    "conditional_watertightness",
    "node_f1",
    "edge_overlap",
)
OPTIONAL_METRICS = ("delta_iou", "ged", "delta_mse")
COVERAGE_STATES = (
    "reported_under_fevp",
    "reported_outside_fevp",
    "not_reported",
    "not_comparable",
)
COMPARABILITY_STATES = ("matched", "context_only", "not_comparable")


class FEVPValidationError(ValueError):
    """Raised when a FEVP document cannot be loaded or validated."""


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FEVPValidationError("FEVP document must be a JSON object")
    return payload


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(
    value: Any,
    name: str,
    errors: list[str],
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> None:
    if not _is_number(value):
        errors.append(f"{name} must be a finite number")
        return
    number = float(value)
    if number < minimum:
        errors.append(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        errors.append(f"{name} must be <= {maximum}")


def _required_string(
    mapping: Mapping,
    key: str,
    name: str,
    errors: list[str],
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{name} must be a non-empty string")
        return None
    return value


def _validate_source(
    metric: Mapping,
    name: str,
    errors: list[str],
) -> None:
    if metric.get("status") in (
        "reported_under_fevp",
        "reported_outside_fevp",
    ):
        _required_string(metric, "source", f"{name}.source", errors)


def _validate_metric(
    mandatory: Mapping,
    name: str,
    errors: list[str],
) -> None:
    metric = mandatory.get(name)
    if not isinstance(metric, Mapping):
        errors.append(f"mandatory.{name} must be an object")
        return
    status = metric.get("status")
    if status not in COVERAGE_STATES:
        errors.append(
            f"mandatory.{name}.status must be one of {COVERAGE_STATES}"
        )
    if status not in ("reported_under_fevp", "reported_outside_fevp"):
        if metric.get("value") is not None:
            errors.append(
                f"mandatory.{name} must not carry a value unless reported"
            )
        return
    _validate_source(metric, f"mandatory.{name}", errors)
    if name == "outside_exact":
        components = (
            "outside_pixel_exact",
            "outside_room_label_exact",
        )
        available = [
            metric.get(component) is not None for component in components
        ]
        if status == "reported_under_fevp":
            for component in components:
                _number(
                    metric.get(component),
                    f"mandatory.{name}.{component}",
                    errors,
                    maximum=1.0,
                )
        elif not any(available):
            errors.append(
                "mandatory.outside_exact reported_outside_fevp requires "
                "at least one outside submetric"
            )
        else:
            for component, present in zip(components, available, strict=True):
                if present:
                    _number(
                        metric.get(component),
                        f"mandatory.{name}.{component}",
                        errors,
                        maximum=1.0,
                    )
    else:
        maximum = None if name == "ged" else 1.0
        _number(
            metric.get("value"),
            f"mandatory.{name}.value",
            errors,
            maximum=maximum,
        )


def _validate_optional(
    optional: Mapping,
    name: str,
    errors: list[str],
) -> None:
    metric = optional.get(name)
    if metric is None:
        return
    if not isinstance(metric, Mapping):
        errors.append(f"optional.{name} must be an object")
        return
    status = metric.get("status")
    if status not in COVERAGE_STATES:
        errors.append(
            f"optional.{name}.status must be one of {COVERAGE_STATES}"
        )
    if status not in ("reported_under_fevp", "reported_outside_fevp"):
        if metric.get("value") is not None:
            errors.append(
                f"optional.{name} must not carry a value unless reported"
            )
        return
    _validate_source(metric, f"optional.{name}", errors)
    _number(
        metric.get("value"),
        f"optional.{name}.value",
        errors,
        maximum=1.0 if name != "ged" else None,
    )


def validate_report(
    report: Mapping,
    *,
    require_complete: bool = False,
) -> list[str]:
    """Validate FEVP report structure and semantic completeness."""

    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION!r}")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(f"protocol_version must equal {PROTOCOL_VERSION!r}")
    for key in ("report_id", "method", "dataset", "task"):
        _required_string(report, key, key, errors)
    comparability = report.get("comparability")
    if not isinstance(comparability, Mapping):
        errors.append("comparability must be an object")
    else:
        status = comparability.get("status")
        if status not in COMPARABILITY_STATES:
            errors.append(
                f"comparability.status must be one of {COMPARABILITY_STATES}"
            )
        if status == "matched":
            _required_string(
                comparability,
                "matched_group",
                "comparability.matched_group",
                errors,
            )
    scorer = report.get("scorer")
    if not isinstance(scorer, Mapping):
        errors.append("scorer must be an object")
    else:
        for key in ("implementation", "version", "code_commit"):
            _required_string(scorer, key, f"scorer.{key}", errors)
    edit_region = report.get("edit_region")
    if not isinstance(edit_region, Mapping):
        errors.append("edit_region must be an object")
    else:
        for key in ("definition", "rasterization_rule", "valid_outside_rule"):
            _required_string(
                edit_region,
                key,
                f"edit_region.{key}",
                errors,
            )
    mandatory = report.get("mandatory")
    if not isinstance(mandatory, Mapping):
        errors.append("mandatory must be an object")
        mandatory = {}
    else:
        unknown = set(mandatory) - set(MANDATORY_METRICS)
        if unknown:
            errors.append(f"unknown mandatory metrics: {sorted(unknown)}")
    for name in MANDATORY_METRICS:
        _validate_metric(mandatory, name, errors)
    optional = report.get("optional", {})
    if not isinstance(optional, Mapping):
        errors.append("optional must be an object")
        optional = {}
    unknown_optional = set(optional) - set(OPTIONAL_METRICS)
    if unknown_optional:
        errors.append(f"unknown optional metrics: {sorted(unknown_optional)}")
    for name in OPTIONAL_METRICS:
        _validate_optional(optional, name, errors)
    cases = report.get("cases")
    if not isinstance(cases, Mapping):
        errors.append("cases must be an object")
    else:
        count = cases.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            errors.append("cases.count must be a non-negative integer")
        for key in ("rows_file", "sha256"):
            if cases.get(key) is not None:
                _required_string(cases, key, f"cases.{key}", errors)
        if (
            report.get("comparability", {}).get("status") == "matched"
            and (not isinstance(count, int) or count <= 0)
        ):
            errors.append("matched reports require non-empty cases")
    source_map = report.get("source_map")
    if not isinstance(source_map, Mapping):
        errors.append("source_map must be an object")
    reported = [
        name
        for name, metric in mandatory.items()
        if isinstance(metric, Mapping)
        and metric.get("status") == "reported_under_fevp"
    ]
    reported.extend(
        f"optional.{name}"
        for name, metric in optional.items()
        if isinstance(metric, Mapping)
        and metric.get("status") == "reported_under_fevp"
    )
    for name in reported:
        metric_name = name.removeprefix("optional.")
        if metric_name not in source_map:
            errors.append(f"source_map is missing {metric_name}")
    if require_complete:
        missing = [
            name
            for name in MANDATORY_METRICS
            if mandatory.get(name, {}).get("status")
            != "reported_under_fevp"
        ]
        if missing:
            errors.append(
                f"report is incomplete under FEVP v1.0: missing {missing}"
            )
    return errors


def validate_case_row(row: Mapping) -> list[str]:
    """Validate one complete FEVP case row."""

    errors: list[str] = []
    if row.get("schema_version") != CASE_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {CASE_SCHEMA_VERSION!r}")
    for key in ("case_id", "operation", "source_hash"):
        _required_string(row, key, key, errors)
    _number(row.get("outside_pixel_exact"), "outside_pixel_exact", errors, maximum=1.0)
    _number(
        row.get("outside_room_label_exact"),
        "outside_room_label_exact",
        errors,
        maximum=1.0,
    )
    _number(row.get("validity"), "validity", errors, maximum=1.0)
    _number(row.get("node_f1"), "node_f1", errors, maximum=1.0)
    _number(row.get("edge_overlap"), "edge_overlap", errors, maximum=1.0)
    target_watertight = row.get("target_watertight")
    if not isinstance(target_watertight, bool):
        errors.append("target_watertight must be boolean")
    watertight = row.get("conditional_watertightness")
    if target_watertight is True:
        _number(
            watertight,
            "conditional_watertightness",
            errors,
            maximum=1.0,
        )
    elif target_watertight is False and watertight is not None:
        errors.append(
            "conditional_watertightness must be null for non-watertight targets"
        )
    optional = row.get("optional", {})
    if not isinstance(optional, Mapping):
        errors.append("optional must be an object")
    else:
        unknown = set(optional) - set(OPTIONAL_METRICS)
        if unknown:
            errors.append(f"unknown optional metrics: {sorted(unknown)}")
        for name, value in optional.items():
            if value is not None:
                _number(
                    value,
                    f"optional.{name}",
                    errors,
                    maximum=1.0 if name != "ged" else None,
                )
    return errors


def assert_valid_report(
    report: Mapping,
    *,
    require_complete: bool = False,
) -> None:
    errors = validate_report(report, require_complete=require_complete)
    if errors:
        raise FEVPValidationError("\n".join(errors))


def aggregate_case_rows(rows: Sequence[Mapping]) -> dict:
    """Reference aggregation for complete FEVP case rows."""

    if not rows:
        raise FEVPValidationError("at least one case row is required")
    errors = {
        str(row.get("case_id", index)): validate_case_row(row)
        for index, row in enumerate(rows)
    }
    invalid = {key: value for key, value in errors.items() if value}
    if invalid:
        raise FEVPValidationError(json.dumps(invalid, indent=2))
    watertight = [
        float(row["conditional_watertightness"])
        for row in rows
        if row["target_watertight"]
    ]
    return {
        "outside_pixel_exact": _mean(rows, "outside_pixel_exact"),
        "outside_room_label_exact": _mean(rows, "outside_room_label_exact"),
        "validity": _mean(rows, "validity"),
        "conditional_watertightness": (
            sum(watertight) / len(watertight) if watertight else None
        ),
        "node_f1": _mean(rows, "node_f1"),
        "edge_overlap": _mean(rows, "edge_overlap"),
        "optional": {
            name: _optional_mean(rows, name)
            for name in OPTIONAL_METRICS
        },
        "cases": len(rows),
        "watertight_eligible_cases": len(watertight),
    }


def _mean(rows: Sequence[Mapping], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _optional_mean(rows: Sequence[Mapping], key: str) -> float | None:
    values = [
        float(row.get("optional", {})[key])
        for row in rows
        if row.get("optional", {}).get(key) is not None
    ]
    return sum(values) / len(values) if values else None


__all__ = [
    "CASE_SCHEMA_VERSION",
    "COMPARABILITY_STATES",
    "COVERAGE_STATES",
    "MANDATORY_METRICS",
    "OPTIONAL_METRICS",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "FEVPValidationError",
    "aggregate_case_rows",
    "assert_valid_report",
    "load_json",
    "validate_case_row",
    "validate_report",
]
