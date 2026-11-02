#!/usr/bin/env python3
"""Audit semantic success for existing predicted-control typed edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from floorflow.data.control_cache import read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.predicted_operation_success import (
    operation_success,
    room_control_request,
)
from floorflow.inference.predicted_control_edit import project_predicted_instruction
from floorflow.inference.room_repair import (
    close_room_partition,
    merge_room_fragments,
)

ROOT = Path(__file__).resolve().parents[1]
ROOM_OPERATIONS = {"room_type", "room_add", "room_remove"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "outputs/reports/residential_instruction_robustness_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_operation_success_val191_v1.json",
    )
    parser.add_argument(
        "--predicted-rasters",
        type=Path,
        default=ROOT
        / "outputs/eval/fp_completion_predicted_refined_val491_v1/rasters",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=ROOT
        / "outputs/control_contracts/predicted_instruction_val191_v1",
    )
    parser.add_argument("--limit-per-category", type=int, default=100)
    return parser.parse_args()


def _before_state(
    path: Path,
    taxonomy: FloorplanTaxonomy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        footprint = payload["footprint"] > 0
        visible = payload["topology_prediction"].astype(np.int64)
        logits = gaussian_filter(
            payload["room_logits"].astype(np.float32),
            sigma=(0.0, 10.0, 10.0),
        )
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    labels = room_ids[logits.argmax(axis=0)]
    labels = np.where(footprint, labels, 0).astype(np.int64)
    labels = merge_room_fragments(labels, taxonomy, min_area=32)
    labels = close_room_partition(labels, footprint, taxonomy)
    room_mask = np.isin(visible, taxonomy.room_ids)
    visible[room_mask] = labels[room_mask]
    return labels, visible, footprint


def _source_control_path(row: dict, name: str, control_root: Path) -> Path:
    manifest_path = (
        Path(row.get("control_root", control_root))
        / row["operation"]
        / row.get("split", "val")
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in manifest["rows"] if item["name"].replace("/", "__") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one source row for {name}, found {len(matches)}")
    path = Path(matches[0]["source"])
    return path if path.is_absolute() else ROOT / path


def _audit_row(
    row: dict,
    taxonomy: FloorplanTaxonomy,
    predicted_rasters: Path,
    control_root: Path,
) -> dict:
    if row["abstained"]:
        return {
            **row,
            "success": False,
            "success_applicable": False,
            "failure_reason": "abstained",
        }
    name = str(row["name"])
    before, visible, footprint = _before_state(
        Path(row.get("predicted_rasters", predicted_rasters)) / f"{name}.npz",
        taxonomy,
    )
    edited_path = (
        Path(row.get("control_root", control_root))
        / row["operation"]
        / row.get("split", "val")
        / f"{name}.npz"
    )
    edited_controls = read_control_cache(edited_path)
    operation = str(row["operation"])
    request = None
    if operation in ROOM_OPERATIONS:
        source_controls = read_control_cache(
            _source_control_path(row, name, control_root)
        )
        try:
            request = room_control_request(
                source_controls,
                edited_controls,
                tuple(row["edit_region"]),
                taxonomy,
            )
        except ValueError as error:
            return {
                **row,
                "success": False,
                "success_applicable": True,
                "failure_reason": f"control_request_invalid:{error}",
                "changed_pixels": 0,
                "source_pixels_before": 0,
                "source_removed_fraction": 0.0,
                "target_share_of_changed": 0.0,
                "target_pixels_after": 0,
            }
    before_source = visible if operation in {"opening_delete", "furniture_delete"} else before
    after, region = project_predicted_instruction(
        torch.from_numpy(before_source.astype(np.int64)),
        edited_controls,
        tuple(row["edit_region"]),
        tuple(row["control_size"]),
        taxonomy,
        operation=operation,
    )
    after = after.numpy()
    if operation in ROOM_OPERATIONS:
        after = merge_room_fragments(after, taxonomy, min_area=32)
        after = close_room_partition(after, footprint, taxonomy)
    result = operation_success(
        before_source,
        after,
        region.numpy(),
        taxonomy,
        operation,
        request=request,
    )
    return {
        **row,
        "success": result.success,
        "success_applicable": True,
        "failure_reason": result.reason,
        "changed_pixels": result.changed_pixels,
        "source_pixels_before": result.source_pixels_before,
        "source_removed_fraction": result.source_removed_fraction,
        "target_share_of_changed": result.target_share_of_changed,
        "target_pixels_after": result.target_pixels_after,
    }


def _summary(rows: list[dict]) -> dict:
    result = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        applicable = [row for row in selected if row["success_applicable"]]
        result[category] = {
            "prompts": len(selected),
            "supported_prompts": len(applicable),
            "abstained_prompts": len(selected) - len(applicable),
            "successful_prompts": sum(bool(row["success"]) for row in applicable),
            "operation_success_rate": (
                float(np.mean([row["success"] for row in applicable]))
                if applicable
                else None
            ),
            "inside_region_changed_rate": (
                float(np.mean([row["changed_pixels"] > 0 for row in applicable]))
                if applicable
                else None
            ),
            "outside_exact_rate": float(
                np.mean([row["outside_pixel_exact"] == 1.0 for row in selected])
            ),
            "validity_rate": float(np.mean([row["validity"] for row in selected])),
        }
    return result


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    payload = json.loads(args.rows.read_text(encoding="utf-8"))
    rows = payload["prompts"]
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("test rows are forbidden before the one-shot freeze")
    selected = []
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        if not row["executed"]:
            selected.append(
                _audit_row(
                    row,
                    taxonomy,
                    args.predicted_rasters,
                    args.control_root,
                )
            )
            continue
        if counts.get(category, 0) >= args.limit_per_category:
            continue
        selected.append(
            _audit_row(
                row,
                taxonomy,
                args.predicted_rasters,
                args.control_root,
            )
        )
        counts[category] = counts.get(category, 0) + 1
    summary = _summary(selected)
    report = {
        "schema_version": "predicted-operation-success-v1",
        "status": "preliminary_validation_model_updating",
        "split": "val191",
        "test_opened": False,
        "model_frozen": False,
        "source_rows": str(args.rows.resolve().relative_to(ROOT)),
        "contract": "configs/predicted_operation_success_contract_v1.json",
        "summary": summary,
        "checks": {
            "all_supported_categories_present": all(
                item["supported_prompts"] > 0 for item in summary.values()
            ),
            "all_unsupported_categories_abstain": all(
                item["abstained_prompts"] == item["prompts"]
                for name, item in summary.items()
                if "unsupported" in name or "ambiguous" in name or "conflict" in name
            ),
            "outside_exact_all_categories": all(
                item["outside_exact_rate"] == 1.0 for item in summary.values()
            ),
            "validity_all_categories": all(
                item["validity_rate"] == 1.0 for item in summary.values()
            ),
        },
        "boundary": (
            "Preliminary audit over existing validation outputs. The predicted "
            "model and this result are not frozen for submission."
        ),
        "rows": selected,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
