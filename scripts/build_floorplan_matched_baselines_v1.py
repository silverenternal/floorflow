#!/usr/bin/env python3
"""Build the matched local baseline table from frozen oracle-control reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDIT = ROOT / "outputs/reports/residential_edit_test503_v1.json"
CONTROL = ROOT / "outputs/reports/residential_editing_aware_baseline_v1.json"
JSON_OUTPUT = ROOT / "outputs/reports/matched_local_editing_baselines_v1.json"
TABLE_OUTPUT = ROOT / "paper/tables/matched_local_editing_baselines_v1.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _operation_row(edit: dict, control: dict, operation: str) -> dict:
    edit_operation = edit["operations"][operation]
    control_operation = control["operations"][operation]
    first_seed = next(iter(edit_operation["per_seed"].values()))
    primary = first_seed["primary_edit"]
    raster2seq = first_seed["raster2seq_local"]
    aware = control_operation["summary"]["control_delta_replace"]
    paired_floorflow = control_operation["paired_floorflow_vs_control_delta"]
    paired_raster2seq = edit_operation["paired_vs_raster2seq_local"]
    return {
        "cases": int(first_seed["cases"]),
        "floorflow_delta_iou": float(primary["delta_iou"]),
        "floorflow_delta_mse": float(primary["delta_mse"]),
        "floorflow_outside_exact": float(primary["outside_exact_ratio"]),
        "raster2seq_delta_iou": float(raster2seq["delta_iou"]),
        "raster2seq_delta_mse": float(raster2seq["delta_mse"]),
        "raster2seq_outside_exact": float(raster2seq["outside_exact_ratio"]),
        "paired_vs_raster2seq": paired_raster2seq,
        "editing_aware_delta_iou": float(aware["delta_iou"]),
        "editing_aware_delta_mse": float(aware["delta_mse"]),
        "paired_vs_editing_aware": paired_floorflow,
    }


def main() -> None:
    edit = _load(EDIT)
    control = _load(CONTROL)
    operations = {
        name: _operation_row(edit, control, name)
        for name in (
            "room_type",
            "room_add",
            "room_remove",
            "opening_delete",
            "furniture_delete",
        )
    }
    payload = {
        "schema_version": "floorplan-matched-local-baselines-v1",
        "status": "partial_model_updating",
        "scope": "CubiCasa test503 oracle-control rows with shared pre/post controls",
        "protocol": {
            "controls": "declared oracle controls",
            "deployable": False,
            "case_unit": "paired pre/post edit",
            "seeds": [19, 20, 21, 22, 23],
        },
        "external_context": {
            "HouseMind-E": {
                "delta_iou": 0.608,
                "delta_mse": 0.0019,
                "status": "published cross-dataset context only",
            }
        },
        "operations": operations,
        "pending": [
            "predicted-control held-out comparison",
            "any stronger executable editing-aware baseline",
        ],
        "sources": {
            str(EDIT.relative_to(ROOT)): _sha256(EDIT),
            str(CONTROL.relative_to(ROOT)): _sha256(CONTROL),
        },
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Matched Local Editing Baselines v1",
        "",
        "Status: partial; oracle-control matched comparison is frozen, while the",
        "predicted-control held-out comparison remains pending.",
        "",
        (
            "| Operation | Cases | FloorFlow IoU | R2S IoU | "
            "Editing-aware IoU | R2S p | Editing-aware Delta-IoU CI |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in operations.items():
        r2s_p = float(row["paired_vs_raster2seq"]["p_value"])
        ci = row["paired_vs_editing_aware"]
        lines.append(
            f"| {name} | {row['cases']} | {row['floorflow_delta_iou']:.6f} "
            f"| {row['raster2seq_delta_iou']:.6f} "
            f"| {row['editing_aware_delta_iou']:.6f} "
            f"| {r2s_p:.3e} "
            f"| [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Raster2Seq is a matched reconstruction wrapper, not a native",
            "editing-aware method. The editing-aware row applies the same declared",
            "after-edit controls through a local control-delta replacement.",
            "HouseMind-E remains cross-dataset protocol context.",
            "",
        ]
    )
    TABLE_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
