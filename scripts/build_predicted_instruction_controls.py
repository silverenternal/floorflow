"""Apply typed local instructions to real FloorFlow predicted controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.control_cache import read_control_cache
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS
from floorflow.data.taxonomy import FloorplanTaxonomy

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_BY_OPERATION = {
    "room_add": "occupied_room_insertion",
    "room_remove": "ambiguous_global_scope",
    "room_type": "conflicting_instruction",
    "opening_delete": "opening_pressure",
    "furniture_delete": "unsupported_or_low_salience",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/control_contracts/predicted_instruction_val191_v1",
    )
    parser.add_argument("--seed", type=int, default=26000)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, controls: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        room_logits=controls["room_logits"].float().cpu().numpy().astype(np.float16),
        program=controls["program"].bool().cpu().numpy(),
        structure=controls["structure"].float().cpu().numpy().astype(np.float16),
        furniture=controls["furniture"].float().cpu().numpy().astype(np.float16),
        support=controls["support"].float().cpu().numpy().astype(np.float16),
    )


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    predicted_paths = sorted(
        (args.predicted_root / args.split).glob("*.npz")
    )
    if args.limit is not None:
        predicted_paths = predicted_paths[: int(args.limit)]
    summaries = {}
    for operation_index, (operation, edit_function) in enumerate(
        LOCAL_EDIT_OPERATIONS.items()
    ):
        output_dir = args.output_root / operation / args.split
        rows = []
        for index, source in enumerate(predicted_paths):
            controls = read_control_cache(source)
            seed = args.seed + index * 11 + operation_index
            edit = edit_function(controls, taxonomy, seed=seed)
            row = {
                "name": source.stem.replace("__", "/"),
                "operation": operation,
                "category": CATEGORY_BY_OPERATION[operation],
                "seed": seed,
                "source": (
                    str(source.relative_to(ROOT))
                    if source.is_absolute()
                    else str(source)
                ),
                "source_sha256": _sha256(source),
                "predicted_control_used": True,
                "target_used_at_inference": False,
                "edited": edit is not None,
            }
            if edit is not None:
                target = output_dir / source.name
                _save(target, edit.controls)
                row.update(
                    {
                        "file": (
                            str(target.relative_to(ROOT))
                            if target.is_absolute()
                            else str(target)
                        ),
                        "edit_region": list(edit.bbox),
                        "control_size": list(edit.control_size),
                        "changed_pixels": edit.changed_pixels,
                    }
                )
            rows.append(row)
        manifest = {
            "schema_version": "predicted-instruction-controls-v1",
            "scope": "residential_floorplan_only",
            "split": args.split,
            "operation": operation,
            "category": CATEGORY_BY_OPERATION[operation],
            "predicted_control_used": True,
            "target_used_at_inference": False,
            "counts": {
                "rows": len(rows),
                "edited": sum(bool(row["edited"]) for row in rows),
            },
            "rows": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[operation] = manifest["counts"]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
