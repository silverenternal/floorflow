"""Materialize validation rasters from the promoted predicted-control refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from floorflow.data.control_cache import control_cache_path, read_control_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.inference.predicted_refiner_runtime import (
    render_predicted_refiner,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/train/residential_predicted_refiner_v1/best.pt"
OUTPUT = ROOT / "outputs/eval/fp_completion_predicted_refined_val491_v1"


class _TargetData(Dataset):
    def __init__(
        self,
        root: Path,
        cache_root: Path,
        split: str,
        limit: int | None,
    ) -> None:
        self.root = root
        self.split = split
        self.names = sorted(
            path.stem for path in (root / split).glob("*.npz")
        )
        if limit is not None:
            self.names = self.names[: int(limit)]
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        category, case_id, floor = name.split("__", maxsplit=2)
        path = (
            self.cache_root
            / self.split
            / category
            / case_id
            / f"{floor.replace('-', '_')}.npz"
        )
        with np.load(path) as payload:
            return (
                torch.from_numpy(payload["target"].astype(np.int64)),
                torch.from_numpy(payload["footprint"].astype(bool)),
                name,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=int(config["room_base_channels"]),
        structure_base_channels=int(config["structure_base_channels"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = _TargetData(
        args.predicted_root,
        args.target_root,
        args.split,
        args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    output_dir = args.output_root / "rasters"
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, footprint, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = [
            read_control_cache(
                control_cache_path(args.predicted_root, args.split, name)
            )
            for name in names
        ]
        predicted = {
            key: torch.stack([row[key] for row in controls]).to(device)
            for key in ("room_logits", "structure", "furniture", "support", "program")
        }
        visible, room_labels, room_logits = render_predicted_refiner(
            model,
            predicted,
            footprint,
            taxonomy,
            room_size=int(config["room_size"]),
            structure_size=int(config["structure_size"]),
        )
        for index, name in enumerate(names):
            np.savez_compressed(
                output_dir / f"{name}.npz",
                prediction=visible[index].cpu().numpy(),
                topology_prediction=visible[index].cpu().numpy(),
                target=target[index].cpu().numpy(),
                footprint=footprint[index].cpu().numpy(),
                room_labels=room_labels[index].cpu().numpy(),
                room_logits=room_logits[index].float().cpu().numpy().astype(
                    np.float16
                ),
                room_support=room_labels[index].ne(0).cpu().numpy(),
            )
    manifest = {
        "schema_version": "predicted-refiner-rasters-v1",
        "split": args.split,
        "cases": len(dataset),
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "predicted_control_used": True,
        "target_used_at_inference": False,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
