#!/usr/bin/env python3
"""Launch one-GPU Raster2Seq fine-tuning on FloorFlow supervision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/raster2seq_floorflow_finetune_v1.json",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    epochs = int(args.epochs or training["epochs_trend"])
    output_dir = ROOT / "outputs/train/raster2seq_floorflow_finetune_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=14261",
        "scripts/run_raster2seq_training.py",
        "--dataset_name=cubicasa",
        f"--dataset_root={ROOT / config['data_root']}",
        "--semantic_classes=12",
        "--job_name=floorflow_adapt",
        f"--batch_size={training['batch_size']}",
        "--input_channels=3",
        f"--output_dir={output_dir}",
        "--poly2seq",
        f"--seq_len={training['sequence_length']}",
        f"--num_bins={training['coordinate_bins']}",
        "--ckpt_every_epoch=8",
        "--eval_every_epoch=4",
        f"--label_smoothing={training['label_smoothing']}",
        f"--epochs={epochs}",
        "--lr_drop=",
        f"--lr={training['learning_rate']}",
        f"--lr_backbone={training['backbone_learning_rate']}",
        f"--weight_decay={training['weight_decay']}",
        f"--clip_max_norm={training['gradient_clip_norm']}",
        f"--cls_loss_coef={training['class_loss_coefficient']}",
        f"--coords_loss_coef={training['coordinate_loss_coefficient']}",
        f"--room_cls_loss_coef={training['room_class_loss_coefficient']}",
        "--ema4eval",
        "--disable_poly_refine",
        "--dec_attn_concat_src",
        "--per_token_sem_loss",
        "--jointly_train",
        "--converter_version=v3",
        "--use_anchor",
        "--num_workers=8",
        "--device=cuda:0",
    ]
    if args.resume is not None:
        command.append(f"--resume={args.resume.resolve()}")
    else:
        command.append(
            f"--start_from_checkpoint={ROOT / config['base_checkpoint']}"
        )
    environment = os.environ.copy()
    environment["WANDB_MODE"] = "offline"
    manifest = {
        "schema_version": "raster2seq-floorflow-finetune-launch-v1",
        "config": str(args.config),
        "epochs": epochs,
        "command": command,
        "environment": {"WANDB_MODE": "offline"},
    }
    (output_dir / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
        )


__all__ = ["main"]


if __name__ == "__main__":
    main()
