#!/usr/bin/env python3
"""Create EMA/non-EMA Raster2Seq weight blends for validation-only selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/external_baselines/raster2seq/checkpoint_blends_v1",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def blend_state_dicts(
    primary_state: dict[str, torch.Tensor],
    secondary_state: dict[str, torch.Tensor],
    alpha: float,
    *,
    mask_from_secondary: bool,
) -> dict[str, torch.Tensor]:
    """Blend state dictionaries without corrupting additive attention masks."""

    output = {}
    for key in sorted(set(primary_state) | set(secondary_state)):
        if key not in primary_state:
            output[key] = secondary_state[key].detach().clone()
            continue
        if key not in secondary_state:
            output[key] = primary_state[key].detach().clone()
            continue
        if key == "attention_mask":
            source = secondary_state[key] if mask_from_secondary else primary_state[key]
            output[key] = source.detach().clone()
            continue
        primary_value = primary_state[key]
        secondary_value = secondary_state[key]
        if primary_value.is_floating_point() and secondary_value.is_floating_point():
            output[key] = (
                (1.0 - alpha) * primary_value
                + alpha * secondary_value
            ).detach().clone()
        else:
            output[key] = secondary_value.detach().clone()
    return output


def blend_state(
    model_state: dict[str, torch.Tensor],
    ema_state: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Blend the model and EMA states of one checkpoint."""

    return blend_state_dicts(
        model_state,
        ema_state,
        alpha,
        mask_from_secondary=alpha >= 0.5,
    )


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload or "ema" not in payload:
        raise KeyError("checkpoint must contain model and ema state dictionaries")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in args.alphas:
        state = blend_state(payload["model"], payload["ema"], float(alpha))
        output = {key: value for key, value in payload.items() if key not in {"model", "ema"}}
        output["model"] = state
        output["ema"] = state
        output["blend_alpha_to_ema"] = float(alpha)
        name = f"ema_alpha_{alpha:.2f}.pth"
        path = (args.output_dir / name).resolve()
        torch.save(output, path)
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "alpha": float(alpha),
                "sha256": sha256(path),
            }
        )
    report = {
        "status": "complete",
        "source": str(args.checkpoint.relative_to(ROOT)),
        "source_sha256": sha256(args.checkpoint),
        "selection_split": "val",
        "best_of_n_used": False,
        "blends": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
