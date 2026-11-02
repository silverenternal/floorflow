"""Verify that the macro-built model loads and runs a legacy checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from floorflow.training.checkpoint import load_checkpoint_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/condition_structured_full_v1_low_lr/merged_objects_preservewall.pt",
    )
    parser.add_argument("--output", default="outputs/reports/p1_model_macro_checkpoint.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, config, _taxonomy, _state = load_checkpoint_model(args.checkpoint, args.device)
    resolution = model.macro_spec.resolution
    x_t = torch.zeros(1, model.target_channels, *resolution, device=args.device)
    time = torch.zeros(1, device=args.device)
    footprint = torch.ones(1, 1, *resolution, device=args.device)
    with torch.no_grad():
        prediction = model(x_t, time, footprint)
    report = {
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "config": config,
        "spec_fingerprint": model.macro_spec.fingerprint(),
        "resolved": model.macro_resolved.as_dict(),
        "load_report": model.macro_load_report,
        "forward_shape": list(prediction.shape),
        "forward_finite": bool(torch.isfinite(prediction).all()),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
