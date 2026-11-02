#!/usr/bin/env python3
"""Build a matched checkpoint soup for chip-flow seed stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.checkpoint_soup import weighted_average_state_dicts

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--model-ema-alpha", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoint
    ]
    architectures = {
        payload.get("config", {}).get("architecture") for payload in payloads
    }
    if len(architectures) != 1:
        raise ValueError("checkpoint soup requires one architecture")
    weights = args.weights
    if weights is not None and len(weights) != len(payloads):
        raise ValueError("--weights must match --checkpoint")
    if args.model_ema_alpha is not None:
        if len(payloads) != 1:
            raise ValueError("--model-ema-alpha requires one checkpoint")
        if not 0.0 <= args.model_ema_alpha <= 1.0:
            raise ValueError("--model-ema-alpha must be in [0, 1]")
        payload = payloads[0]
        state = weighted_average_state_dicts(
            [payload["model"], payload["ema"]],
            [1.0 - args.model_ema_alpha, args.model_ema_alpha],
        )
        blended = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "ema"}
        }
        blended["model"] = state
        blended["ema"] = state
        blended["model_ema_alpha"] = args.model_ema_alpha
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blended, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "model_ema_alpha": args.model_ema_alpha,
                },
                indent=2,
            )
        )
        return
    output = {
        key: value
        for key, value in payloads[0].items()
        if key not in {"model", "ema"}
    }
    output["model"] = weighted_average_state_dicts(
        [payload["model"] for payload in payloads],
        weights,
    )
    output["ema"] = weighted_average_state_dicts(
        [payload["ema"] for payload in payloads],
        weights,
    )
    output["soup"] = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "weights": weights or [1.0] * len(payloads),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoints": len(payloads),
                "architecture": architectures.pop(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
