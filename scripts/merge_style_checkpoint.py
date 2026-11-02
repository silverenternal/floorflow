#!/usr/bin/env python3
"""Attach trained style adapters to a newer style-free distribution checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from floorflow.distribution.checkpoint import (
    restore_distribution_model,
    save_distribution_model,
)


def compatible(module, state):
    current = module.state_dict()
    return {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--style-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-styles", type=int, default=8)
    parser.add_argument("--style-rank", type=int, default=8)
    parser.add_argument("--zero-style-id", type=int, default=0)
    args = parser.parse_args()
    model, taxonomy, _ = restore_distribution_model(
        args.base_checkpoint,
        "cpu",
        num_styles=args.num_styles,
        style_rank=args.style_rank,
    )
    style_payload = torch.load(
        args.style_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    for module, state_name in (
        (model.hub.pixel, "pixel_head"),
        (model.hub.element, "element_head"),
        (model.velocity_style_adapter, "velocity_style_adapter"),
    ):
        if module is None or style_payload.get(state_name) is None:
            continue
        module.load_state_dict(
            compatible(module, style_payload[state_name]),
            strict=False,
        )
    if args.zero_style_id is not None:
        style_id = int(args.zero_style_id)
        for module in model.modules():
            if isinstance(module, nn.Embedding) and style_id < module.num_embeddings:
                with torch.no_grad():
                    module.weight[style_id].zero_()
    save_distribution_model(model, args.output, taxonomy=taxonomy)


if __name__ == "__main__":
    main()
