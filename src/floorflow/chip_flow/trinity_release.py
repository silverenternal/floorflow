"""Load a pinned Trinity release for inference without training dependencies."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file

TRINITY_SOURCE = Path(__file__).resolve().parents[3] / "third_party/Trinity/src"
if str(TRINITY_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRINITY_SOURCE))

from trinity import FRAMING, GRAPH_PE, LATENT_PARAM, build
from trinity.conditioning import CondOptions
from trinity.models.backbone import SetTransformerDenoiser, X0View
from trinity.models.presets import DenoiserArchConfig


@dataclass(frozen=True)
class ReleaseComponents:
    backbone: object
    param: object
    framing: object
    graph_pe: object
    cond_options: CondOptions
    config: dict


def load_release(path: Path, device: str) -> ReleaseComponents:
    """Build a release from config and safetensors, bypassing trainer imports."""

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    arch = DenoiserArchConfig(**config["arch"])
    backbone = SetTransformerDenoiser(arch)
    backbone.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    backbone = backbone.to(device).eval()
    framing = build(config["framing"], FRAMING)
    if arch.output_kind != "x0":
        backbone = X0View(backbone, framing)
    return ReleaseComponents(
        backbone=backbone,
        param=build(config["latent_param"], LATENT_PARAM),
        framing=framing,
        graph_pe=build(config["graph_pe"], GRAPH_PE),
        cond_options=CondOptions(**config["cond_options"]),
        config=config,
    )


__all__ = ["ReleaseComponents", "load_release"]
