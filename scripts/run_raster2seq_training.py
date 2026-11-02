"""Launch Raster2Seq training without its interactive W&B credential prompt."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RASTER2SEQ_ROOT = ROOT / "third_party/Raster2Seq"
sys.path.insert(0, str(RASTER2SEQ_ROOT))
sys.path.insert(0, str(RASTER2SEQ_ROOT / "models/ops"))

import util.misc as utils

_torch_load = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load
utils.setup_wandb = lambda: None

if __name__ == "__main__":
    runpy.run_path(str(RASTER2SEQ_ROOT / "main_ddp.py"), run_name="__main__")
