"""Convert exported refiner/legalizer outcomes into a response sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.response_sidecar import (
    build_response_sidecar,
    write_response_sidecar,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = np.load(args.input, allow_pickle=False)
    sidecar = build_response_sidecar(
        payload["case_indices"],
        torch.from_numpy(payload["before_boxes"]),
        torch.from_numpy(payload["after_boxes"]),
        torch.from_numpy(payload["areas"]),
        torch.from_numpy(payload["rung"]),
        metadata={
            "source": str(args.input),
            "schema": "case_indices,before_boxes,after_boxes,areas,rung",
        },
    )
    write_response_sidecar(args.output, sidecar)


if __name__ == "__main__":
    main()
