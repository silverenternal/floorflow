"""Legalizer/refiner response sidecar storage and batch attachment."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.legalizer_response import legalizer_response_targets


class ResponseSidecar:
    def __init__(
        self,
        case_indices: np.ndarray,
        displacement: np.ndarray,
        rung: np.ndarray,
        metadata: dict | None = None,
        lengths: np.ndarray | None = None,
    ) -> None:
        indices = np.asarray(case_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sidecar case indices must be non-empty")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("sidecar case indices must be unique")
        self.case_indices = indices
        self.displacement = np.asarray(displacement, dtype=np.float32)
        self.rung = np.asarray(rung, dtype=np.int64)
        if self.displacement.shape[0] != len(indices) or self.rung.shape != (
            len(indices),
        ):
            raise ValueError("sidecar label counts do not match indices")
        self.lengths = (
            np.full(len(indices), self.displacement.shape[1], dtype=np.int16)
            if lengths is None
            else np.asarray(lengths, dtype=np.int16)
        )
        if self.lengths.shape != (len(indices),) or (
            self.lengths < 0
        ).any() or (self.lengths > self.displacement.shape[1]).any():
            raise ValueError("sidecar lengths are invalid")
        self.metadata = dict(metadata or {})
        self.rows = {int(index): row for row, index in enumerate(indices)}
        self._device_arrays: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _arrays_for(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._device_arrays.get(key)
        if cached is not None:
            return cached
        displacement = torch.from_numpy(self.displacement)
        rung = torch.from_numpy(self.rung)
        if device.type == "cuda":
            displacement = displacement.pin_memory()
            rung = rung.pin_memory()
        cached = (
            displacement.to(device, non_blocking=device.type == "cuda"),
            rung.to(device, non_blocking=device.type == "cuda"),
        )
        self._device_arrays[key] = cached
        return cached

    def attach(self, batch: dict) -> None:
        indices = np.asarray(batch["_indices"], dtype=np.int64)
        missing = [int(index) for index in indices if int(index) not in self.rows]
        if missing:
            raise KeyError(f"response sidecar is missing cases: {missing[:8]}")
        rows = np.asarray([self.rows[int(index)] for index in indices])
        device = batch["areas"].device
        device_rows = torch.as_tensor(rows, device=device)
        displacement, rung = self._arrays_for(device)
        width = batch["areas"].shape[-1]
        if width > displacement.shape[1]:
            raise ValueError("sidecar padding is narrower than the batch")
        batch["legalizer_displacement"] = displacement[device_rows, :width]
        batch["legalizer_rung"] = rung[device_rows]
        batch["legalizer_mask"] = (
            torch.arange(width, device=device)[None]
            < self._lengths_for(device)[device_rows, None]
        )

    def _lengths_for(self, device: torch.device) -> torch.Tensor:
        key = f"{device}:lengths"
        cached = self._device_arrays.get(key)
        if cached is None:
            lengths = torch.from_numpy(self.lengths)
            if device.type == "cuda":
                lengths = lengths.pin_memory()
            cached = (
                lengths.to(device, non_blocking=device.type == "cuda"),
                torch.empty(0),
            )
            self._device_arrays[key] = cached
        return cached[0]


def write_response_sidecar(path: str | Path, sidecar: ResponseSidecar) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".npz",
        dir=target.parent,
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                case_indices=sidecar.case_indices,
                displacement=sidecar.displacement,
                rung=sidecar.rung,
                lengths=sidecar.lengths,
                metadata=np.asarray(json.dumps(sidecar.metadata)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_response_sidecar(path: str | Path) -> ResponseSidecar:
    payload = np.load(path, allow_pickle=False)
    return ResponseSidecar(
        payload["case_indices"],
        payload["displacement"],
        payload["rung"],
        json.loads(str(payload["metadata"].item())),
        lengths=payload.get("lengths", None),
    )


def build_response_sidecar(
    case_indices: np.ndarray,
    before_boxes: torch.Tensor,
    after_boxes: torch.Tensor,
    areas: torch.Tensor,
    rung: torch.Tensor,
    *,
    lengths: torch.Tensor | np.ndarray | None = None,
    metadata: dict | None = None,
) -> ResponseSidecar:
    targets = legalizer_response_targets(
        before_boxes.float(),
        after_boxes.float(),
        rung.long(),
        areas.float(),
    )
    return ResponseSidecar(
        case_indices,
        targets["legalizer_displacement"].cpu().numpy(),
        targets["legalizer_rung"].cpu().numpy(),
        metadata,
        lengths=(
            None
            if lengths is None
            else np.asarray(lengths, dtype=np.int16)
        ),
    )


__all__ = [
    "ResponseSidecar",
    "build_response_sidecar",
    "load_response_sidecar",
    "write_response_sidecar",
]
