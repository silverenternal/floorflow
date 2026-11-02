"""Compact, memory-mappable storage for FloorSet chip flow experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.proposals import prepare_flow_pair

ARRAY_FIELDS = (
    "areas",
    "boxes",
    "constraints",
    "features",
    "b2b",
    "p2b",
    "pins",
    "source_latent",
    "target_latent",
)

DEVICE_BATCH_FIELDS = (
    "n",
    "n_b2b",
    "n_p2b",
    "n_pins",
    "b2b_offset",
    "p2b_offset",
    "pins_offset",
    "areas",
    "boxes",
    "constraints",
    "features",
    "source_latent",
    "target_latent",
    "b2b",
    "p2b",
    "pins",
)


def gather_padded_rows(
    flat: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    lengths: np.ndarray,
    width: int,
) -> np.ndarray:
    """Vectorized gather of variable-length rows into a padded batch."""

    rows = np.asarray(rows, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    maximum = int(lengths.max()) if lengths.size else 0
    output = np.zeros((len(rows), maximum, width), dtype=np.float32)
    if maximum == 0:
        return output
    positions = np.arange(maximum, dtype=np.int64)[None, :]
    valid = positions < lengths[:, None]
    flat_indices = np.asarray(offsets, dtype=np.int64)[rows, None] + positions
    output[valid] = np.asarray(flat)[flat_indices[valid]]
    return output


def _source_table(cases: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = sorted({str(case.get("source")) for case in cases})
    lookup = {path: index for index, path in enumerate(paths)}
    source_ids = np.asarray(
        [lookup[str(case.get("source"))] for case in cases],
        dtype=np.int32,
    )
    source_indices = np.asarray(
        [int(case.get("source_index", -1)) for case in cases],
        dtype=np.int16,
    )
    return source_ids, source_indices, paths


def write_packed_split(cases: list[dict], directory: str | Path) -> dict:
    """Write one split as mmap-friendly padded arrays plus metadata."""

    if not cases:
        raise ValueError("packed split requires at least one case")
    feature_widths = {int(case["features"].shape[-1]) for case in cases}
    if len(feature_widths) != 1:
        raise ValueError("packed split cases have inconsistent feature widths")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    block_counts = np.asarray(
        [case["areas"].numel() for case in cases],
        dtype=np.int16,
    )
    max_blocks = int(block_counts.max())
    feature_dim = feature_widths.pop()

    def flattened(name: str, width: int) -> np.ndarray:
        parts = [case[name].numpy() for case in cases if len(case[name])]
        if not parts:
            return np.zeros((0, width), dtype=np.float32)
        return np.concatenate(parts)

    arrays = {
        "n": block_counts,
        "n_b2b": np.asarray([len(case["b2b"]) for case in cases], dtype=np.int32),
        "n_p2b": np.asarray([len(case["p2b"]) for case in cases], dtype=np.int32),
        "n_pins": np.asarray([len(case["pins"]) for case in cases], dtype=np.int32),
        "areas": np.zeros((len(cases), max_blocks), dtype=np.float32),
        "boxes": np.zeros((len(cases), max_blocks, 4), dtype=np.float32),
        "constraints": np.zeros((len(cases), max_blocks, 5), dtype=np.float32),
        "features": np.zeros(
            (len(cases), max_blocks, feature_dim),
            dtype=np.float32,
        ),
        "b2b": flattened("b2b", 3),
        "p2b": flattened("p2b", 3),
        "pins": flattened("pins", 2),
        "source_latent": np.zeros((len(cases), max_blocks, 3), dtype=np.float32),
        "target_latent": np.zeros((len(cases), max_blocks, 3), dtype=np.float32),
    }
    for row, case in enumerate(cases):
        count = int(case["areas"].numel())
        arrays["areas"][row, :count] = case["areas"].numpy()
        arrays["boxes"][row, :count] = case["boxes"].numpy()
        arrays["constraints"][row, :count] = case["constraints"].numpy()
        arrays["features"][row, :count] = case["features"].numpy()
        source, target = prepare_flow_pair(
            case["boxes"],
            case["areas"],
        )
        arrays["source_latent"][row, :count] = source.numpy()
        arrays["target_latent"][row, :count] = target.numpy()
    for name, count_name in (
        ("b2b", "n_b2b"),
        ("p2b", "n_p2b"),
        ("pins", "n_pins"),
    ):
        arrays[f"{name}_offset"] = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(arrays[count_name])]
        )

    source_ids, source_indices, source_paths = _source_table(cases)
    arrays["source_id"] = source_ids
    arrays["source_index"] = source_indices
    for name, array in arrays.items():
        np.save(out / f"{name}.npy", array)
    metadata = {
        "cases": len(cases),
        "min_blocks": int(block_counts.min()),
        "max_blocks": max_blocks,
        "max_b2b_edges": int(arrays["n_b2b"].max()),
        "max_p2b_edges": int(arrays["n_p2b"].max()),
        "max_pins": int(arrays["n_pins"].max()),
        "feature_dim": feature_dim,
        "source_paths": source_paths,
        "block_count_histogram": {
            str(count): int((block_counts == count).sum())
            for count in np.unique(block_counts)
        },
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


class PackedCaseStore:
    """Read-only random access over one compact split."""

    def __init__(self, directory: str | Path) -> None:
        self.root = Path(directory)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.arrays = {
            path.stem: np.load(path, mmap_mode="r")
            for path in self.root.glob("*.npy")
        }
        required = {
            "n",
            "n_b2b",
            "n_p2b",
            "n_pins",
            "b2b_offset",
            "p2b_offset",
            "pins_offset",
            "source_id",
            "source_index",
            *ARRAY_FIELDS,
        }
        missing = sorted(required - set(self.arrays))
        if missing:
            raise ValueError(f"packed case cache is missing arrays: {missing}")
        cases = int(self.metadata["cases"])
        if cases != int(self.arrays["n"].shape[0]):
            raise ValueError("packed case metadata count does not match n.npy")
        if int(self.arrays["features"].shape[-1]) != int(
            self.metadata["feature_dim"]
        ):
            raise ValueError("packed case feature width does not match metadata")
        if cases > 0 and int(self.arrays["n"].max()) != int(
            self.arrays["areas"].shape[1]
        ):
            raise ValueError("packed case block width does not match metadata")
        for name, count_name, width in (
            ("b2b", "n_b2b", 3),
            ("p2b", "n_p2b", 3),
            ("pins", "n_pins", 2),
        ):
            lengths = self.arrays[count_name]
            offsets = self.arrays[f"{name}_offset"]
            if lengths.shape != (cases,) or offsets.shape != (cases + 1,):
                raise ValueError(f"packed {name} count layout is invalid")
            if int(offsets[0]) != 0 or int(offsets[-1]) != int(lengths.sum()):
                raise ValueError(f"packed {name} offsets are inconsistent")
            if self.arrays[name].shape != (int(lengths.sum()), width):
                raise ValueError(f"packed {name} tensor shape is inconsistent")
        self.by_block_count: dict[int, np.ndarray] = defaultdict(
            lambda: np.empty(0, dtype=np.int64)
        )
        for count in np.unique(self.arrays["n"]):
            self.by_block_count[int(count)] = np.flatnonzero(
                self.arrays["n"] == count
            )

    def __len__(self) -> int:
        return int(self.arrays["n"].shape[0])

    def indices(self, block_count: int | None = None) -> np.ndarray:
        if block_count is None:
            return np.arange(len(self), dtype=np.int64)
        return self.by_block_count[int(block_count)]

    def batch(
        self,
        indices: np.ndarray,
        *,
        device: str | torch.device = "cpu",
        pin_memory: bool | None = None,
        non_blocking: bool | None = None,
    ) -> dict:
        """Gather one block-count batch with optional pinned async transfer."""

        rows = np.asarray(indices, dtype=np.int64)
        if rows.size == 0:
            raise ValueError("cannot read an empty batch")
        counts = self.arrays["n"][rows]
        if not np.all(counts == counts[0]):
            raise ValueError("PackedCaseStore batch requires one block count")
        count = int(counts[0])

        def gather_flat(name: str, count_name: str, width: int) -> np.ndarray:
            lengths = self.arrays[count_name][rows]
            return gather_padded_rows(
                self.arrays[name],
                self.arrays[f"{name}_offset"],
                rows,
                lengths,
                width,
            )

        batch = {
            "n": count,
            "areas": self.arrays["areas"][rows, :count],
            "boxes": self.arrays["boxes"][rows, :count],
            "constraints": self.arrays["constraints"][rows, :count],
            "features": self.arrays["features"][rows, :count],
            "source_latent": self.arrays["source_latent"][rows, :count],
            "target_latent": self.arrays["target_latent"][rows, :count],
            "b2b": gather_flat("b2b", "n_b2b", 3),
            "p2b": gather_flat("p2b", "n_p2b", 3),
            "pins": gather_flat("pins", "n_pins", 2),
            "n_b2b": self.arrays["n_b2b"][rows],
            "n_p2b": self.arrays["n_p2b"][rows],
            "n_pins": self.arrays["n_pins"][rows],
        }
        resolved_device = torch.device(device)
        pinned = (
            resolved_device.type == "cuda"
            if pin_memory is None
            else bool(pin_memory)
        )
        asynchronous = (
            pinned and resolved_device.type == "cuda"
            if non_blocking is None
            else bool(non_blocking)
        )
        transferred = {}
        for key, value in batch.items():
            if key == "n":
                transferred[key] = value
                continue
            tensor = torch.from_numpy(np.asarray(value))
            if pinned:
                tensor = tensor.pin_memory()
            transferred[key] = tensor.to(
                resolved_device,
                non_blocking=asynchronous,
            )
        return transferred

    def to_device(
        self,
        device: str | torch.device,
        *,
        max_gib: float = 16.0,
    ) -> DeviceCaseStore | PackedCaseStore:
        """Return a device-resident copy when the configured budget permits."""

        resolved = torch.device(device)
        if resolved.type != "cuda":
            return self
        return DeviceCaseStore(self, resolved, max_gib=max_gib)


class DeviceCaseStore:
    """Read-only batch source with active cache arrays resident on one device."""

    def __init__(
        self,
        source: PackedCaseStore,
        device: str | torch.device,
        *,
        max_gib: float = 16.0,
    ) -> None:
        if max_gib <= 0.0:
            raise ValueError("GPU cache budget must be positive")
        resolved = torch.device(device)
        if resolved.type == "cuda":
            required = sum(
                int(source.arrays[name].nbytes)
                for name in DEVICE_BATCH_FIELDS
                if name in source.arrays
            )
            free, _total = torch.cuda.mem_get_info(resolved)
            if required > int(max_gib * 1024**3):
                raise MemoryError(
                    f"GPU data cache needs {required / 1024**3:.2f} GiB, "
                    f"above the {max_gib:.2f} GiB budget"
                )
            if required > int(free * 0.85):
                raise MemoryError(
                    f"GPU data cache needs {required / 1024**3:.2f} GiB but "
                    f"only {free / 1024**3:.2f} GiB is free"
                )
        self.metadata = source.metadata
        self.by_block_count = source.by_block_count
        self._counts_cpu = source.arrays["n"]
        self._lengths_cpu = {
            name: source.arrays[name]
            for name in ("n_b2b", "n_p2b", "n_pins")
        }
        self.resident_device = resolved
        self.required_bytes = 0
        staged = []
        self.arrays = {}
        for name in DEVICE_BATCH_FIELDS:
            tensor = torch.from_numpy(
                np.array(source.arrays[name], copy=True)
            )
            if resolved.type == "cuda":
                tensor = tensor.pin_memory()
                staged.append(tensor)
            self.arrays[name] = tensor.to(resolved, non_blocking=True)
            self.required_bytes += int(tensor.numel() * tensor.element_size())
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)
        del staged

    def __len__(self) -> int:
        return int(self._counts_cpu.shape[0])

    def indices(self, block_count: int | None = None) -> np.ndarray:
        if block_count is None:
            return np.arange(len(self), dtype=np.int64)
        return self.by_block_count[int(block_count)]

    def batch(
        self,
        indices: np.ndarray,
        *,
        device: str | torch.device | None = None,
        pin_memory: bool | None = None,
        non_blocking: bool | None = None,
    ) -> dict:
        del pin_memory, non_blocking
        resolved = self.resident_device if device is None else torch.device(device)
        rows_np = np.asarray(indices, dtype=np.int64)
        if rows_np.size == 0:
            raise ValueError("cannot read an empty batch")
        counts = self._counts_cpu[rows_np]
        if not np.all(counts == counts[0]):
            raise ValueError("DeviceCaseStore batch requires one block count")
        count = int(counts[0])
        rows = torch.as_tensor(rows_np, device=self.resident_device)

        def gather_flat(name: str, count_name: str, width: int) -> torch.Tensor:
            lengths = self.arrays[count_name][rows]
            maximum = int(
                np.asarray(self._lengths_cpu[count_name][rows_np]).max(
                    initial=0
                )
            )
            if maximum == 0:
                return self.arrays[name].new_zeros((len(rows), 0, width))
            positions = torch.arange(
                maximum,
                device=self.resident_device,
            )[None]
            valid = positions < lengths[:, None]
            offsets = self.arrays[f"{name}_offset"][rows, None]
            indices_flat = offsets + positions
            gathered = self.arrays[name][indices_flat.clamp_max(
                self.arrays[name].shape[0] - 1
            )]
            output = self.arrays[name].new_zeros((len(rows), maximum, width))
            output[valid] = gathered[valid]
            return output

        batch = {
            "n": count,
            "areas": self.arrays["areas"][rows, :count],
            "boxes": self.arrays["boxes"][rows, :count],
            "constraints": self.arrays["constraints"][rows, :count],
            "features": self.arrays["features"][rows, :count],
            "source_latent": self.arrays["source_latent"][rows, :count],
            "target_latent": self.arrays["target_latent"][rows, :count],
            "b2b": gather_flat("b2b", "n_b2b", 3),
            "p2b": gather_flat("p2b", "n_p2b", 3),
            "pins": gather_flat("pins", "n_pins", 2),
            "n_b2b": self.arrays["n_b2b"][rows],
            "n_p2b": self.arrays["n_p2b"][rows],
            "n_pins": self.arrays["n_pins"][rows],
        }
        if resolved != self.resident_device:
            return {
                key: value if key == "n" else value.to(resolved)
                for key, value in batch.items()
            }
        return batch


def prepare_training_store(
    store: PackedCaseStore,
    device: torch.device,
    *,
    enabled: bool,
    max_gib: float,
) -> tuple[PackedCaseStore | DeviceCaseStore, dict]:
    """Move the active train arrays to the device or return a fallback status."""

    status = {"requested": bool(enabled), "resident": False}
    if not enabled or device.type != "cuda":
        return store, status
    try:
        resident = DeviceCaseStore(store, device, max_gib=max_gib)
    except MemoryError as error:
        status.update(
            {
                "fallback": "pinned_cpu_prefetch",
                "reason": str(error),
            }
        )
        return store, status
    status.update(
        {
            "resident": True,
            "required_gib": resident.required_bytes / 1024**3,
        }
    )
    return resident, status


__all__ = [
    "ARRAY_FIELDS",
    "DeviceCaseStore",
    "PackedCaseStore",
    "gather_padded_rows",
    "prepare_training_store",
    "write_packed_split",
]
