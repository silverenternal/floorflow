"""Audit parity between independently materialized predicted-control sources."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.control_cache import read_control_cache
from floorflow.data.model_candidate_dataset import target_cache_path
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta

ARRAY_NAMES = ("room_logits", "program", "structure", "furniture", "support")
CONTRACT_KEYS = (
    "schema_version",
    "source",
    "analysis_variant",
    "target_used_for_control",
    "deployable",
    "flow_steps",
    "noise_scale",
    "control_source",
    "checkpoint_sha256",
    "taxonomy_sha256",
    "control_sizes",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary_counts(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.bool()
    target = target.bool()
    tp = float((prediction & target).sum())
    fp = float((prediction & ~target).sum())
    fn = float((~prediction & target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-8),
        "iou": tp / max(tp + fp + fn, 1.0),
    }


def _resize_labels(labels: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        labels[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0].long()


def _target_label_path(
    cache_root: str | Path,
    split: str,
    name: str,
) -> Path:
    """Resolve flat legacy and nested full-cache target layouts."""

    parts = name.split("__")
    if len(parts) >= 2:
        nested = (
            Path(cache_root)
            / split
            / parts[0]
            / f"{parts[1].replace('-', '_')}.npz"
        )
        if nested.is_file():
            return nested
    return target_cache_path(cache_root, split, name)


def _room_macro_iou(
    control: dict[str, torch.Tensor],
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> float:
    room_logits = control["room_logits"]
    size = tuple(room_logits.shape[-2:])
    labels = _resize_labels(labels, size)
    prediction = room_logits.argmax(dim=0)
    values = []
    for class_index, room_id in enumerate(taxonomy.room_ids):
        target = labels == int(room_id)
        if not bool(target.any()):
            continue
        values.append(_binary_counts(prediction == class_index, target)["iou"])
    return float(np.mean(values)) if values else 1.0


def _program_metrics(
    program: torch.Tensor,
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    label_values = {int(value) for value in labels.unique()}
    target = torch.tensor(
        [int(room_id) in label_values for room_id in taxonomy.room_ids]
    )
    prediction = program.bool().cpu()
    return _binary_counts(prediction, target) | {
        "exact": float(torch.equal(prediction, target))
    }


def control_fidelity_row(
    control: dict[str, torch.Tensor],
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    room_count = len(taxonomy.room_ids)
    structure = control["structure"]
    furniture = control["furniture"]
    structure_size = tuple(structure.shape[-2:])
    furniture_size = tuple(furniture.shape[-2:])
    resized = _resize_labels(labels, structure_size)
    opening_ids = torch.tensor(taxonomy.opening_ids)
    furniture_ids = torch.tensor(taxonomy.furniture_ids)
    wall = _binary_counts(structure[room_count] > 0.5, resized == taxonomy.class_id("wall"))
    opening = _binary_counts(
        structure[room_count + 1] > 0.5,
        torch.isin(resized, opening_ids),
    )
    furniture_target = F.interpolate(
        torch.isin(labels[None, None], furniture_ids).float(),
        size=furniture_size,
        mode="nearest",
    )[0, 0].bool()
    furnishings = _binary_counts(
        furniture[room_count:-1].amax(dim=0) > 0.5,
        furniture_target,
    )
    row = {
        "room_macro_iou": _room_macro_iou(control, labels, taxonomy),
        **{f"wall_{key}": value for key, value in wall.items()},
        **{f"opening_{key}": value for key, value in opening.items()},
        **{f"furniture_{key}": value for key, value in furnishings.items()},
    }
    row.update(
        {
            f"program_{key}": value
            for key, value in _program_metrics(
                control["program"],
                labels,
                taxonomy,
            ).items()
        }
    )
    return row


def _sampled_stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.float().flatten()
    stride = max(1, int(flat.numel()) // 4096)
    sample = flat[::stride]
    return {
        "mean": float(sample.mean()),
        "std": float(sample.std(unbiased=False)),
        "positive_fraction": float((sample > 0.5).float().mean()),
        "q05": float(torch.quantile(sample, 0.05)),
        "q50": float(torch.quantile(sample, 0.50)),
        "q95": float(torch.quantile(sample, 0.95)),
    }


def _manifest_audit(reference: dict, candidate: dict) -> dict[str, object]:
    mismatches = {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in CONTRACT_KEYS
        if reference.get(key) != candidate.get(key)
    }
    return {"passed": not mismatches, "mismatches": mismatches}


def relative_equivalence_verdict(row: dict[str, float]) -> dict[str, object]:
    """Use a 10% relative margin with a one-point floor for fidelity parity."""

    margin = max(0.01, 0.1 * abs(float(row["baseline_mean"])))
    return {
        "margin": margin,
        "passed": (
            abs(float(row["mean_delta"])) <= margin
            and float(row["ci_low"]) >= -margin
            and float(row["ci_high"]) <= margin
        ),
    }


def audit_control_source_pair(
    reference_root: str | Path,
    candidate_root: str | Path,
    *,
    splits: tuple[str, ...],
    cache_root: str | Path,
    bootstrap_iterations: int = 1000,
) -> dict[str, object]:
    reference_root = Path(reference_root)
    candidate_root = Path(candidate_root)
    cache_root = Path(cache_root)
    taxonomy = FloorplanTaxonomy.read(cache_root / "taxonomy.json")
    report: dict[str, object] = {
        "version": "control-source-parity-v1",
        "reference_root": str(reference_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "cache_root": str(cache_root.resolve()),
        "splits": {},
        "checks": {},
    }
    all_passed = True
    for split in splits:
        reference_manifest = json.loads(
            (reference_root / split / "manifest.json").read_text(encoding="utf-8")
        )
        candidate_manifest = json.loads(
            (candidate_root / split / "manifest.json").read_text(encoding="utf-8")
        )
        reference_index = {row["name"]: row for row in reference_manifest["rows"]}
        candidate_index = {row["name"]: row for row in candidate_manifest["rows"]}
        reference_names = set(reference_index)
        candidate_names = set(candidate_index)
        names = sorted(reference_names & candidate_names)
        manifest_audit = _manifest_audit(reference_manifest, candidate_manifest)
        reference_rows = []
        candidate_rows = []
        dtype_mismatches: Counter[str] = Counter()
        shape_mismatches: Counter[str] = Counter()
        checksum_mismatches = {"reference": [], "candidate": []}
        raw_summary = {
            name: {"reference": [], "candidate": []}
            for name in ARRAY_NAMES
        }
        for name in names:
            filename = f"{name.replace('/', '__')}.npz"
            reference_path = reference_root / split / filename
            candidate_path = candidate_root / split / filename
            if sha256_file(reference_path) != reference_index[name].get("sha256"):
                checksum_mismatches["reference"].append(name)
            if sha256_file(candidate_path) != candidate_index[name].get("sha256"):
                checksum_mismatches["candidate"].append(name)
            reference = read_control_cache(reference_path)
            candidate = read_control_cache(candidate_path)
            label_path = _target_label_path(cache_root, split, name)
            with np.load(label_path) as payload:
                labels = torch.from_numpy(payload["target"].astype(np.int64))
            reference_rows.append(control_fidelity_row(reference, labels, taxonomy))
            candidate_rows.append(control_fidelity_row(candidate, labels, taxonomy))
            for array_name in ARRAY_NAMES:
                if reference[array_name].shape != candidate[array_name].shape:
                    shape_mismatches[array_name] += 1
                    continue
                if reference[array_name].dtype != candidate[array_name].dtype:
                    dtype_mismatches[array_name] += 1
                raw_summary[array_name]["reference"].append(
                    _sampled_stats(reference[array_name])
                )
                raw_summary[array_name]["candidate"].append(
                    _sampled_stats(candidate[array_name])
                )
        metric_names = sorted(reference_rows[0]) if reference_rows else []
        paired = {
            metric: paired_delta(
                candidate_rows,
                reference_rows,
                metric,
                iterations=int(bootstrap_iterations),
            )
            for metric in metric_names
        }
        raw_parity = {}
        for array_name, rows in raw_summary.items():
            raw_parity[array_name] = {
                key: {
                    "reference": float(
                        np.mean([row[key] for row in rows["reference"]])
                    ),
                    "candidate": float(
                        np.mean([row[key] for row in rows["candidate"]])
                    ),
                }
                for key in ("mean", "std", "positive_fraction", "q05", "q50", "q95")
            }
        equivalence_metrics = (
            "room_macro_iou",
            "wall_f1",
            "opening_f1",
            "furniture_f1",
            "program_f1",
        )
        equivalence = {
            metric: relative_equivalence_verdict(paired[metric])
            for metric in equivalence_metrics
        }
        split_checks = {
            "manifest_contract": bool(manifest_audit["passed"]),
            "manifest_case_count_consistent": (
                int(reference_manifest.get("cases", -1)) == len(reference_names)
                and int(candidate_manifest.get("cases", -1)) == len(candidate_names)
            ),
            "candidate_covers_reference_count": (
                len(candidate_names) >= len(reference_names)
            ),
            "reference_names_are_subset": reference_names <= candidate_names,
            "shared_cases_complete": len(names) == len(reference_names),
            "arrays_shape_exact": not shape_mismatches,
            "arrays_dtype_exact": not dtype_mismatches,
            "manifest_file_sha256_exact": not any(checksum_mismatches.values()),
            "target_fidelity_relative_equivalence": all(
                bool(row["passed"]) for row in equivalence.values()
            ),
        }
        all_passed &= all(split_checks.values())
        report["splits"][split] = {
            "reference_manifest": str(reference_root / split / "manifest.json"),
            "candidate_manifest": str(candidate_root / split / "manifest.json"),
            "reference_cases": len(reference_names),
            "candidate_cases": len(candidate_names),
            "shared_cases": len(names),
            "manifest_audit": manifest_audit,
            "shape_mismatches": dict(shape_mismatches),
            "dtype_mismatches": dict(dtype_mismatches),
            "checksum_mismatches": checksum_mismatches,
            "raw_distribution_parity": raw_parity,
            "target_fidelity_paired": paired,
            "target_fidelity_equivalence": equivalence,
            "checks": split_checks,
        }
    report["checks"] = {"all_splits_passed": all_passed}
    report["decision"] = "parity_pass" if all_passed else "parity_fail"
    return report


__all__ = [
    "ARRAY_NAMES",
    "CONTRACT_KEYS",
    "audit_control_source_pair",
    "control_fidelity_row",
    "relative_equivalence_verdict",
    "sha256_file",
]
