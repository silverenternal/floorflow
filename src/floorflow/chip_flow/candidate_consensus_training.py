"""Offline candidate-consensus distillation contracts and objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.candidate_manifest import SOURCE_GEOMETRY_KEYS
from floorflow.chip_flow.legalization_survival import (
    legalization_rank_retention_loss,
)


@dataclass(frozen=True)
class CandidateConsensusBatch:
    candidate_states: torch.Tensor
    candidate_features: torch.Tensor
    quality: torch.Tensor
    target_state: torch.Tensor
    candidate_mask: torch.Tensor
    context: torch.Tensor | None = None

    def validate(self) -> None:
        if self.candidate_states.ndim != 4 or self.candidate_states.shape[-1] != 3:
            raise ValueError("candidate_states must have shape (batch, candidates, blocks, 3)")
        shape = self.candidate_states.shape[:2]
        if self.candidate_features.shape[:2] != shape:
            raise ValueError("candidate feature candidate axes mismatch")
        if self.candidate_features.ndim != 3 or self.candidate_features.shape[-1] <= 0:
            raise ValueError("candidate features must have a positive feature width")
        if self.quality.shape != shape:
            raise ValueError("quality must match candidate batch axes")
        if self.candidate_mask.shape != shape:
            raise ValueError("candidate mask must match candidate batch axes")
        if self.target_state.shape != (
            shape[0],
            self.candidate_states.shape[2],
            3,
        ):
            raise ValueError("target state must have shape (batch, blocks, 3)")
        if not bool(self.candidate_mask[:, 0].all()):
            raise ValueError("reference candidate must remain valid")
        valid = self.candidate_mask
        if not torch.isfinite(self.candidate_states[valid]).all():
            raise ValueError("valid candidate states must be finite")
        if not torch.isfinite(self.candidate_features[valid]).all():
            raise ValueError("valid candidate features must be finite")
        if not torch.isfinite(self.quality[valid]).all():
            raise ValueError("valid candidate quality must be finite")
        if not torch.isfinite(self.target_state).all():
            raise ValueError("candidate target state must be finite")


@dataclass(frozen=True)
class CandidateConsensusTerms:
    total: torch.Tensor
    endpoint: torch.Tensor
    teacher: torch.Tensor
    entropy: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "candidate_consensus": self.total,
            "candidate_consensus_endpoint": self.endpoint,
            "candidate_consensus_teacher": self.teacher,
            "candidate_consensus_entropy": self.entropy,
        }


def load_candidate_consensus_batch(path: Path) -> CandidateConsensusBatch:
    payload = np.load(path)
    required = {
        "candidate_states",
        "candidate_features",
        "quality",
        "target_state",
        "candidate_mask",
    }
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"candidate cohort is missing arrays: {sorted(missing)}")
    batch = CandidateConsensusBatch(
        candidate_states=torch.as_tensor(payload["candidate_states"]).float(),
        candidate_features=torch.as_tensor(payload["candidate_features"]).float(),
        quality=torch.as_tensor(payload["quality"]).float(),
        target_state=torch.as_tensor(payload["target_state"]).float(),
        candidate_mask=torch.as_tensor(payload["candidate_mask"]).bool(),
        context=(
            torch.as_tensor(payload["context"]).float()
            if "context" in payload.files
            else None
        ),
    )
    batch.validate()
    return batch


def write_candidate_consensus_cohort(
    output_dir: Path,
    cases: list[dict],
) -> list[Path]:
    """Write one self-contained NPZ shard per case from a candidate manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    indices = [int(case["index"]) for case in cases]
    if len(indices) != len(set(indices)):
        raise ValueError("candidate cohort case indices must be unique")
    for case in cases:
        index = int(case["index"])
        candidates = case["candidates"]
        if not candidates:
            raise ValueError(f"candidate case {index} contains no candidates")
        states = np.asarray(
            [candidate["state"] for candidate in candidates],
            dtype="float32",
        )
        features = np.asarray(
            [candidate["features"] for candidate in candidates],
            dtype="float32",
        )
        quality = np.asarray(
            [candidate["quality"] for candidate in candidates],
            dtype="float32",
        )
        target = np.asarray(case["target_state"], dtype="float32")
        mask = np.asarray(
            [candidate.get("valid", True) for candidate in candidates],
            dtype=bool,
        )
        batch = CandidateConsensusBatch(
            candidate_states=torch.as_tensor(states[None]),
            candidate_features=torch.as_tensor(features[None]),
            quality=torch.as_tensor(quality[None]),
            target_state=torch.as_tensor(target[None]),
            candidate_mask=torch.as_tensor(mask[None]),
            context=(
                torch.as_tensor(np.asarray(case["context"], dtype="float32"))[None]
                if "context" in case
                else None
            ),
        )
        batch.validate()
        arrays = {
            "case_index": np.asarray(index, dtype=np.int64),
            "candidate_states": states[None],
            "candidate_features": features[None],
            "quality": quality[None],
            "target_state": target[None],
            "candidate_mask": mask[None],
        }
        if batch.context is not None:
            arrays["context"] = batch.context.numpy()
        for key in SOURCE_GEOMETRY_KEYS:
            if key in case.get("source_geometry", {}):
                arrays[f"source_{key}"] = np.asarray(
                    case["source_geometry"][key],
                    dtype=np.float32,
                )
        path = output_dir / f"case_{index:08d}.npz"
        np.savez(path, **arrays)
        written.append(path)
    return written


def candidate_consensus_objective(
    model: CandidateConsensusFuser,
    batch: CandidateConsensusBatch,
    *,
    teacher_temperature: float = 0.1,
    endpoint_weight: float = 1.0,
    teacher_weight: float = 0.1,
    entropy_weight: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, CandidateConsensusTerms]:
    """Train the fuser against both a best endpoint and M75 teacher ranking."""

    batch.validate()
    if teacher_temperature <= 0.0:
        raise ValueError("teacher temperature must be positive")
    if min(endpoint_weight, teacher_weight, entropy_weight) < 0.0:
        raise ValueError("consensus weights must be non-negative")
    fused, weights = model(
        batch.candidate_states,
        batch.candidate_features,
        batch.candidate_mask,
        batch.context,
    )
    if model.selection_mode == "topk_residual":
        teacher_loss = legalization_rank_retention_loss(
            model.candidate_scores(
                batch.candidate_features,
                batch.candidate_mask,
            ),
            batch.quality,
            batch.candidate_mask,
            temperature=teacher_temperature,
        )
    else:
        teacher_logits = -batch.quality / float(teacher_temperature)
        teacher_logits = teacher_logits.masked_fill(
            ~batch.candidate_mask,
            float("-inf"),
        )
        teacher = torch.softmax(teacher_logits, dim=-1)
        teacher_loss = -(
            teacher * weights.clamp_min(1.0e-8).log()
        ).sum(dim=-1).mean()
    endpoint = F.smooth_l1_loss(
        fused.float(),
        batch.target_state.float(),
        beta=0.1,
    )
    entropy = (
        -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
    )
    total = (
        float(endpoint_weight) * endpoint
        + float(teacher_weight) * teacher_loss
        - float(entropy_weight) * entropy
    )
    return total, weights, CandidateConsensusTerms(
        total=total,
        endpoint=endpoint,
        teacher=teacher_loss,
        entropy=entropy,
    )


def load_candidate_consensus_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> CandidateConsensusFuser:
    """Restore a distilled consensus fuser using its serialized dimensions."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("candidate consensus checkpoint lacks config")
    model = CandidateConsensusFuser(
        int(config["hidden"]),
        candidate_feature_dim=int(config["feature_dim"]),
        context_dim=int(config.get("context_dim", 0)),
        selection_mode=str(config.get("selection_mode", "soft")),
        residual_limit=float(config.get("residual_limit", 0.05)),
        top_k=(
            int(config["top_k"])
            if config.get("top_k") is not None
            else None
        ),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


__all__ = [
    "CandidateConsensusBatch",
    "CandidateConsensusTerms",
    "candidate_consensus_objective",
    "load_candidate_consensus_batch",
    "load_candidate_consensus_checkpoint",
    "write_candidate_consensus_cohort",
]
