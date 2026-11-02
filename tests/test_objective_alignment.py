from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from floorflow.data.mixed_dataset import MixtureCurriculum
from floorflow.data.taxonomy import build_taxonomy
from floorflow.flow.cfm_backend import sample_flow_batch
from floorflow.flow.presets import resolve_flow_preset
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.objectives.semantic_losses import boundary_distance_loss
from floorflow.objectives.structural_consistency import opening_wall_affinity_loss
from floorflow.objectives.token_optimal_transport import room_token_transport_loss


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_occluded_pixels_are_excluded_from_identity_target() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((16, 16), dtype=np.int64)
    labels[2:14, 2:8] = taxonomy.room_ids[0]
    labels[2:14, 8:14] = taxonomy.room_ids[1]
    footprint = np.ones((16, 16), dtype=np.float32)
    labels[5:8, 5:11] = taxonomy.class_id("furniture:Sink")
    visible = build_room_partition_target(
        labels,
        footprint,
        taxonomy,
        fill_occluded=False,
    )
    dense = build_room_partition_target(labels, footprint, taxonomy)
    assert (visible[5:8, 5:11] == -100).all()
    assert (dense[footprint.astype(bool)] >= 0).all()


def test_boundary_distance_loss_prefers_correct_transition() -> None:
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, 8:] = 1
    logits = torch.full((1, 2, 16, 16), -4.0)
    logits[:, 0] = 4.0
    logits[:, 1] = -4.0
    logits[:, :, :, 8:] = logits[:, :, :, 8:].flip(1)
    wrong = logits.clone().flip(1).requires_grad_(True)
    correct = logits.clone().requires_grad_(True)
    correct_loss = boundary_distance_loss(correct, target)
    wrong_loss = boundary_distance_loss(wrong, target)
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert correct.grad is not None
    assert correct.grad.abs().sum() > 0


def test_opening_wall_affinity_penalizes_far_openings() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, 7:9] = taxonomy.class_id("wall")
    footprint = torch.ones(1, 16, 16)
    background = torch.zeros(1, 16, 16)
    near = torch.full((1, 1, 16, 16), -8.0)
    near[:, :, :, 8] = 8.0
    far = torch.full((1, 1, 16, 16), -8.0)
    far[:, :, :, 1] = 8.0
    near_loss = opening_wall_affinity_loss(
        near, background, target, footprint, taxonomy
    )
    far_loss = opening_wall_affinity_loss(
        far, background, target, footprint, taxonomy
    )
    assert near_loss < far_loss


def test_flow_presets_and_time_sampling_are_explicit() -> None:
    preset = resolve_flow_preset("fast_vp")
    assert preset.matcher == "torchcfm_vp"
    assert preset.time_sampling == "logit_normal"
    source = torch.randn(4, 2, 8, 8)
    target = torch.randn_like(source)
    flow = sample_flow_batch(
        source,
        target,
        backend="legacy",
        time_sampling="logit_normal",
    )
    assert flow.location.shape == source.shape
    expected = source + flow.time[:, None, None, None] * (target - source)
    assert torch.allclose(flow.location, expected, atol=1.0e-6, rtol=1.0e-6)
    fast_cfm = resolve_flow_preset("fast_cfm")
    assert fast_cfm.matcher == "torchcfm"
    assert fast_cfm.time_sampling == "logit_normal"


def test_mixture_curriculum_ramps_after_real_only_phase() -> None:
    curriculum = MixtureCurriculum(
        real_only_epochs=1,
        ramp_epochs=2,
        final_real_fraction=0.8,
    )
    assert curriculum.real_fraction(0) == 1.0
    assert curriculum.real_fraction(1) == 0.9
    assert curriculum.real_fraction(2) == 0.8
    assert curriculum.real_fraction(5) == 0.8


def test_room_token_transport_is_differentiable() -> None:
    predicted = torch.randn(1, 3, 5, requires_grad=True)
    target = torch.randn(1, 2, 5)
    classes = torch.tensor([[0, 1, 0]])
    target_classes = torch.tensor([[0, 1]])
    valid = torch.tensor([[True, True]])
    loss, components = room_token_transport_loss(
        predicted,
        target,
        predicted_classes=classes,
        target_classes=target_classes,
        target_valid=valid,
    )
    assert torch.isfinite(loss)
    assert "room_transport_entropy" in components
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0
