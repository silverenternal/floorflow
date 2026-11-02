from __future__ import annotations

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutWall,
)
from floorflow.heads.opening_detection import (
    OpeningDetectionHead,
    opening_detection_loss,
)
from floorflow.heads.structured_objects import (
    StructuredObjectHeads,
    group_ce_dice_loss,
)
from floorflow.inference.opening_center_width_decoder import (
    read_center_width_image_detections,
    read_center_width_opening_detections,
)
from floorflow.inference.opening_heatmap_decoder import (
    encode_opening_features,
    learned_opening_detections,
    opening_detection_maps,
    opening_family_probabilities,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Swing",
            "window:Regular",
            "room:LivingRoom",
            "furniture:Bed",
        ),
        room_map={"LivingRoom": "room:LivingRoom"},
        furniture_map={"Bed": "furniture:Bed"},
        door_map={"Swing": "door:Swing"},
        window_map={"Regular": "window:Regular"},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_opening_head_has_finite_gradients_and_family_probabilities() -> None:
    taxonomy = _taxonomy()
    head = StructuredObjectHeads(8, taxonomy)
    logits = head(torch.randn(1, 8, 16, 16))
    probability = opening_family_probabilities(logits, taxonomy)
    assert probability.shape == (1, 2, 16, 16)
    assert torch.all(probability >= 0.0)
    assert torch.all(probability.sum(dim=1) <= 1.0 + 1.0e-6)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 7:9, 4:10] = taxonomy.class_id("door:Swing")
    loss, _ce, _dice = group_ce_dice_loss(
        logits["opening_logits"],
        target,
        torch.tensor(taxonomy.opening_ids),
        torch.ones(1, 16, 16),
        None,
        logits["opening_background_logits"],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert head.opening.weight.grad is not None
    assert torch.isfinite(head.opening.weight.grad).all()


def test_dedicated_opening_head_has_spatial_gradients() -> None:
    taxonomy = _taxonomy()
    head = OpeningDetectionHead(8, hidden_dim=16)
    features = torch.randn(1, 8, 16, 16)
    outputs = head(features)
    probability = opening_family_probabilities(outputs, taxonomy)
    assert probability.shape == (1, 2, 16, 16)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 7:9, 4:10] = taxonomy.class_id("window:Regular")
    loss, components = opening_detection_loss(
        outputs,
        target,
        torch.ones(1, 16, 16),
        taxonomy,
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "opening_foreground_loss",
        "opening_dice_loss",
        "opening_family_loss",
        "opening_wall_band_focal",
        "opening_wall_band_tversky",
    }
    loss.backward()
    assert head.stem[0].net[0].weight.grad is not None
    assert head.stem[0].net[0].weight.grad.abs().sum() > 0


def test_multiscale_opening_head_shapes_and_gradients() -> None:
    head = OpeningDetectionHead(8, hidden_dim=16, variant="multiscale")
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    outputs = head(features)
    (
        outputs["opening_logits"].mean()
        + outputs["family_logits"].mean()
    ).backward()
    assert outputs["opening_logits"].shape == (2, 1, 16, 16)
    assert outputs["family_logits"].shape == (2, 2, 16, 16)
    assert features.grad is not None


def test_center_width_opening_head_outputs_peaks_and_widths() -> None:
    taxonomy = _taxonomy()
    head = OpeningDetectionHead(8, hidden_dim=16, variant="center_width")
    outputs = head(torch.randn(2, 8, 16, 16))
    maps = opening_detection_maps(outputs, taxonomy)
    assert outputs["opening_center_logits"].shape == (2, 1, 16, 16)
    assert outputs["opening_width_logits"].shape == (2, 1, 16, 16)
    assert maps["center"].shape == (2, 16, 16)
    assert maps["interval_width"].shape == (2, 16, 16)
    target = torch.zeros(2, 16, 16, dtype=torch.long)
    target[:, 7:9, 4:10] = taxonomy.class_id("window:Regular")
    loss, components = opening_detection_loss(
        outputs,
        target,
        torch.ones(2, 16, 16),
        taxonomy,
    )
    assert torch.isfinite(loss)
    assert "opening_center_loss" in components
    assert "opening_width_loss" in components
    loss.backward()


def test_spatial_gap_mita_routes_opening_tokens_to_spatial_memory() -> None:
    head = OpeningDetectionHead(
        8,
        hidden_dim=16,
        variant="center_width",
        spatial_gap_attention=True,
    )
    features = torch.randn(2, 8, 32, 32, requires_grad=True)
    outputs = head(features)
    assert outputs["gap_route_entropy"].ndim == 0
    assert outputs["gap_attention_entropy"].ndim == 0
    assert torch.isfinite(outputs["gap_route_entropy"])
    (
        outputs["opening_center_logits"].square().mean()
        + outputs["opening_width_logits"].square().mean()
    ).backward()
    spatial = head.spatial_gap_attention
    assert spatial is not None
    assert spatial.q_proj.weight.grad is not None
    assert torch.count_nonzero(spatial.q_proj.weight.grad) > 0


def test_center_width_decoder_uses_local_peaks() -> None:
    taxonomy = _taxonomy()
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(
                start=(8.0, 32.0),
                end=(56.0, 32.0),
                thickness=3.0,
            )
        ],
        rooms=[],
        openings=[],
    )
    center = np.zeros((64, 64), dtype=np.float32)
    width = np.zeros_like(center)
    family = np.zeros((2, 64, 64), dtype=np.float32)
    for x in (20, 40):
        center[32, x] = 0.9
        width[32, x] = 8.0 / 64.0
        family[1, 32, x] = 2.0
    width[32, 20] = 2.0
    detections = read_center_width_opening_detections(
        plan,
        center,
        width,
        family,
        taxonomy,
        threshold=0.3,
        min_width=2.0,
    )
    assert len(detections) == 2
    assert {item.family for item in detections} == {"window"}
    assert all(item.width >= 7.0 for item in detections)
    assert all(item.width <= 48.0 for item in detections)

    image_detections = read_center_width_image_detections(
        plan,
        center,
        width,
        family,
        taxonomy,
        threshold=0.3,
        min_width=2.0,
        min_peak_distance=8.0,
    )
    assert len(image_detections) == 2
    assert {item.wall_index for item in image_detections} == {0}
    assert all(item.width <= 48.0 for item in image_detections)


def test_opening_feature_alignment_resizes_image_and_footprint() -> None:
    class RoomEncoder(torch.nn.Module):
        def encode_control(self, footprint, control_image):
            del footprint, control_image
            return torch.zeros(2, 8, 16, 16)

    class FeatureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.room = RoomEncoder()

    features = encode_opening_features(
        FeatureModel(),
        torch.ones(2, 32, 32),
        torch.zeros(2, 3, 32, 32),
        include_image=True,
    )
    assert features.shape == (2, 12, 16, 16)


def test_family_heatmap_reads_multiple_openings_on_one_wall() -> None:
    taxonomy = _taxonomy()
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(8.0, 32.0), end=(56.0, 32.0), thickness=3.0),
        ],
        rooms=[],
        openings=[],
    )
    probability = np.zeros((2, 64, 64), dtype=np.float32)
    probability[1, 31:34, 14:21] = 0.95
    probability[1, 31:34, 39:47] = 0.95
    detections = learned_opening_detections(
        plan,
        probability,
        taxonomy,
        threshold=0.30,
        min_area=2,
        max_wall_distance=4.0,
    )
    assert len(detections) == 2
    assert {detection.family for detection in detections} == {"window"}
    assert {detection.wall_index for detection in detections} == {0}
