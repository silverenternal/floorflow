import numpy as np
import torch

from floorflow.data.raster_control import raster_control_image
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.convnext_room_control import ConvNeXtRoomControl
from floorflow.heads.raster_control_bank import (
    RasterDenseQueryRoomBlend,
    RasterInstanceClassRoutedAdapter,
    RasterInstanceControl,
    RasterInstanceRoomAdapter,
)
from floorflow.heads.room_segmenter import ContourRoomSegmenter, prepare_room_condition
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)


def test_contour_room_segmenter_shapes_and_gradients() -> None:
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    condition = prepare_room_condition(footprint)
    assert condition.shape == (1, 3, 64, 64)
    model = ContourRoomSegmenter(5, pretrained=False, decoder_channels=16)
    logits = model(footprint)
    assert logits.shape == (1, 5, 64, 64)
    logits.mean().backward()
    assert model.final[-1].weight.grad is not None


def test_room_segmenter_accepts_non_room_raster_control() -> None:
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    control = torch.zeros(1, 3, 64, 64)
    model = ContourRoomSegmenter(5, pretrained=False, decoder_channels=16)
    logits = model(footprint, control_image=control)
    assert logits.shape == (1, 5, 64, 64)


def test_raster_control_excludes_room_identity_pixels() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room_id = taxonomy.class_id("room:LivingRoom")
    wall_id = taxonomy.class_id("wall")
    labels = np.full((8, 8), room_id, dtype=np.uint8)
    labels[:, 4] = wall_id
    footprint = np.ones((8, 8), dtype=bool)
    first = raster_control_image(labels, footprint, taxonomy)
    other_room = labels.copy()
    other_room[other_room == room_id] = taxonomy.class_id("room:Kitchen")
    second = raster_control_image(other_room, footprint, taxonomy)
    assert first.shape == (3, 8, 8)
    assert np.array_equal(first, second)


def test_room_segmenter_prior_restores_and_predicts(tmp_path) -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    model = ContourRoomSegmenter(
        len(taxonomy.room_ids),
        pretrained=False,
        decoder_channels=16,
    )
    checkpoint = tmp_path / "segmenter.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": {"decoder_channels": 16},
        },
        checkpoint,
    )
    restored, config = load_room_segmenter_prior(
        checkpoint,
        taxonomy,
        torch.device("cpu"),
    )
    footprint = torch.zeros(1, 1, 64, 64)
    footprint[:, :, 8:56, 8:56] = 1.0
    logits = predict_room_segmenter_prior(restored, footprint)
    assert config["decoder_channels"] == 16
    assert logits.shape == (1, len(taxonomy.room_ids), 64, 64)


def test_raster_instance_control_outputs_dense_and_instance_rooms() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    control = torch.zeros(1, 3, 64, 64)
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
    )
    outputs = model(footprint, control_image=control)
    assert outputs["dense_logits"].shape == (
        1,
        len(taxonomy.room_ids),
        64,
        64,
    )
    assert outputs["mask_logits"].shape == (
        1,
        4,
        64,
        64,
    )
    assert outputs["fused_logits"].shape == outputs["dense_logits"].shape


def test_adaptive_query_reference_is_wired_into_room_instance_model() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    control = torch.zeros(1, 3, 64, 64)
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
        adaptive_query_refinement=True,
    )
    outputs = model(footprint, control_image=control)
    diagnostics = outputs["adaptive_query_diagnostics"]
    assert diagnostics is not None
    assert diagnostics["reference_points"].shape == (1, 4, 2)
    outputs["mask_logits"].square().mean().backward()
    assert model.mask_decoder.adaptive_query_refiner is not None
    assert (
        model.mask_decoder.adaptive_query_refiner.feature_projection.weight.grad
        is not None
    )


def test_room_head_suite_is_wired_with_zero_initialized_extensions() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    control = torch.zeros(1, 3, 64, 64)
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
        adaptive_query_refinement=True,
        mask_pooled_class_fusion=True,
        full_resolution_class_sampler=True,
        owner_semantic_reassignment=True,
        room_boundary_owner_refiner=True,
        query_set_head=True,
        dense_identity_refiner=True,
        room_program_head=True,
    )
    outputs = model(footprint, control_image=control)
    assert outputs["room_count_logits"].shape == (1, 5)
    assert outputs["room_class_count_logits"].shape == (
        1,
        len(taxonomy.room_ids),
        5,
    )
    assert outputs["room_program_cardinality_logits"].shape[-1] == (
        len(taxonomy.room_ids) + 1
    )
    assert outputs["mask_pooled_class_diagnostics"]["residual_norm"] == 0
    assert (
        outputs["full_resolution_class_sampler_diagnostics"]["residual_norm"]
        == 0
    )
    assert "owner_semantic_reassignment_diagnostics" in outputs
    assert outputs["room_boundary_logits"].shape == (
        outputs["mask_logits"].shape[0],
        *outputs["mask_logits"].shape[-2:],
    )
    assert model.dense_identity_refiner is not None
    outputs["fused_logits"].square().mean().backward()
    assert model.mask_pooled_class_fusion.feature_projection.weight.grad is not None
    assert (
        model.full_resolution_class_sampler.feature_projection.weight.grad
        is not None
    )


def test_raster_instance_room_adapter_blends_existing_branches() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    control = torch.zeros(1, 3, 64, 64)
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
    )
    model.eval()
    outputs = model(footprint, control_image=control)

    class FixedOutputs(torch.nn.Module):
        def forward(self, footprint, control_image=None):
            del footprint, control_image
            return outputs

    dense_adapter = RasterInstanceRoomAdapter(FixedOutputs(), 0.0)
    query_adapter = RasterInstanceRoomAdapter(FixedOutputs(), 1.0)
    dense = dense_adapter(footprint, control_image=control)
    query = query_adapter(footprint, control_image=control)
    assert torch.allclose(
        dense,
        outputs["dense_logits"].softmax(dim=1).clamp_min(1.0e-8).log(),
    )
    assert torch.allclose(
        query,
        outputs["fused_logits"].softmax(dim=1).clamp_min(1.0e-8).log(),
    )


def test_dense_query_blend_consumes_independent_checkpoints() -> None:
    class Dense(torch.nn.Module):
        def forward(self, footprint, control_image=None):
            del footprint, control_image
            return torch.tensor(
                [[[[0.0, 4.0], [0.0, 4.0]], [[4.0, 0.0], [4.0, 0.0]]]],
                dtype=torch.float32,
            )

    class Query(torch.nn.Module):
        def forward(self, footprint, control_image=None):
            del footprint, control_image
            logits = torch.tensor(
                [[[[4.0, 0.0], [4.0, 0.0]], [[0.0, 4.0], [0.0, 4.0]]]],
                dtype=torch.float32,
            )
            return {"fused_logits": logits}

    footprint = torch.ones(1, 2, 2)
    adapter = RasterDenseQueryRoomBlend(Dense(), Query(), blend=0.5)
    blended, dense = adapter.forward_with_dense(footprint)
    expected = 0.5 * dense.softmax(dim=1) + 0.5 * torch.tensor(
        [[[[4.0, 0.0], [4.0, 0.0]], [[0.0, 4.0], [0.0, 4.0]]]],
        dtype=torch.float32,
    ).softmax(dim=1)
    assert torch.allclose(blended, expected.clamp_min(1.0e-8).log())
    assert blended.shape == (1, 2, 2, 2)


def test_raster_instance_class_router_preserves_dense_logit_shape() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    control = torch.zeros(1, 3, 64, 64)
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
    )
    adapter = RasterInstanceClassRoutedAdapter(model, threshold=0.5, strength=4.0)
    output = adapter(footprint, control_image=control)
    assert output.shape == (
        1,
        len(taxonomy.room_ids),
        64,
        64,
    )


def test_convnext_room_control_shapes() -> None:
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    model = ConvNeXtRoomControl(
        5,
        pretrained=False,
        decoder_channels=16,
    )
    logits = model(footprint)
    assert logits.shape == (1, 5, 64, 64)
