from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.raster_control_bank import (
    RasterInstanceControl,
    RasterInstanceSoftRoutedAdapter,
)
from floorflow.heads.room_mask_decoder import (
    RoomMaskDecoder,
    fused_room_mask_logits,
    position_embedding_2d,
    room_mask_loss,
    room_mask_metrics,
    room_mask_partition_loss,
    room_mask_targets,
)
from floorflow.heads.room_mask_fusion import (
    apply_program_class_prior,
    mask_region_class_logits,
    render_room_masks,
)
from floorflow.objectives import room_query_matching as room_query_matching_module
from floorflow.objectives.room_query_matching import RoomQueryMatch


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_room_mask_decoder_loss_and_metrics() -> None:
    taxonomy = _taxonomy()
    decoder = RoomMaskDecoder(16, len(taxonomy.room_ids), hidden_dim=32, num_queries=8, layers=2)
    predictions = decoder(torch.randn(1, 16, 32, 32))
    predictions["prior_logits"] = torch.zeros_like(predictions["mask_logits"][:, : len(taxonomy.room_ids)])
    prior = torch.randn_like(predictions["prior_logits"])
    predictions["prior_logits"] = prior
    assert torch.equal(
        fused_room_mask_logits(
            predictions,
            len(taxonomy.room_ids),
            fusion_mode="legacy",
        ).argmax(dim=1),
        prior.argmax(dim=1),
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, 4:16, 4:16] = taxonomy.class_id("room:Bedroom")
    labels[:, 16:28, 16:28] = taxonomy.class_id("room:Bath")
    target_classes, target_masks, target_valid = room_mask_targets(labels, taxonomy, max_rooms=8)
    loss, components = room_mask_loss(predictions, target_classes, target_masks, target_valid)
    assert torch.isfinite(loss)
    assert "room_mask_loss" in components
    assert "room_mask_program_loss" in components
    assert any(name.startswith("room_mask_aux_") for name in components)
    loss.backward()
    assert decoder.class_head.weight.grad is not None
    metrics = room_mask_metrics(
        predictions,
        target_classes,
        target_masks,
        target_valid,
        taxonomy,
        labels,
        torch.ones(1, 32, 32),
    )
    assert 0.0 <= metrics["mask_iou"] <= 1.0
    assert 0.0 <= metrics["rendered_room_macro_iou"] <= 1.0
    footprint = torch.ones(1, 32, 32)
    partition_loss, partition_parts = room_mask_partition_loss(
        predictions,
        labels,
        footprint,
        taxonomy,
        pixel_weights=torch.ones_like(footprint),
    )
    assert torch.isfinite(partition_loss)
    assert "room_mask_partition_ce" in partition_parts


def test_room_mask_partition_loss_handles_no_room_targets() -> None:
    taxonomy = _taxonomy()
    predictions = {
        "mask_logits": torch.randn(1, 2, 4, 4, requires_grad=True),
        "class_logits": torch.randn(
            1,
            2,
            len(taxonomy.room_ids) + 1,
            requires_grad=True,
        ),
    }
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    loss, components = room_mask_partition_loss(
        predictions,
        target,
        torch.ones_like(target, dtype=torch.float32),
        taxonomy,
    )
    assert loss.item() == 0.0
    assert all(value.item() == 0.0 for value in components.values())
    loss.backward()
    assert predictions["mask_logits"].grad is not None


def test_position_embedding_is_cached_and_projection_pooling_commutes() -> None:
    first = position_embedding_2d(16, 16, 32, torch.device("cpu"))
    second = position_embedding_2d(16, 16, 32, torch.device("cpu"))
    assert first is second
    decoder = RoomMaskDecoder(
        4,
        2,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        feature_scales=(1, 2),
    )
    features = torch.randn(1, 4, 16, 16)
    projected = decoder.mask_features(features)
    expected = decoder.mask_features(
        torch.nn.functional.avg_pool2d(features, 2, stride=2)
    )
    actual = torch.nn.functional.avg_pool2d(
        projected,
        2,
        stride=2,
    )
    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)


def test_raster_control_exposes_owner_readout_and_soft_train_gate() -> None:
    taxonomy = _taxonomy()
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        query_set_head=True,
        query_count_gate=True,
    )
    footprint = torch.ones(2, 32, 32)
    image = torch.randn(2, 3, 32, 32)
    model.train()
    training = model(footprint, control_image=image)
    assert training["query_count_selection_mask"].is_floating_point()
    assert training["owner_map"].shape == (2, 32, 32)
    assert training["owner_boundary"].dtype == torch.bool
    model.eval()
    evaluation = model(footprint, control_image=image)
    assert evaluation["query_count_selection_mask"].dtype == torch.bool
    contract = model.module_contract()
    assert contract["ordered_stages"][-3:] == [
        "class_mass_fusion",
        "full_resolution_room_readout",
        "query_owner_readout",
    ]
    assert contract["train_count_routing"] == "soft"
    assert contract["eval_count_routing"] == "hard"


def test_room_mask_loss_uses_shared_cross_head_matches(monkeypatch) -> None:
    predictions = {
        "class_logits": torch.tensor(
            [[[4.0, 0.0], [0.0, 4.0], [0.0, 0.0]]]
        ),
        "mask_logits": torch.stack(
            [
                torch.full((4, 4), -8.0),
                torch.full((4, 4), 8.0),
            ]
        )[None],
    }
    target_classes = torch.tensor([[0, 1]])
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :, :2] = 1.0
    target_masks[0, 1, :, 2:] = 1.0
    target_valid = torch.tensor([[True, True]])
    shared = [
        RoomQueryMatch(
            predicted_indices=[0, 1],
            target_indices=[1, 0],
        )
    ]

    def fail_if_resolved(*_args, **_kwargs):
        raise AssertionError("shared matches must not be re-solved")

    monkeypatch.setattr(
        room_query_matching_module,
        "exact_assignment_batch",
        fail_if_resolved,
    )
    loss, components = room_mask_loss(
        predictions,
        target_classes,
        target_masks,
        target_valid,
        auxiliary_weight=0.0,
        matches=shared,
    )
    assert torch.isfinite(loss)
    assert components["room_mask_class_loss"] > 0.0


def test_room_mask_program_loss_is_finite_and_reaches_class_logits() -> None:
    decoder = RoomMaskDecoder(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        feature_scales=(1,),
    )
    predictions = decoder(torch.randn(1, 8, 8, 8))
    target_classes = torch.tensor([[0, 2, 0, 0]])
    target_masks = torch.zeros(1, 4, 8, 8)
    target_valid = torch.tensor([[True, True, False, False]])
    loss, components = room_mask_loss(
        predictions,
        target_classes,
        target_masks,
        target_valid,
        mask_loss_weight=0.0,
        class_loss_weight=0.0,
        program_loss_weight=1.0,
        auxiliary_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert components["room_mask_program_loss"] >= 0
    loss.backward()
    assert decoder.class_head.weight.grad is not None


def test_room_mask_loss_exposes_query_separation_and_partition_contracts() -> None:
    decoder = RoomMaskDecoder(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        feature_scales=(1,),
        partition_consistent=True,
    )
    predictions = decoder(torch.randn(1, 8, 8, 8))
    target_classes = torch.tensor([[0, 1, 2, 0]])
    target_masks = torch.zeros(1, 4, 8, 8)
    target_masks[0, 0, :, :4] = 1.0
    target_masks[0, 1, :, 4:] = 1.0
    target_valid = torch.tensor([[True, True, False, False]])
    loss, components = room_mask_loss(
        predictions,
        target_classes,
        target_masks,
        target_valid,
        auxiliary_weight=0.0,
        partition_consistency_weight=0.25,
        query_separation_weight=0.25,
        supervised_query_separation=True,
    )
    assert torch.isfinite(loss)
    assert "room_mask_partition_query_loss" in components
    assert "room_query_separation_loss" in components
    loss.backward()
    assert decoder.mask_head[-1].weight.grad is not None


def test_invalid_room_memory_tokens_cannot_change_decoder_output() -> None:
    decoder = RoomMaskDecoder(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=2,
        heads=4,
        feature_scales=(1,),
    ).eval()
    features = torch.randn(1, 8, 8, 8)
    room_tokens = torch.randn(1, 4, 16)
    room_valid = torch.tensor([[True, True, False, False]])
    reference = decoder(
        features,
        room_tokens=room_tokens,
        room_token_valid=room_valid,
    )
    changed = room_tokens.clone()
    changed[:, 2:] = 1.0e4
    candidate = decoder(
        features,
        room_tokens=changed,
        room_token_valid=room_valid,
    )
    assert torch.allclose(reference["mask_logits"], candidate["mask_logits"])


def test_existence_gate_removes_background_query_mask_mass() -> None:
    predictions = {
        "mask_logits": torch.zeros(1, 2, 8, 8),
        "class_logits": torch.tensor(
            [
                [
                    [6.0, 0.0, -4.0],
                    [0.0, 0.0, 6.0],
                ]
            ]
        ),
        "query_existence_logits": torch.tensor([[6.0, -6.0]]),
    }
    plain = fused_room_mask_logits(
        predictions,
        2,
        fusion_mode="sigmoid",
    ).softmax(dim=1)
    gated = fused_room_mask_logits(
        predictions,
        2,
        fusion_mode="sigmoid_existence",
    ).softmax(dim=1)
    assert gated[0, 0].mean() > plain[0, 0].mean() + 0.2
    assert gated[0, 1].mean() < 0.01


def test_query_count_selection_mask_removes_unselected_queries() -> None:
    predictions = {
        "mask_logits": torch.zeros(1, 2, 8, 8),
        "class_logits": torch.tensor(
            [
                [
                    [6.0, 0.0, -4.0],
                    [0.0, 6.0, -4.0],
                ]
            ]
        ),
        "query_count_selection_mask": torch.tensor([[True, False]]),
    }
    fused = fused_room_mask_logits(
        predictions,
        2,
        fusion_mode="sigmoid",
    ).softmax(dim=1)
    assert fused[0, 0].mean() > 0.9
    assert fused[0, 1].mean() < 0.1


def test_render_prefers_the_model_fused_readout() -> None:
    taxonomy = _taxonomy()
    room_ids = torch.tensor(taxonomy.room_ids)
    predictions = {
        "mask_logits": torch.zeros(1, 2, 4, 4),
        "class_logits": torch.zeros(1, 2, 3),
        "fused_logits": torch.zeros(1, len(room_ids), 4, 4),
    }
    predictions["fused_logits"][0, 1] = 8.0
    rendered = render_room_masks(
        predictions,
        taxonomy,
        torch.ones(1, 4, 4),
    )
    assert (rendered == int(room_ids[1])).all()


def test_existence_and_unmatched_mask_losses_reach_queries() -> None:
    mask_logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
    existence_logits = torch.zeros(1, 2, requires_grad=True)
    predictions = {
        "class_logits": torch.zeros(1, 2, 2, requires_grad=True),
        "mask_logits": mask_logits,
        "query_existence_logits": existence_logits,
    }
    classes = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.zeros(1, 1, 8, 8)
    masks[0, 0, 2:6, 2:6] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    loss, components = room_mask_loss(
        predictions,
        classes,
        masks,
        valid,
        unmatched_mask_weight=0.5,
        existence_loss_weight=0.5,
    )
    loss.backward()
    assert components["room_mask_unmatched_loss"] > 0.0
    assert components["room_mask_existence_loss"] > 0.0
    assert existence_logits.grad is not None
    assert mask_logits.grad is not None


def test_sigmoid_query_fusion_preserves_independent_queries() -> None:
    mask_logits = torch.full((1, 2, 4, 4), -12.0)
    mask_logits[:, 0, :2] = 12.0
    mask_logits[:, 1, 2:] = 12.0
    class_logits = torch.full((1, 2, 3), -12.0)
    class_logits[0, 0, 0] = 12.0
    class_logits[0, 1, 1] = 12.0
    predictions = {"mask_logits": mask_logits, "class_logits": class_logits}
    fused = fused_room_mask_logits(
        predictions,
        2,
        fusion_mode="sigmoid",
    )
    assert fused[0, :, 0, 0].argmax().item() == 0
    assert fused[0, :, 3, 3].argmax().item() == 1


def test_query_fusion_modes_preserve_overlapping_class_mixtures() -> None:
    mask_logits = torch.full((1, 2, 2, 2), 8.0)
    class_logits = torch.full((1, 2, 3), -12.0)
    class_logits[0, 0, 0] = 12.0
    class_logits[0, 1, 1] = 12.0
    predictions = {"mask_logits": mask_logits, "class_logits": class_logits}
    mixture = fused_room_mask_logits(predictions, 2, fusion_mode="query_mixture")
    hard = fused_room_mask_logits(predictions, 2, fusion_mode="hard_query")
    assert torch.allclose(mixture[0, 0], mixture[0, 1], atol=1.0e-5)
    assert torch.allclose(hard[0, 0], hard[0, 1], atol=1.0e-5)


def test_winner_query_assigns_overlap_to_highest_scoring_instance() -> None:
    mask_logits = torch.full((1, 2, 2, 2), -12.0)
    mask_logits[0, 0] = 2.0
    mask_logits[0, 1, :, 1] = 2.0
    class_logits = torch.full((1, 2, 3), -12.0)
    class_logits[0, 0, 0] = 8.0
    class_logits[0, 1, 1] = 12.0
    predictions = {"mask_logits": mask_logits, "class_logits": class_logits}
    fused = fused_room_mask_logits(predictions, 2, fusion_mode="winner_query")
    assert fused[0, :, 0, 0].argmax().item() == 0
    assert fused[0, :, 0, 1].argmax().item() == 1


def test_program_class_prior_is_centered_and_leaves_no_object_unchanged() -> None:
    class_logits = torch.zeros(1, 2, 4)
    program = torch.tensor([[5.0, 1.0, 3.0]])
    adjusted = apply_program_class_prior(class_logits, program, scale=0.5)
    assert adjusted[0, :, -1].eq(0).all()
    assert torch.isclose(adjusted[0, 0, :3].mean(), torch.tensor(0.0))


def test_mask_regions_take_classes_from_external_logits() -> None:
    mask_logits = torch.full((1, 2, 6, 6), -12.0)
    mask_logits[0, 0, :, :3] = 12.0
    mask_logits[0, 1, :, 3:] = 12.0
    predictions = {"mask_logits": mask_logits}
    external = torch.zeros(1, 2, 6, 6)
    external[0, 1, :, :3] = 4.0
    external[0, 0, :, 3:] = 4.0
    output = mask_region_class_logits(
        predictions,
        external,
        torch.ones(1, 6, 6),
    )
    assert output[0, :, :, :3].argmax(dim=0).eq(1).all()
    assert output[0, :, :, 3:].argmax(dim=0).eq(0).all()


def test_full_resolution_query_masks_and_soft_route_are_usable() -> None:
    taxonomy = _taxonomy()
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=8,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        full_resolution_masks=True,
        full_resolution_channels=8,
    )
    footprint = torch.ones(2, 32, 32)
    image = torch.randn(2, 3, 32, 32)
    outputs = model(footprint, control_image=image)
    assert outputs["mask_logits"].shape == (2, 4, 32, 32)
    assert outputs["fused_logits"].shape == (2, len(taxonomy.room_ids), 32, 32)
    outputs["mask_logits"].mean().backward()
    assert model.mask_refiner.output.weight.grad is not None
    routed, dense = RasterInstanceSoftRoutedAdapter(model).forward_with_dense(
        footprint,
        control_image=image,
    )
    assert routed.shape == dense.shape
    assert torch.isfinite(routed).all()


def test_program_tokens_condition_query_classes_and_receive_gradient() -> None:
    taxonomy = _taxonomy()
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=8,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        room_program_head=True,
        program_token_conditioning=True,
    ).eval()
    assert model.mask_decoder.class_fusion.program_attention is not None
    with torch.no_grad():
        model.mask_decoder.class_fusion.program_gate.fill_(0.2)
    outputs = model(
        torch.ones(1, 32, 32),
        control_image=torch.randn(1, 3, 32, 32),
    )
    outputs["class_logits"].square().mean().backward()
    assert model.room_program_head is not None
    assert model.room_program_head.queries.grad is not None
    assert model.mask_decoder.class_fusion.program_projection is not None
    assert model.mask_decoder.class_fusion.program_attention.in_proj_weight.grad is not None


def test_program_class_prior_reaches_query_classes() -> None:
    taxonomy = _taxonomy()
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=8,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        room_program_head=True,
        program_class_prior_weight=0.25,
    ).eval()
    outputs = model(
        torch.ones(1, 32, 32),
        control_image=torch.randn(1, 3, 32, 32),
    )
    assert outputs["program_class_prior_weight"].item() == 0.25
    outputs["class_logits"].square().mean().backward()
    assert model.room_program_head is not None
    assert model.room_program_head.presence_class_head.weight.grad is not None


def test_dual_control_projection_starts_as_native_identity() -> None:
    taxonomy = _taxonomy()
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=8,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        dual_control=True,
    ).eval()
    footprint = torch.ones(1, 16, 16)
    native = torch.randn(1, 3, 16, 16)
    predicted = torch.randn(1, 3, 16, 16)
    dual = torch.cat([native, predicted], dim=1)
    with torch.no_grad():
        expected = model.room.encode_control(
            footprint,
            control_image=native,
        )
        actual = model.room.encode_control(
            footprint,
            control_image=native + model.control_projection(dual),
        )
    assert torch.allclose(expected, actual)


def test_partition_loss_uses_the_final_presence_fusion_contract() -> None:
    taxonomy = _taxonomy()
    masks = torch.full((1, 2, 8, 8), -8.0)
    masks[:, 0, :, :4] = 8.0
    masks[:, 1, :, 4:] = 8.0
    classes = torch.full((1, 2, 3), -4.0)
    classes[:, 0, 0] = 4.0
    classes[:, 1, 1] = 4.0
    predictions = {
        "mask_logits": masks,
        "class_logits": classes,
        "query_existence_logits": torch.tensor([[8.0, 8.0]]),
    }
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, :4] = taxonomy.room_ids[0]
    target[:, :, 4:] = taxonomy.room_ids[1]
    footprint = torch.ones(1, 8, 8)
    enabled = room_mask_partition_loss(predictions, target, footprint, taxonomy)[0]
    predictions["query_existence_logits"] = torch.tensor([[8.0, -8.0]])
    suppressed = room_mask_partition_loss(predictions, target, footprint, taxonomy)[0]
    assert suppressed > enabled
