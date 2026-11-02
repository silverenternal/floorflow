import pytest
import torch

from floorflow.heads.room_mask_decoder import RoomMaskDecoder, room_mask_loss
from floorflow.training.room_mask_finetune import (
    CLASS_PATH_PREFIXES,
    CLASSIFIER_PATH_PREFIXES,
    configure_room_mask_train_scope,
    load_room_mask_decoder_state,
    parameter_snapshot,
    trainable_parameters,
)


def test_class_only_scope_blocks_mask_gradients_and_updates() -> None:
    torch.manual_seed(7)
    decoder = RoomMaskDecoder(
        12,
        3,
        hidden_dim=24,
        num_queries=5,
        layers=1,
        heads=4,
        feature_scales=(1,),
    )
    manifest = configure_room_mask_train_scope(decoder, "class_only")

    assert manifest["scope"] == "class_only"
    assert all(
        name.startswith(CLASS_PATH_PREFIXES)
        for name in manifest["trainable_parameters"]
    )
    assert "query_feature.weight" not in manifest["trainable_parameters"]
    assert "mask_head.0.weight" not in manifest["trainable_parameters"]
    frozen_before = parameter_snapshot(decoder, trainable=False)
    features = torch.randn(1, 12, 8, 8)
    target_classes = torch.tensor([[0, 1]])
    target_masks = torch.zeros(1, 5, 8, 8)
    target_masks[0, 0, :2, :4] = 1.0
    target_masks[0, 1, 4:, 4:] = 1.0
    target_valid = torch.tensor([[True, True, False, False, False]])
    optimizer = torch.optim.AdamW(
        trainable_parameters(decoder.parameters()),
        lr=1.0e-2,
    )

    optimizer.zero_grad(set_to_none=True)
    predictions = decoder(features)
    mask_before = predictions["mask_logits"].detach().clone()
    loss, _components = room_mask_loss(
        predictions,
        target_classes,
        target_masks,
        target_valid,
    )
    loss.backward()

    assert decoder.class_head.weight.grad is not None
    assert decoder.decoder_norm.weight.grad is not None
    assert decoder.mask_features.weight.grad is None
    assert decoder.mask_head[0].weight.grad is None
    assert decoder.query_feature.weight.grad is None
    optimizer.step()

    frozen_after = parameter_snapshot(decoder, trainable=False)
    assert frozen_after.keys() == frozen_before.keys()
    assert all(
        torch.equal(frozen_after[name], frozen_before[name])
        for name in frozen_before
    )
    with torch.no_grad():
        assert torch.equal(decoder(features)["mask_logits"], mask_before)


def test_unknown_train_scope_is_rejected() -> None:
    decoder = RoomMaskDecoder(8, 2, hidden_dim=16, num_queries=4, layers=1)
    with pytest.raises(ValueError, match="unsupported train scope"):
        configure_room_mask_train_scope(decoder, "semantic_only")


def test_classifier_only_scope_freezes_the_shared_normalization() -> None:
    decoder = RoomMaskDecoder(8, 2, hidden_dim=16, num_queries=4, layers=1)
    manifest = configure_room_mask_train_scope(decoder, "classifier_only")
    assert all(
        name.startswith(CLASSIFIER_PATH_PREFIXES)
        for name in manifest["trainable_parameters"]
    )
    assert not decoder.decoder_norm.weight.requires_grad


def test_legacy_room_mask_state_loads_with_zero_class_adapter() -> None:
    decoder = RoomMaskDecoder(8, 2, hidden_dim=16, num_queries=4, layers=1)
    state = {
        name: value
        for name, value in decoder.state_dict().items()
        if not name.startswith("class_fusion.")
    }
    load_room_mask_decoder_state(decoder, state)
    assert decoder.class_fusion.adapter[-1].weight.eq(0).all()
