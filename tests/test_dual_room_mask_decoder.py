import torch

from floorflow.heads.dual_room_mask_decoder import DualPathRoomMaskDecoder


def test_dual_decoder_isolates_mask_and_class_gradients() -> None:
    decoder = DualPathRoomMaskDecoder(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
    )
    output = decoder(torch.randn(1, 8, 16, 16))
    output["mask_logits"].mean().backward()
    assert decoder.mask_branch.query_feature.weight.grad is not None
    assert decoder.class_branch.query_feature.weight.grad is None

    decoder.zero_grad(set_to_none=True)
    output = decoder(torch.randn(1, 8, 16, 16))
    output["class_logits"].mean().backward()
    assert decoder.class_branch.query_feature.weight.grad is not None
    assert decoder.mask_branch.query_feature.weight.grad is None
