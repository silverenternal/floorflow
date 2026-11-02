import torch

from floorflow.flow.coarse_to_face import CoarseToFaceFlowHead


def test_coarse_semantics_and_faces_condition_flow_without_shape_drift() -> None:
    head = CoarseToFaceFlowHead(
        feature_channels=8,
        target_channels=4,
        num_room_classes=3,
        hidden_dim=16,
        max_faces=5,
    )
    with torch.no_grad():
        head.output_scale.fill_(1.0)
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    x_t = torch.randn(2, 4, 16, 16)
    coarse = torch.randn(2, 3, 8, 8)
    face_ids = torch.zeros(2, 16, 16, dtype=torch.long)
    face_ids[:, :, 8:] = 2
    residual = head(
        x_t,
        torch.rand(2),
        features,
        torch.ones(2, 16, 16),
        coarse,
        face_ids,
    )
    assert residual.shape == x_t.shape
    residual.sum().backward()
    assert features.grad is not None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in head.parameters()
    )
