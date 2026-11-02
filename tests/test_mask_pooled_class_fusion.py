import torch

from floorflow.heads.mask_pooled_class_fusion import MaskPooledClassFusion


def test_mask_pooled_class_fusion_is_identity_at_initialization() -> None:
    fusion = MaskPooledClassFusion(8, 6, 3, hidden_dim=16, pool_size=4)
    queries = torch.randn(2, 5, 8)
    class_logits = torch.randn(2, 5, 4)
    masks = torch.randn(2, 5, 16, 16)
    features = torch.randn(2, 6, 16, 16)
    fused, diagnostics = fusion(queries, masks, class_logits, features)
    assert torch.equal(fused, class_logits)
    assert diagnostics["pooled_feature_norm"] >= 0


def test_mask_pooled_class_fusion_uses_region_specific_evidence() -> None:
    fusion = MaskPooledClassFusion(4, 3, 2, hidden_dim=8, pool_size=2)
    with torch.no_grad():
        fusion.output.weight.normal_()
        fusion.output.bias.zero_()
    queries = torch.zeros(1, 2, 4)
    masks = torch.tensor(
        [
            [
                [[8.0] * 4 for _ in range(4)],
                [[-8.0] * 4 for _ in range(4)],
            ]
        ]
    )
    classes = torch.zeros(1, 2, 3)
    first = torch.zeros(1, 3, 4, 4)
    first[:, 0] = 2.0
    second = torch.zeros(1, 3, 4, 4)
    second[:, 1] = 2.0
    first_logits, _ = fusion(queries, masks, classes, first)
    second_logits, _ = fusion(queries, masks, classes, second)
    assert not torch.equal(first_logits, second_logits)
    first_logits.square().mean().backward()
    assert fusion.feature_projection.weight.grad is not None
    assert fusion.feature_projection.weight.grad.abs().sum() > 0
