import torch

from floorflow.geometry.diff_ras import diff_soft_polygon_masks


def _square():
    return torch.tensor(
        [[[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]]],
        dtype=torch.float32,
    )


def test_diff_ras_renders_inside_and_outside() -> None:
    mask = diff_soft_polygon_masks(_square(), 16)
    assert mask.shape == (1, 1, 16, 16)
    assert float(mask[0, 0, 8, 8]) > 0.99
    assert float(mask[0, 0, 0, 0]) < 0.01


def test_diff_ras_supports_3d_queries_and_gradients() -> None:
    polygons = _square()[0].clone().requires_grad_(True)
    mask = diff_soft_polygon_masks(polygons, 12)
    assert mask.shape == (1, 12, 12)
    (mask.sum() - 50.0).pow(2).backward()
    assert polygons.grad is not None
    assert torch.isfinite(polygons.grad).all()
