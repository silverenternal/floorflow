import pytest
import torch

from floorflow.objectives.triton_ops import (
    endpoint_incidence,
    triton_available,
    triton_status,
    triton_vq_indices,
    triton_vq_quantize,
)


def test_triton_ops_keep_cpu_fallbacks() -> None:
    torch.manual_seed(7)
    values = torch.randn(5, 4)
    codebook = torch.randn(3, 4)
    expected = torch.cdist(values, codebook).argmin(dim=-1)
    assert torch.equal(triton_vq_indices(values, codebook), expected)
    quantized, indices, loss = triton_vq_quantize(values, codebook)
    assert quantized.shape == values.shape
    assert indices.shape == values.shape[:-1]
    assert torch.isfinite(loss)
    points = torch.zeros(1, 2, 2)
    endpoints = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    incidence = endpoint_incidence(points, endpoints)
    assert incidence.shape == (1, 2, 2)
    assert incidence[0, 0, 0] > incidence[0, 0, 1]
    assert triton_status()["available"] == triton_available()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="CUDA and Triton are required",
)
def test_triton_kernels_match_torch_on_cuda() -> None:
    device = torch.device("cuda")
    torch.manual_seed(11)
    values = torch.randn(7, 16, 32, device=device)
    codebook = torch.randn(9, 32, device=device)
    expected = torch.cdist(values, codebook).argmin(dim=-1)
    actual = triton_vq_indices(values, codebook, enabled=True)
    assert torch.equal(actual, expected.reshape(-1))
    points = torch.rand(2, 3, 2, device=device)
    endpoints = torch.rand(2, 5, 2, device=device)
    expected_incidence = endpoint_incidence(
        points,
        endpoints,
        enabled=False,
    )
    actual_incidence = endpoint_incidence(
        points,
        endpoints,
        enabled=True,
    )
    assert torch.allclose(actual_incidence, expected_incidence, atol=1.0e-5)
