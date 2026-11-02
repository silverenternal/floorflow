import torch
from torch.utils.data import TensorDataset

from floorflow.data.mixed_dataset import make_real_synthetic_mixture


def test_real_synthetic_mixture_controls_expected_fraction() -> None:
    real = TensorDataset(torch.zeros(80, 1))
    synthetic = TensorDataset(torch.zeros(20, 1))
    dataset, sampler, spec = make_real_synthetic_mixture(
        real,
        synthetic,
        real_fraction=0.8,
        samples_per_epoch=1000,
        seed=17,
    )
    assert len(dataset) == 100
    assert spec.real_count == 80
    assert spec.synthetic_count == 20
    indices = list(sampler)
    real_fraction = sum(index < 80 for index in indices) / len(indices)
    assert 0.75 <= real_fraction <= 0.85


def test_real_only_is_default() -> None:
    _, sampler, spec = make_real_synthetic_mixture(
        TensorDataset(torch.zeros(16, 1)),
        TensorDataset(torch.zeros(4, 1)),
        samples_per_epoch=32,
    )
    assert spec.real_fraction == 1.0
    assert all(index < 16 for index in list(sampler))
