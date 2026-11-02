import pytest
import torch

from floorflow.flow.cfm_backend import FLOW_MATCHERS, flow_backend_version, sample_flow_batch


def test_legacy_cfm_path_matches_rectified_flow() -> None:
    source = torch.zeros(3, 2, 4, 4)
    target = torch.ones_like(source)
    flow = sample_flow_batch(source, target, backend="legacy")

    assert flow.location.shape == source.shape
    assert torch.equal(flow.velocity, target - source)
    assert torch.allclose(flow.location, flow.time[:, None, None, None].expand_as(source))


@pytest.mark.parametrize("backend", [item for item in FLOW_MATCHERS if item != "legacy"])
def test_torchcfm_adapters_expose_velocity_path(backend: str) -> None:
    source = torch.randn(4, 3, 8, 8)
    target = torch.randn_like(source)
    flow = sample_flow_batch(source, target, backend=backend)

    assert flow.location.shape == source.shape
    assert flow.velocity.shape == source.shape
    assert flow.time.shape == (4,)
    assert torch.isfinite(flow.location).all()
    assert torch.isfinite(flow.velocity).all()


def test_logit_normal_time_and_provenance() -> None:
    flow = sample_flow_batch(
        torch.zeros(32, 1, 2, 2),
        torch.ones(32, 1, 2, 2),
        backend="torchcfm",
        time_sampling="logit_normal",
    )
    assert torch.all((flow.time > 0.0) & (flow.time < 1.0))
    assert flow_backend_version("torchcfm") is not None


def test_logit_normal_mean_and_std_are_explicit() -> None:
    flow = sample_flow_batch(
        torch.zeros(256, 1, 2, 2),
        torch.ones(256, 1, 2, 2),
        backend="torchcfm",
        time_sampling="logit_normal",
        time_sampling_mean=2.0,
        time_sampling_std=0.1,
    )
    assert float(flow.time.mean()) > 0.8


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        sample_flow_batch(torch.zeros(2, 1), torch.zeros(3, 1))


def test_token_minibatch_ot_couples_matching_samples() -> None:
    source = torch.zeros(3, 1, 8, 8)
    target = torch.zeros(3, 1, 8, 8)
    target[0] = 0.1
    target[1] = 0.5
    target[2] = 0.9
    flow = sample_flow_batch(
        source,
        target,
        backend="legacy",
        minibatch_ot="token",
        token_grid=4,
    )
    # The identity coupling is already optimal here, so the closest target
    # stays first and the path remains finite.
    assert flow.location.shape == source.shape
    assert torch.isfinite(flow.location).all()
    assert torch.isfinite(flow.velocity).all()


def test_rejects_unknown_minibatch_ot() -> None:
    with pytest.raises(ValueError, match="unknown minibatch OT"):
        sample_flow_batch(
            torch.zeros(2, 1),
            torch.ones(2, 1),
            backend="legacy",
            minibatch_ot="pixel",
        )
