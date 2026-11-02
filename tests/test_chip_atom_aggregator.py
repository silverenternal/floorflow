from __future__ import annotations

import torch

from floorflow.chip_flow.structured_heads import AtomResidualAggregator


def _inputs(aggregator: AtomResidualAggregator) -> tuple:
    return (
        torch.randn(1, 5, 32),
        torch.randn(1, 4),
        torch.randn(1, 9, 32),
        torch.ones(1, 9, dtype=torch.bool),
        torch.randn(1, 3, 32),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([0.5]),
    )


def test_atom_residual_aggregator_starts_as_exact_sum_and_learns_gates() -> None:
    torch.manual_seed(37)
    aggregator = AtomResidualAggregator(32)
    residuals = torch.randn(2, aggregator.atom_count, 5, 3)
    block_tokens = torch.randn(2, 5, 32)
    structural = torch.randn(2, 9, 32)
    ports = torch.randn(2, 3, 32)
    global_prediction = torch.randn(2, 4)

    initial, diagnostics = aggregator(
        residuals,
        block_tokens,
        global_prediction,
        structural,
        torch.ones(2, 9, dtype=torch.bool),
        ports,
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([0.5, 0.5]),
    )

    torch.testing.assert_close(initial, residuals.sum(dim=1))
    assert diagnostics["atom_aggregator_gain"].item() == 0.0
    with torch.no_grad():
        aggregator.mix_gate.fill_(0.5)
    learned, diagnostics = aggregator(
        residuals,
        block_tokens,
        global_prediction,
        structural,
        torch.ones(2, 9, dtype=torch.bool),
        ports,
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([0.5, 0.5]),
    )
    assert not torch.allclose(initial, learned)
    learned.square().mean().backward()
    assert aggregator.mix_gate.grad is not None
    assert aggregator.residual_encoder[-1].weight.grad is not None
    assert diagnostics["atom_aggregator_gate"].shape == (
        2,
        aggregator.atom_count,
        5,
    )
    assert diagnostics["atom_aggregator_residual_rms"].shape == (
        2,
        aggregator.atom_count,
        5,
    )
    with torch.no_grad():
        aggregator.mix_gate.fill_(100.0)
    _extreme, diagnostics = aggregator(
        residuals,
        block_tokens,
        global_prediction,
        structural,
        torch.ones(2, 9, dtype=torch.bool),
        ports,
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([0.5, 0.5]),
    )
    assert diagnostics["atom_aggregator_gate"].min() > 0.0


def test_atom_residual_aggregator_is_block_permutation_equivariant() -> None:
    torch.manual_seed(41)
    aggregator = AtomResidualAggregator(32)
    with torch.no_grad():
        aggregator.mix_gate.fill_(0.25)
    residuals = torch.randn(1, aggregator.atom_count, 5, 3)
    block_tokens = torch.randn(1, 5, 32)
    global_prediction = torch.randn(1, 4)
    structural = torch.randn(1, 9, 32)
    ports = torch.randn(1, 3, 32)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    expected, _ = aggregator(
        residuals,
        block_tokens,
        global_prediction,
        structural,
        torch.ones(1, 9, dtype=torch.bool),
        ports,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([0.5]),
    )
    actual, _ = aggregator(
        residuals[:, :, permutation],
        block_tokens[:, permutation],
        global_prediction,
        structural,
        torch.ones(1, 9, dtype=torch.bool),
        ports,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([0.5]),
    )

    torch.testing.assert_close(
        actual,
        expected[:, permutation],
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_atom_residual_aggregator_ignores_unavailable_branches() -> None:
    aggregator = AtomResidualAggregator(32)
    residuals = torch.zeros(1, aggregator.atom_count, 5, 3)
    branch_mask = torch.ones(1, aggregator.atom_count, dtype=torch.bool)
    branch_mask[:, 0] = False
    common = _inputs(aggregator)

    baseline, _ = aggregator(
        residuals,
        *common,
        branch_mask=branch_mask,
    )
    residuals[:, 0] = 10.0
    changed, diagnostics = aggregator(
        residuals,
        *common,
        branch_mask=branch_mask,
    )

    assert torch.count_nonzero(baseline) == 0
    torch.testing.assert_close(changed, baseline)
    torch.testing.assert_close(
        diagnostics["atom_aggregator_active_fraction"],
        torch.tensor((aggregator.atom_count - 1) / aggregator.atom_count),
    )
    assert torch.isfinite(diagnostics["atom_aggregator_gate_min_positive"])
    none_active, diagnostics = aggregator(
        residuals,
        *common,
        branch_mask=torch.zeros(1, aggregator.atom_count, dtype=torch.bool),
    )
    assert torch.count_nonzero(none_active) == 0
    assert diagnostics["atom_aggregator_active_fraction"] == 0
    assert diagnostics["atom_aggregator_gate_min_positive"] == 0


def test_inactive_atom_token_cannot_contaminate_active_atom_attention() -> None:
    torch.manual_seed(43)
    aggregator = AtomResidualAggregator(32)
    with torch.no_grad():
        aggregator.mix_gate.fill_(0.5)
    residuals = torch.randn(1, aggregator.atom_count, 5, 3)
    branch_mask = torch.ones(1, aggregator.atom_count, dtype=torch.bool)
    branch_mask[:, 0] = False
    common = _inputs(aggregator)

    baseline, baseline_diagnostics = aggregator(
        residuals,
        *common,
        branch_mask=branch_mask,
    )
    with torch.no_grad():
        aggregator.type_embedding[0].add_(10.0)
    changed, diagnostics = aggregator(
        residuals,
        *common,
        branch_mask=branch_mask,
    )

    torch.testing.assert_close(changed, baseline)
    torch.testing.assert_close(
        diagnostics["atom_aggregator_gate"][:, 1:],
        baseline_diagnostics["atom_aggregator_gate"][:, 1:],
    )
    torch.testing.assert_close(
        diagnostics["atom_aggregator_entropy"],
        baseline_diagnostics["atom_aggregator_entropy"],
    )


def test_reliability_gate_can_reject_a_harmful_atom() -> None:
    torch.manual_seed(47)
    aggregator = AtomResidualAggregator(32)
    with torch.no_grad():
        aggregator.mix_gate.fill_(100.0)
        aggregator.reliability[-1].bias[0] = -4.0
    residuals = torch.randn(1, aggregator.atom_count, 5, 3)
    output, diagnostics = aggregator(residuals, *_inputs(aggregator))

    reliability = diagnostics["atom_aggregator_reliability"]
    weights = diagnostics["atom_aggregator_gate"]
    assert reliability.max() < 0.05
    assert weights.max() < 0.1
    assert not torch.allclose(output, residuals.sum(dim=1))
    output.square().mean().backward()
    assert aggregator.reliability[-1].weight.grad is not None
