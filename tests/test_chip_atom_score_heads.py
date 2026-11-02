from __future__ import annotations

import torch

from floorflow.chip_flow.atom_score_heads import (
    ATOM_NAMES,
    AtomHeadBank,
    atom_head_loss,
    masked_atom_rank_loss,
    masked_atom_regression_loss,
    selector_metrics_to_atom_targets,
)
from floorflow.chip_flow.candidate_structural_selector import SELECTOR_METRICS


def test_selector_metrics_map_to_six_atom_targets_and_masks() -> None:
    metrics = {
        name: float(index)
        for index, name in enumerate(SELECTOR_METRICS)
        if name not in {"legalized_rho_mae", "legalized_p2b_hpwl_gap"}
    }
    targets, mask = selector_metrics_to_atom_targets(metrics)

    assert targets.shape == mask.shape == (len(ATOM_NAMES),)
    assert mask[ATOM_NAMES.index("grouping")]
    assert not mask[ATOM_NAMES.index("area_rho")]
    assert not mask[ATOM_NAMES.index("p2b")]
    assert torch.equal(
        targets[ATOM_NAMES.index("grouping")],
        torch.tensor(5.0),
    )


def test_atom_targets_mask_nonfinite_tensor_metrics() -> None:
    values = torch.arange(len(SELECTOR_METRICS), dtype=torch.float32)
    values[SELECTOR_METRICS.index("legalized_boundary")] = float("nan")
    targets, mask = selector_metrics_to_atom_targets(values)

    assert not mask[ATOM_NAMES.index("boundary")]
    assert targets[ATOM_NAMES.index("boundary")] == 0.0


def test_masked_losses_ignore_missing_atoms_and_generate_gradients() -> None:
    predictions = torch.tensor(
        [[0.2, 0.4, 0.1, 0.3, 0.8, 0.5], [0.9, 0.7, 0.6, 0.4, 0.2, 0.1]],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    atom_mask = torch.tensor(
        [[True, False, True, True, False, True], [True, True, False, True, True, True]]
    )
    log_variance = torch.zeros_like(predictions, requires_grad=True)

    regression = masked_atom_regression_loss(
        predictions,
        targets,
        atom_mask,
        log_variance=log_variance,
    )
    ranking = masked_atom_rank_loss(
        predictions,
        targets,
        atom_mask,
        candidate_mask=torch.tensor([True, True]),
    )
    loss = regression + ranking
    loss.backward()

    assert torch.isfinite(loss)
    assert predictions.grad is not None
    assert log_variance.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert torch.isfinite(log_variance.grad).all()


def test_rank_loss_prefers_lower_utility_for_lower_atom_target() -> None:
    targets = torch.tensor([[0.0] * len(ATOM_NAMES), [1.0] * len(ATOM_NAMES)])
    mask = torch.ones_like(targets, dtype=torch.bool)
    good = torch.tensor([[0.0] * len(ATOM_NAMES), [1.0] * len(ATOM_NAMES)])
    bad = torch.tensor([[1.0] * len(ATOM_NAMES), [0.0] * len(ATOM_NAMES)])

    assert masked_atom_rank_loss(good, targets, mask) < masked_atom_rank_loss(
        bad,
        targets,
        mask,
    )


def test_atom_head_bank_disabled_and_gate_zero_are_legacy_identity() -> None:
    torch.manual_seed(17)
    features = torch.randn(2, 4, 14)
    legacy = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    bank = AtomHeadBank(14)

    disabled = bank(features, legacy_utility=legacy)
    bank.set_enabled(True)
    zero_gate = bank(features, legacy_utility=legacy)

    torch.testing.assert_close(disabled.aggregate_utility, legacy, rtol=0, atol=0)
    torch.testing.assert_close(zero_gate.aggregate_utility, legacy, rtol=0, atol=0)


def test_atom_head_bank_isolates_single_atom_training_and_backpropagates() -> None:
    torch.manual_seed(29)
    bank = AtomHeadBank(14, hidden=8)
    bank.set_enabled(True)
    bank.set_trainable_atoms("grouping")
    features = torch.randn(2, 4, 14)
    legacy = torch.rand(2, 4)
    mask = torch.ones(2, 4, len(ATOM_NAMES), dtype=torch.bool)
    output = bank(
        features,
        legacy_utility=legacy,
        atom_mask=mask,
    )
    bank.gate.data.fill_(1.0)
    targets = torch.zeros_like(output.utilities)
    terms = atom_head_loss(output, targets, mask)
    terms["atom_total"].backward()

    assert bank.gate.grad is None
    for name, head in bank.heads.items():
        gradients = [
            parameter.grad
            for parameter in head.parameters()
            if parameter.grad is not None
        ]
        if name == "grouping":
            assert gradients
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
        else:
            assert not gradients
