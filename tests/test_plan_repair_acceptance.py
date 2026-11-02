import torch

from floorflow.heads.plan_repair_acceptance import (
    PlanRepairAcceptanceHead,
    select_accepted_repairs,
)


def test_plan_repair_acceptance_is_fail_closed_and_budgeted() -> None:
    head = PlanRepairAcceptanceHead(6, 8, hidden_dim=12)
    logits = head(
        torch.randn(2, 6),
        torch.randn(2, 5, 8),
    )
    assert logits.shape == (2, 5)
    assert not select_accepted_repairs(logits).any()
    explicit = torch.tensor([[8.0, 7.0, -8.0, 6.0, 5.0]])
    selected = select_accepted_repairs(explicit, max_edits=2)
    assert selected.tolist() == [[True, True, False, False, False]]
