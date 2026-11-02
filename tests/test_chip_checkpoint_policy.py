from __future__ import annotations

import pytest
import torch

from floorflow.chip_flow.checkpoint_policy import resolve_checkpoint_state


def test_final_policy_uses_declared_final_state(tmp_path) -> None:
    state = {"weight": torch.tensor([1.0])}
    selected, step, path, policy = resolve_checkpoint_state(
        "final",
        final_state=state,
        final_step=20,
        final_path=tmp_path / "final.pt",
        best_path=tmp_path / "best.pt",
        state_key="ema",
    )
    assert selected["weight"].item() == 1.0
    assert step == 20
    assert path.name == "final.pt"
    assert policy == "last_step_ema"


def test_best_policy_loads_frozen_validation_state(tmp_path) -> None:
    best_path = tmp_path / "best.pt"
    torch.save(
        {
            "step": 7,
            "ema": {"weight": torch.tensor([2.0])},
            "model": {"weight": torch.tensor([3.0])},
        },
        best_path,
    )
    selected, step, path, policy = resolve_checkpoint_state(
        "best",
        final_state={"weight": torch.tensor([1.0])},
        final_step=20,
        final_path=tmp_path / "final.pt",
        best_path=best_path,
        state_key="ema",
    )
    assert selected["weight"].item() == 2.0
    assert step == 7
    assert path == best_path
    assert policy == "best_validation_ema"


def test_best_policy_requires_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_state(
            "best",
            final_state={"weight": torch.tensor([1.0])},
            final_step=20,
            final_path=tmp_path / "final.pt",
            best_path=tmp_path / "missing.pt",
            state_key="ema",
        )


def test_legalized_best_policy_reads_selection_report(tmp_path) -> None:
    selected_path = tmp_path / "shortlist_step_7.pt"
    torch.save(
        {"step": 7, "ema": {"weight": torch.tensor([4.0])}},
        selected_path,
    )
    report = tmp_path / "selection.json"
    report.write_text(
        '{"selected": {"checkpoint": "' + str(selected_path) + '"}}',
        encoding="utf-8",
    )
    selected, step, path, policy = resolve_checkpoint_state(
        "legalized_best",
        final_state={"weight": torch.tensor([1.0])},
        final_step=20,
        final_path=tmp_path / "final.pt",
        best_path=tmp_path / "best.pt",
        state_key="ema",
        selection_report=report,
    )
    assert selected["weight"].item() == 4.0
    assert step == 7
    assert path == selected_path
    assert policy == "legalized_shortlist_validation"
