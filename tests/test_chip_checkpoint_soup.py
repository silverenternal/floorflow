import torch

from floorflow.chip_flow.checkpoint_soup import weighted_average_state_dicts


def test_weighted_checkpoint_soup_averages_float_tensors() -> None:
    first = {"weight": torch.tensor([0.0, 2.0]), "step": torch.tensor(1)}
    second = {"weight": torch.tensor([2.0, 4.0]), "step": torch.tensor(2)}
    result = weighted_average_state_dicts([first, second], [1.0, 3.0])
    assert torch.allclose(result["weight"], torch.tensor([1.5, 3.5]))
    assert torch.equal(result["step"], first["step"])


def test_weighted_checkpoint_soup_rejects_key_mismatch() -> None:
    with torch.no_grad():
        try:
            weighted_average_state_dicts(
                [{"a": torch.ones(1)}, {"b": torch.ones(1)}]
            )
        except ValueError as error:
            assert "state keys differ" in str(error)
        else:
            raise AssertionError("expected key mismatch")
