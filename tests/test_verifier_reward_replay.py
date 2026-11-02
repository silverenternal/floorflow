import json

import torch

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.objectives.verifier_policy import verifier_policy_loss
from floorflow.training.verifier_reward_replay import (
    REWARD_REPLAY_VERSION,
    VerifierRewardReplay,
    _to_cpu,
)


def _batch() -> dict:
    return {
        "boundary_mask": torch.zeros(2, 1, 4, 4),
        "room_exists": torch.tensor([[True, False], [True, True]]),
        "verifier_target_keys": ["target-a", "target-b"],
    }


def test_verifier_replay_loads_target_and_namespace_locked_rewards(
    tmp_path,
) -> None:
    payload = {
        "version": REWARD_REPLAY_VERSION,
        "namespace": "vocab-v1",
        "entries": {
            "vocab-v1:target-a": {
                "reward": 2.5,
                "components": {"reward_validity": 1.0},
            }
        },
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    replay = VerifierRewardReplay(
        object(),
        FEVPTolerances(),
        namespace="vocab-v1",
        replay_path=path,
        synchronous=True,
    )
    result = replay.rewards(_batch(), 10, device=torch.device("cpu"))
    assert torch.equal(result.valid, torch.tensor([True, False]))
    assert result.rewards[0] == 2.5
    assert result.rewards[1] == 0.0
    output = tmp_path / "roundtrip.json"
    replay.write_replay(output)
    assert json.loads(output.read_text())["entries"]["vocab-v1:target-a"][
        "reward"
    ] == 2.5


def test_verifier_policy_mask_excludes_unreplayed_samples() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.randint(0, 5, (2, 3))
    loss, components = verifier_policy_loss(
        logits,
        targets,
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([4.0, -100.0]),
        reward_mask=torch.tensor([True, False]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert components["verifier_reward_coverage"] == 0.5
    assert components["verifier_reward_mean"] == 4.0


def test_cpu_contract_snapshot_promotes_bfloat16_outputs() -> None:
    snapshot = _to_cpu(torch.ones(2, dtype=torch.bfloat16))
    assert snapshot.device.type == "cpu"
    assert snapshot.dtype == torch.float32
    token_ids = torch.tensor([1, 2], dtype=torch.long)
    assert _to_cpu(token_ids).dtype == torch.long
