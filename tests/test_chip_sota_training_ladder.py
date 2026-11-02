import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chip_sota_training_ladder import (
    evaluation_command,
    should_advance,
    training_command,
)


def _config() -> dict:
    return json.loads(
        (
            ROOT / "configs/chip_sota_training_compute_ladder_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_training_ladder_gate_requires_improvement_without_regression() -> None:
    previous = {"cost": 1.2, "v_rel": 0.05, "feasible_fraction": 1.0}
    improved = should_advance(
        previous,
        {"cost": 1.19, "v_rel": 0.049, "feasible_fraction": 1.0},
        max_relative_regression=0.002,
        min_relative_improvement=0.005,
    )
    regressed = should_advance(
        previous,
        {"cost": 1.19, "v_rel": 0.051, "feasible_fraction": 1.0},
        max_relative_regression=0.002,
        min_relative_improvement=0.005,
    )
    stagnant = should_advance(
        previous,
        {"cost": 1.2, "v_rel": 0.05, "feasible_fraction": 1.0},
        max_relative_regression=0.002,
        min_relative_improvement=0.005,
    )

    assert improved["advance"] is True
    assert regressed["advance"] is False
    assert stagnant["advance"] is False


def test_training_command_is_subset_and_accumulated() -> None:
    config = _config()
    command = training_command(config, 50_000, Path("/tmp/sota"))

    assert "--allow-full-data" not in command
    assert command[command.index("--train-indices") + 1] == config["train_indices"]
    assert command[command.index("--grad-accum-steps") + 1] == "2"
    assert command[command.index("--device") + 1] == "cuda:0"


def test_full_data_command_requires_explicit_opt_in() -> None:
    config = _config()
    config.pop("train_indices")
    command = training_command(config, 50_000, Path("/tmp/sota-full"))
    assert "--train-indices" not in command
    assert "--allow-full-data" in command


def test_rejected_combined_recipe_fails_reference_gate() -> None:
    rejected = should_advance(
        {"cost": 1.189661233502801, "v_rel": 0.042707499085844924},
        {"cost": 1.2603348763308093, "v_rel": 0.05387706005742511},
        max_relative_regression=0.002,
        min_relative_improvement=0.005,
    )
    assert rejected["advance"] is False
    assert rejected["cost_relative"] > 0.05
    assert rejected["v_rel_relative"] > 0.25


def test_evaluation_command_uses_primary_budget() -> None:
    config = _config()
    command = evaluation_command(
        config,
        Path("/tmp/final.pt"),
        Path("/tmp/eval.json"),
    )

    assert command[command.index("--nfe") + 1] == "4"
    assert command[command.index("--refiner-steps") + 1] == "400"
    assert "--use-ema" in command
    assert "--deterministic" in command
