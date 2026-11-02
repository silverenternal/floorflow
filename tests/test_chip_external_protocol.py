from __future__ import annotations

from floorflow.chip_flow.refiner_config import append_refiner_protocol


def test_external_refiner_protocol_is_serialized_exactly() -> None:
    command = ["python", "placement.py"]
    append_refiner_protocol(
        command,
        {
            "refiner_learning_rate": 3.0e-4,
            "refiner_schedule": "cosine",
            "refiner_optimizer": "sgd",
            "refiner_chunk": 7,
            "refiner_weights": {"boundary": 2.0},
            "refiner_shape_update": True,
            "legalizer_time_limit_ms": 2000,
            "deterministic": True,
        },
    )
    assert command[command.index("--refiner-lr") + 1] == "0.0003"
    assert command[command.index("--refiner-schedule") + 1] == "cosine"
    assert command[command.index("--refiner-weight") + 1] == "boundary=2.0"
    assert "--refiner-shape" in command
    assert "--deterministic" in command
    assert command[command.index("--legalizer-time-limit-ms") + 1] == "2000"
