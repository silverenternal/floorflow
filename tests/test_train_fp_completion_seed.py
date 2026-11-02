from __future__ import annotations

from pathlib import Path

from floorflow.training.fp_completion_seed import build_stage_command


def test_stage_commands_share_seed_and_output_root() -> None:
    for stage in ("layout", "structure", "furniture", "support"):
        output_dir, command = build_stage_command(
            stage=stage,
            seed=20,
            device="cuda",
            output_root=Path("outputs/train/test_seed"),
            data_root=Path(".data/cubicasa5k"),
            cache_root=Path("outputs/cache/cubicasa5k_full_v2"),
            geometry_root=Path("outputs/cache/cubicasa5k_geometry_v1"),
            batch_size=16,
            workers=8,
            limit_val=191,
        )
        assert output_dir == Path("outputs/train/test_seed") / stage
        assert "--seed" in command
        assert command[command.index("--seed") + 1] == "20"
        assert "--output-dir" in command
