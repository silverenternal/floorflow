import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_capacity_screen.py"
    spec = importlib.util.spec_from_file_location("capacity_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capacity_gate_requires_no_regression_and_improvement() -> None:
    module = _module()
    control = {"cost": 1.2, "v_rel": 0.05}
    passed = module.arm_gate(
        control,
        {"cost": 1.19, "v_rel": 0.048},
    )
    failed = module.arm_gate(
        control,
        {"cost": 1.19, "v_rel": 0.051},
    )
    assert passed["passes_budget_class"] is True
    assert failed["passes_budget_class"] is False


def test_capacity_runner_maps_quick_phase_name() -> None:
    module = _module()
    plan = {"quick_screen": {"arms": []}, "confirmation": {"arms": []}}
    assert module.phase_payload(plan, "quick")["arms"] == []
    assert module.phase_payload(plan, "confirmation")["arms"] == []


def test_capacity_runner_rejects_empty_s1_ladder(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"status": "complete", "steps": []}))
    try:
        module.validate_ladder_report(path)
    except ValueError as error:
        assert "no completed training arm" in str(error)
    else:
        raise AssertionError("empty ladder should be rejected")


def test_confirmation_phase_resizes_to_selected_s1_horizon() -> None:
    module = _module()
    phase = {
        "arms": [
            {
                "id": "d192-control",
                "budget_class": "control",
                "measured_block_ratio": 1.0,
                "steps": 20000,
            },
            {
                "id": "d256-equal-exposure",
                "budget_class": "equal-exposure",
                "measured_block_ratio": 1.6,
                "steps": 20000,
            },
            {
                "id": "d256-equal-flops",
                "budget_class": "equal-flops",
                "measured_block_ratio": 1.6,
                "steps": 12500,
            },
        ]
    }
    resized = module.resize_confirmation_phase(phase, 50000)
    assert resized["base_steps"] == 50000
    assert resized["arms"][0]["steps"] == 50000
    assert resized["arms"][1]["steps"] == 50000
    assert resized["arms"][2]["steps"] == 31250


def test_capacity_training_command_uses_arm_width_and_steps() -> None:
    module = _module()
    base = {
        "training": {
            "architecture": "typed_hierarchy",
            "micro_batch": 128,
            "grad_accum_steps": 2,
            "sampling_mode": "random",
            "curve_per_count": 5,
            "sampling_steps": 8,
            "hierarchy_depth": 10,
            "typed_layers": 1,
            "head_dim": 64,
            "projector_iterations": 8,
            "overlap_weight_scale": 10.0,
            "hierarchy_bbox_weight": 0.5,
            "hierarchy_bbox_decay_steps": 10000,
            "eval_state": "ema",
            "checkpoint_every": 1000,
        }
    }
    phase = {
        "cache": "outputs/cache",
        "train_indices": "configs/indices.json",
        "seed": 19,
        "device": "cuda:0",
    }
    arm = {"id": "d256-equal-flops", "width": 256, "steps": 3022}
    command = module.training_command(base, phase, arm, Path("/tmp/s3"))
    assert command[command.index("--hidden") + 1] == "256"
    assert command[command.index("--heads") + 1] == "4"
    assert command[command.index("--steps") + 1] == "3022"


def test_capacity_report_metric_reader_uses_diagnostic_key(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "nfe4_refine1600": {
                        "legalized_hard_cost": 1.1,
                        "legalized_v_rel": 0.02,
                        "legalized_feasible_fraction": 1.0,
                    }
                }
            }
        )
    )
    metrics = module.evaluation_metrics_for_key(path, "nfe4_refine1600")
    assert metrics["cost"] == 1.1
    assert metrics["v_rel"] == 0.02
