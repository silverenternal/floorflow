import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_u1_dynamic_graph_screen.py"
    spec = importlib.util.spec_from_file_location("u1_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u1_train_command_toggles_only_dynamic_relay() -> None:
    module = _module()
    config = module.load(
        ROOT / "configs/chip_u1_dynamic_graph_screen_v1.json"
    )
    commands = []
    original = module.run_checked
    module.run_checked = lambda command, log_path: commands.append(command)
    try:
        module.train(config, Path("/tmp/u1"), 2000, "control_quick", False, False)
        module.train(config, Path("/tmp/u1"), 2000, "candidate_quick", True, False)
    finally:
        module.run_checked = original
    assert "--dynamic-graph-relay" not in commands[0]
    assert "--dynamic-graph-relay" in commands[1]
    assert commands[0][commands[0].index("--train-indices") + 1] == (
        "configs/chip_train_screen_5k_v1.json"
    )


def test_generic_mechanism_screen_uses_candidate_flag() -> None:
    module = _module()
    config = module.load(
        ROOT / "configs/chip_order_staged_warmstart_screen_v1.json"
    )
    commands = []
    original = module.run_checked
    module.run_checked = lambda command, log_path: commands.append(command)
    try:
        module.train(config, Path("/tmp/order"), 2000, "control_quick", False, False)
        module.train(config, Path("/tmp/order"), 2000, "candidate_quick", True, False)
    finally:
        module.run_checked = original
    assert "--hierarchy-order-conditioning" not in commands[0]
    assert "--hierarchy-order-conditioning" in commands[1]
    assert commands[1][commands[1].index("--device") + 1] == "cuda:1"
    assert "--resume" in commands[0] and "--resume" in commands[1]
    assert "--resume-allow-new-params" in commands[0]
    assert "--resume-allow-new-params" in commands[1]


def test_mechanism_screen_forwards_candidate_value_arguments() -> None:
    module = _module()
    config = module.load(
        ROOT / "configs/chip_order_staged_warmstart_screen_v1.json"
    )
    config["candidate_flag"] = None
    config["candidate_args"] = ["--physics-tail-weight", "0.25"]
    commands = []
    original = module.run_checked
    module.run_checked = lambda command, log_path: commands.append(command)
    try:
        module.train(config, Path("/tmp/tail"), 2000, "control_quick", False, False)
        module.train(config, Path("/tmp/tail"), 2000, "candidate_quick", True, False)
    finally:
        module.run_checked = original
    assert "--physics-tail-weight" not in commands[0]
    index = commands[1].index("--physics-tail-weight")
    assert commands[1][index + 1] == "0.25"


def test_s1_screen_applies_atom_isolation_only_to_candidate() -> None:
    module = _module()
    config = module.load(
        ROOT / "configs/chip_s1_atom_isolated_screen_v1.json"
    )
    commands = []
    original = module.run_checked
    module.run_checked = lambda command, log_path: commands.append(command)
    try:
        module.train(config, Path("/tmp/s1"), 1000, "control_quick", False, False)
        module.train(config, Path("/tmp/s1"), 1000, "candidate_quick", True, False)
    finally:
        module.run_checked = original

    assert "--require-atom-isolation" not in commands[0]
    assert "--training-scope-schedule" not in commands[0]
    assert "--require-atom-isolation" in commands[1]
    schedule_index = commands[1].index("--training-scope-schedule")
    assert commands[1][schedule_index + 1] == (
        "0:area,150:b2b,300:port_route,450:boundary,600:tail,750:fusion"
    )
    assert "--response-sidecar" in commands[0]
    assert "--response-sidecar" in commands[1]


def test_s1_screen_applies_shared_post_refiner_projector() -> None:
    module = _module()
    config = module.load(
        ROOT / "configs/chip_s1_atom_isolated_screen_v1.json"
    )
    commands = []
    original = module.run_checked
    module.run_checked = lambda command, log_path: commands.append(command)
    try:
        module.evaluate(
            config,
            Path("/tmp/candidate.pt"),
            Path("/tmp/s1"),
            "candidate_quick",
            400,
            False,
        )
    finally:
        module.run_checked = original

    command = commands[0]
    index = command.index("--post-refiner-projector-iterations")
    assert command[index + 1] == "8"
    assert "--projector-boundary-axes" in command
