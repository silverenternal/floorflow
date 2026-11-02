import importlib.util
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chipbench_external_screen.py"
    spec = importlib.util.spec_from_file_location("chipbench_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_commands_reuse_placement_and_materializer_clis() -> None:
    module = _module()
    args = Namespace(
        checkpoint=Path("checkpoint.pt"),
        device="cuda:0",
        nfe=1,
        refiner_steps=0,
        projector_iterations=8,
        seed=19,
        protocol_tag="chipbench_screen_v1",
    )
    row = {
        "id": "VeriGPU",
        "pre_place_def": ".data/chipbench_d_v1/data/VeriGPU/def/pre_place.def",
        "macro_lefs": [
            ".data/chipbench_d_v1/data/VeriGPU/lef/core.lef",
            ".data/chipbench_d_v1/data/VeriGPU/lef/small_mem.lef",
        ],
    }
    placement = module.placement_command(args, row, Path("placement.json"))
    assert placement[1].endswith("scripts/run_chip_external_placement.py")
    assert "--def" in placement and "--macro-lef" in placement
    materialize = module.materialize_command(
        args,
        row,
        Path("boxes.json"),
        Path("out.def"),
        Path("materialize.json"),
    )
    assert materialize[1].endswith("scripts/materialize_chipbench_def.py")
    assert materialize.count("--verify-macro-lef") == 2
    assert "--remove-blockages" in materialize
