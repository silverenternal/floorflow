from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/summarize_chip_v2_authorized_quick.py"
    spec = importlib.util.spec_from_file_location("chip_v2_authorized_quick", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorized_quick_selects_only_m64_for_confirmation() -> None:
    report = _module().build_summary()
    stages = report["stages"]

    assert report["status"] == "joint_primary_candidate_selected_confirmation_pending"
    assert set(stages) == {
        "M50",
        "M51",
        "M52",
        "M53",
        "M54",
        "M55",
        "M57",
        "M58",
        "M59",
        "M60",
        "M61A",
        "M61B",
        "M64",
        "M65",
    }
    assert {
        name for name, stage in stages.items() if stage["decision"] == "promote_to_confirmation"
    } == {"M64"}
    assert stages["M65"]["decision"].startswith("reject")
    assert stages["M50"]["raw"]["total"]["relative"] > 0
    assert stages["M53"]["legalized"]["legalized_hard_cost"]["relative"] > 0
    assert stages["M53"]["legalized"]["legalized_v_rel"]["relative"] < 0
    assert stages["M55"]["legalized"]["legalized_hard_cost"]["relative"] < 0
    assert stages["M55"]["legalized"]["legalized_v_rel"]["relative"] > 0
    assert stages["M64"]["legalized"]["legalized_hard_cost"]["relative"] < 0
    assert stages["M64"]["legalized"]["legalized_v_rel"]["relative"] < 0
    assert stages["M64"]["legalized_absolute"]["legalized_feasible_fraction"] == 1.0
    assert report["selected_candidate"]["stage"] == "M64"
    assert report["sota_ready"] is False
