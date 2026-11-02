from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_pair_integrity_cli_detects_collapsed_legalized_layouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    adapter_runs = []
    comparisons = []
    for case, candidate_text, control_text in (
        ("p/distinct", "candidate", "control"),
        ("p/collapsed", "same", "same"),
    ):
        for arm, text in (
            ("candidate", candidate_text),
            ("control", control_text),
        ):
            tcl = root / "placements" / case.replace("/", "__") / f"{arm}.tcl"
            tcl.parent.mkdir(parents=True, exist_ok=True)
            tcl.write_text(text, encoding="utf-8")
            placement = tcl.with_suffix(".json")
            placement.write_text(
                json.dumps(
                    {
                        "legalizer_route": "scale_pack",
                        "connectivity": {
                            "legalized_output": {"p2b_hpwl_um": 1.0}
                        },
                        "outputs": {"tcl": str(tcl.relative_to(root))},
                    }
                ),
                encoding="utf-8",
            )
            adapter_runs.append(
                {
                    "design_key": case,
                    "arm": arm,
                    "report": str(placement.relative_to(root)),
                }
            )
        comparisons.append({"case": case, "status": "completed"})
    adapter = root / "adapter.json"
    adapter.write_text(
        json.dumps({"placement_runs": adapter_runs}),
        encoding="utf-8",
    )
    ppa = root / "ppa.json"
    ppa.write_text(json.dumps({"comparisons": comparisons}), encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts/audit_chip_external_pair_integrity.py"
    spec = importlib.util.spec_from_file_location("pair_integrity", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "adapter_report": adapter,
                "ppa_report": ppa,
                "output": root / "report.json",
            },
        )(),
    )

    try:
        module.main()
    except SystemExit as error:
        assert error.code == 1
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["collapsed_cases"] == ["p/collapsed"]
    assert report["valid_completed_cases"] == ["p/distinct"]
    assert report["status"] == "reject_legalized_candidate_control_collapse"
