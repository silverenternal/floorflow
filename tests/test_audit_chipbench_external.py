import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chip_external_holdout_chipbench_v1.json"
MANIFEST = ROOT / "outputs/reports/chip_chipbench_holdout_manifest_v1.json"


def test_chipbench_selection_is_frozen_before_outcomes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    selected = config["selection"]["chipbench_designs"]
    families = config["selection"]["independent_families"]
    assert len(selected) == 13
    assert len(set(selected)) == 13
    assert config["selection"]["required_total_designs"] == 15
    expected = config["selection"]["expected_macro_counts"]
    assert expected["ariane81"] == 81
    assert expected["VeriGPU"] == 12
    assert expected["nangate45/bp_quad"] == 220
    assert config["evaluation_contract"]["candidate_outcomes_opened"] is False
    assert config["evaluation_contract"]["adapter_smoke_cases"] == [
        "VeriGPU",
        "or1200",
        "dft68",
    ]
    assert all(name in families for name in selected)
    supplements = config["selection"]["orf_supplements"]
    assert [row["design_key"] for row in supplements] == [
        "nangate45/bp_quad",
        "sky130hd/chameleon",
    ]
    assert all(row["design_key"] in families for row in supplements)
    chameleon = supplements[1]
    assert chameleon["macro_master_count"] == 4
    assert chameleon["macro_instance_count_static_estimate"] == 6
    assert chameleon["static_estimate"]["DFFRAM_4K"] == 3


def test_chipbench_does_not_count_aspect_variants_as_independent() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = config["evaluation_contract"]["do_not_count_as_independent"]
    assert any("aspect ratios" in item for item in forbidden)
    assert config["source"]["license"] == "BSD-3-Clause"
    assert "Croissant metadata states MIT" in config["source"]["license_note"]


def test_frozen_manifest_records_all_chipbench_graph_audits() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["design_count"] == 15
    assert manifest["family_count"] == 10
    assert manifest["checks"]["candidate_outcomes_unopened"] is True
    assert manifest["checks"]["chipbench_graph_audits_pass"] is True
    assert set(manifest["adapter_smoke_cases"]) == {
        "VeriGPU",
        "or1200",
        "dft68",
    }
    audits = [
        row["graph_audit"]
        for row in manifest["designs"]
        if row["source"] == "ChiPBench-D"
    ]
    assert len(audits) == 13
    assert all(audit["macros"] > 0 for audit in audits)
    assert all(audit["b2b_edges"] > 0 and audit["p2b_edges"] > 0 for audit in audits)
    chameleon = next(
        row for row in manifest["designs"] if row["id"] == "sky130hd/chameleon"
    )
    assert chameleon["macro_master_count"] == 4
    assert chameleon["macro_instance_count_static_estimate"] == 6
