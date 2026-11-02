from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.evaluation.atom_metric_contract import (
    build_atom_metric_registry_report,
    load_metric_pairs,
    validate_atom_plan,
)
from floorflow.evaluation.atom_proxy_consistency import (
    ConsistencyThresholds,
    build_consistency_report,
    load_measurement_payloads,
)

ROOT = Path(__file__).resolve().parents[1]
FPTODO = ROOT / "fptodo.json"
PAIRS = ROOT / "configs/atom_metric_pairs_v1.json"


def _payload(*, target_used: bool = False, flat_proxy: bool = False) -> dict:
    rows = []
    for seed in (19, 23):
        for checkpoint in range(6):
            for case in range(8):
                latent = checkpoint + 0.1 * case + 0.01 * seed
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint": f"epoch_{checkpoint:02d}",
                        "case_id": f"case_{case}",
                        "proxy": 0.0 if flat_proxy else latent,
                        "canonical": latent,
                    }
                )
    return {
        "target_used_at_inference": target_used,
        "measurements": [
            {
                "id": "RM-01.synthetic",
                "atom_id": "RM-01",
                "proxy_metric": "query_mask_dice",
                "canonical_metric": "room_macro_iou",
                "proxy_direction": "maximize",
                "canonical_direction": "maximize",
                "baseline_checkpoint": "epoch_00",
                "target_used_at_inference": target_used,
                "rows": rows,
                "sensitivity": [
                    {
                        "case_id": "case_0",
                        "attack": "drop_query",
                        "proxy_baseline": 1.0,
                        "proxy_attack": 0.4,
                        "canonical_baseline": 1.0,
                        "canonical_attack": 0.3,
                    }
                ],
            }
        ],
    }


def test_atom_metric_registry_covers_every_plan_atom() -> None:
    report = build_atom_metric_registry_report(
        FPTODO,
        PAIRS,
        root=ROOT,
    )
    assert report["complete"] is True
    assert report["counts"]["atoms"] == 18
    assert report["counts"]["pairs"] == 36
    assert all(
        atom["status"] == "ready_for_measurement"
        for atom in report["atoms"].values()
    )


def test_plan_rejects_pair_metric_ownership_mismatch() -> None:
    plan = json.loads(FPTODO.read_text(encoding="utf-8"))
    pairs = load_metric_pairs(PAIRS)
    audit = validate_atom_plan(plan, pairs, root=ROOT)
    assert audit["errors"] == []

    broken = json.loads(json.dumps(plan))
    broken["atoms"]["RM-01"]["proxy"].remove("fast_room_macro_iou")
    audit = validate_atom_plan(broken, pairs, root=ROOT)
    assert any(
        "fast_room_macro_iou is not owned" in error
        for error in audit["errors"]
    )


def test_consistent_proxy_passes_all_contract_checks(tmp_path: Path) -> None:
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    measurements = load_measurement_payloads([path])
    report = build_consistency_report(
        measurements,
        ConsistencyThresholds(),
    )
    row = report["measurements"][0]
    assert row["passed"] is True
    assert row["summary"]["spearman_rho"] == pytest.approx(1.0)
    assert row["summary"]["top_checkpoint_agreement"] == 1.0
    assert row["summary"]["paired_sign_agreement"] == 1.0
    assert row["summary"]["sensitivity_rate"] == 1.0


@pytest.mark.parametrize(
    ("payload", "failed_check"),
    [
        (_payload(target_used=True), "target_free"),
        (_payload(flat_proxy=True), "not_flat"),
    ],
)
def test_invalid_proxy_contracts_fail_closed(
    tmp_path: Path,
    payload: dict,
    failed_check: str,
) -> None:
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_consistency_report(
        load_measurement_payloads([path]),
        ConsistencyThresholds(),
    )
    row = report["measurements"][0]
    assert row["checks"][failed_check] is False
    assert row["passed"] is False


def test_duplicate_measurement_rows_are_rejected(tmp_path: Path) -> None:
    payload = _payload()
    measurement = payload["measurements"][0]
    measurement["rows"].append(dict(measurement["rows"][0]))
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate row"):
        load_measurement_payloads([path])


def test_top_checkpoint_accepts_tolerance_near_ties(tmp_path: Path) -> None:
    payload = _payload()
    measurement = payload["measurements"][0]
    measurement["canonical_tolerance"] = 0.01
    for row in measurement["rows"]:
        row["canonical"] = 0.1 * int(row["case_id"].split("_")[-1])
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_consistency_report(
        load_measurement_payloads([path]),
        ConsistencyThresholds(),
    )
    per_seed = report["measurements"][0]["per_seed"]
    assert all(
        row["top_checkpoint_agreement"] is True
        for row in per_seed.values()
    )
