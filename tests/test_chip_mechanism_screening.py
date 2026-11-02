import json
from pathlib import Path

from floorflow.chip_flow.mechanism_screening import infer_stage, screen_status

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_screen_promotion_requires_confirmation_and_matched_full_control() -> None:
    screen = _read("configs/chip_mechanism_screen_v1.json")
    promotion = screen["promotion"]
    full_data = promotion["full_data"]
    assert promotion["quick_pass_is_candidate"] is False
    assert promotion["confirmation_required"] is True
    assert promotion["full_training_after_registration_only"] is True
    assert promotion["full_data_is_not_an_iteration_budget"] is True
    assert full_data["matched_control_required"] is True
    assert full_data["same_training_seed_required"] is True
    assert full_data["frozen_finalist_only"] is True
    assert full_data["no_variant_comparison_allowed"] is True
    assert full_data["multiseed_after_full_pass_only"] is True
    iteration = screen["iteration_policy"]
    assert iteration["exploratory_data_ceiling_cases"] == 5000
    assert iteration["full_data_variant_search_forbidden"] is True


def test_registered_candidates_have_confirmation_evidence() -> None:
    registry = _read("configs/chip_candidate_registry.json")
    policy = registry["policy"]
    assert policy["subset_confirmation_required"] is True
    assert policy["quick_pass_is_final_validation"] is False
    assert policy["full_data_requires_same_seed_control"] is True
    for candidate in registry["candidates"]:
        assert candidate["status"] not in {
            "provisional",
            "quick_pass_promote_to_confirmation",
            "provisional_pending_confirmation",
        }
        quick_report = candidate.get("report", candidate.get("quick_report"))
        assert quick_report is not None
        assert (ROOT / quick_report).is_file()
        assert (ROOT / candidate["confirmation_report"]).is_file()
        if candidate.get("training_kind", "trainable") == "trainable":
            assert candidate["confirmation_steps"] > candidate["quick_steps"]
        else:
            assert (
                candidate["confirmation_refiner_steps"]
                > candidate["quick_refiner_steps"]
            )
        assert candidate["feasible_fraction"] == 1.0


def test_full_data_screen_has_a_distinct_promotion_state() -> None:
    assert infer_stage("nfe4_refine400") == "quick"
    assert infer_stage("nfe4_refine1600") == "confirmation"
    assert screen_status("quick", True) == "quick_pass_promote_to_confirmation"
    assert screen_status("confirmation", False) == "rejected_confirmation"
    assert screen_status("full", True) == "full_pass_promote_to_multiseed"
    assert screen_status("full", False) == "rejected_full_data"
