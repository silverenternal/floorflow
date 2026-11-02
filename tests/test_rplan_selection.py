from floorflow.training.rplan_selection import rplan_selection_score


def test_rplan_selection_rejects_room_gain_with_invalid_legal_contract() -> None:
    room_heavy_invalid = {
        "matched_geometric_iou": 0.70,
        "instance_f1_at_threshold": 0.60,
        "matched_type_accuracy": 0.90,
        "room_count_mae": 0.2,
        "contract": {"contract_valid": False},
    }
    gate_balanced_valid = {
        "matched_geometric_iou": 0.45,
        "instance_f1_at_threshold": 0.50,
        "matched_type_accuracy": 0.75,
        "room_count_mae": 1.0,
        "contract": {"contract_valid": True},
    }
    assert rplan_selection_score(
        gate_balanced_valid,
        token_enabled=False,
    ) > rplan_selection_score(
        room_heavy_invalid,
        token_enabled=False,
    )


def test_rplan_token_selection_penalizes_consumer_regression() -> None:
    clean = {
        "token_accuracy": 0.95,
        "token_macro_f1": 0.90,
        "canonical_token_ce": 1.0,
        "canonical_token_grammar_violation_rate": 0.0,
        "token_consumer": {
            "token_consumer_generated_contract_valid": 1.0,
            "token_consumer_generated_functional_watertight": 1.0,
            "token_consumer_generated_validity_regression": 0.0,
            "token_consumer_generated_watertight_regression": 0.0,
        },
    }
    regressed = {
        **clean,
        "token_consumer": {
            **clean["token_consumer"],
            "token_consumer_generated_contract_valid": 0.0,
            "token_consumer_generated_validity_regression": 1.0,
        },
    }
    assert rplan_selection_score(
        clean,
        token_enabled=True,
    ) > rplan_selection_score(
        regressed,
        token_enabled=True,
    )


def test_rplan_selection_includes_wall_opening_and_adjacency_gates() -> None:
    room_only = {
        "matched_geometric_iou": 0.70,
        "instance_f1_at_threshold": 0.60,
        "matched_type_accuracy": 0.90,
        "room_count_mae": 0.2,
        "wall_total_loss": 1.4,
        "opening_total_loss": 3.0,
        "room_adjacency_bce": 0.9,
        "contract": {"contract_valid": True},
    }
    balanced = {
        **room_only,
        "matched_geometric_iou": 0.45,
        "instance_f1_at_threshold": 0.50,
        "matched_type_accuracy": 0.75,
        "room_count_mae": 1.0,
        "wall_total_loss": 0.5,
        "opening_total_loss": 0.2,
        "room_adjacency_bce": 0.2,
    }
    assert rplan_selection_score(
        balanced,
        token_enabled=False,
    ) > rplan_selection_score(
        room_only,
        token_enabled=False,
    )
