from __future__ import annotations

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.predicted_control_state import (
    _mask_inactive_query_slots,
    predicted_control_typed_state,
)
from floorflow.model.predicted_channel_model import PredictedChannelModel


def test_predicted_output_builds_editable_typed_state() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Bed": 4}),
            "door": Counter({"Swing": 4}),
            "window": Counter({"Regular": 4}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(
            len(taxonomy.room_ids),
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            len(taxonomy.furniture_ids) + 1,
            2,
        ),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=False,
        use_action_graph=True,
        enable_instance_room_decoder=True,
        enable_room_adjacency=True,
        room_instance_hidden=16,
        room_instance_queries=4,
        room_instance_layers=1,
    ).to(device)
    controls = {
        "room_logits": torch.randn(
            1,
            len(taxonomy.room_ids),
            16,
            16,
            device=device,
        ),
        "structure": torch.randn(
            1,
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            16,
            16,
            device=device,
        ),
        "furniture": torch.randn(
            1,
            len(taxonomy.furniture_ids) + 1,
            16,
            16,
            device=device,
        ),
        "support": torch.randn(1, 2, 16, 16, device=device),
        "program": torch.tensor([[1, 0]], dtype=torch.bool, device=device),
    }
    output = model(controls, torch.ones(1, 16, 16, device=device))
    state = predicted_control_typed_state(output)
    assert state.room_features is not None
    assert state.room_exists is not None
    assert state.room_class is not None
    assert state.room_adjacency_logits.shape == (1, 4, 4)
    assert state.stage_tokens.shape[1] == 8


def test_predicted_edit_state_respects_query_count_selection() -> None:
    output = {
        "room_identity_tokens": torch.zeros(1, 3, 4),
        "room_instance_query_existence_logits": torch.full(
            (1, 3),
            8.0,
        ),
        "room_instance_class_logits": torch.ones(1, 3, 3),
        "room_adjacency_logits": torch.ones(1, 3, 3),
        "action_stage_tokens": torch.zeros(1, 2, 4),
        "query_count_selection_mask": torch.tensor(
            [[True, False, True]]
        ),
    }
    state = predicted_control_typed_state(output)
    assert state.room_exists[0, 1] < -1.0e30
    assert state.room_exists[0, 0] == 8.0
    assert state.room_instance_ids.tolist() == [[0, -1, 2]]
    assert torch.count_nonzero(state.room_features[0, 1]) == 0
    assert state.room_class[0, 1, :-1].max() < -1.0e30
    assert state.room_class[0, 1, -1] == 0.0
    assert state.room_adjacency_logits[0, 1].max() < -1.0e30
    assert state.room_adjacency_logits[0, :, 1].max() < -1.0e30
    assert state.room_adjacency_logits[0, 0, 2] == 1.0


def test_predicted_edit_state_masks_all_query_indexed_logits() -> None:
    payload = {
        "room_identity_tokens": torch.zeros(1, 3, 4),
        "room_instance_ids": torch.tensor([[5, 6, 7]]),
        "room_instance_query_existence_logits": torch.zeros(1, 3),
        "room_instance_class_logits": torch.zeros(1, 3, 3),
        "room_instance_mask_logits": torch.zeros(1, 3, 4, 4),
        "room_instance_aux_mask_logits": [
            torch.zeros(1, 3, 4, 4),
        ],
        "room_adjacency_logits": torch.zeros(1, 3, 3),
        "program_room_assignment_logits": torch.zeros(1, 3, 2),
        "program_room_scheduled_logits": torch.zeros(1, 3, 2),
    }
    _mask_inactive_query_slots(
        payload,
        torch.tensor([[True, False, True]]),
    )
    assert payload["room_instance_ids"].tolist() == [[5, -1, 7]]
    assert payload["room_instance_mask_logits"][0, 1].max() < -1.0e30
    assert payload["room_instance_aux_mask_logits"][0][
        0, 1
    ].max() < -1.0e30
    assert payload["program_room_assignment_logits"][0, 1].max() < -1.0e30
    assert payload["program_room_scheduled_logits"][0, 1].max() < -1.0e30


def test_predicted_edit_state_rejects_misaligned_adjacency() -> None:
    output = {
        "room_identity_tokens": torch.zeros(1, 3, 4),
        "room_instance_query_existence_logits": torch.zeros(1, 3),
        "room_instance_class_logits": torch.zeros(1, 3, 3),
        "room_adjacency_logits": torch.zeros(1, 2, 2),
        "action_stage_tokens": torch.zeros(1, 2, 4),
    }
    with pytest.raises(ValueError, match=r"adjacency.*\[B,Q,Q\]"):
        predicted_control_typed_state(output)
