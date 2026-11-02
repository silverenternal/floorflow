from __future__ import annotations

import json

import numpy as np
import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.hierarchy_context import encode_typed_channel_state
from floorflow.chip_flow.training_scopes import (
    apply_training_scope,
    parse_training_scope,
)
from floorflow.chip_flow.typed_channel_cache import (
    cache_key,
    collate_typed_channel_states,
    encoder_state_fingerprint,
    input_fingerprint,
    load_typed_channel_states,
    read_typed_channel_state,
    typed_channel_case_path,
    write_typed_channel_shard,
    write_typed_channel_state,
)
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder


def _inputs() -> tuple[torch.Tensor, torch.Tensor, dict]:
    constraints = torch.zeros(1, 4, 5)
    constraints[:, :2, 2] = 1.0
    constraints[:, :, 3] = torch.tensor([1, 1, 2, 2])
    constraints[:, :, 4] = torch.tensor([1, 2, 0, 0])
    batch = {
        "areas": torch.ones(1, 4),
        "boxes": torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
              [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]
        ),
        "constraints": constraints,
        "b2b": torch.tensor([[[0.0, 1.0, 1.0]]]),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "pins": torch.tensor([[[0.2, 0.4]]]),
        "n_b2b": torch.tensor([1]),
        "n_p2b": torch.tensor([1]),
        "n_pins": torch.tensor([1]),
        "global_spec": torch.randn(1, 7),
        "node_features": torch.randn(1, 4, 2),
        "port_features": torch.randn(1, 1, 2),
    }
    return torch.randn(1, 4, 19), constraints, batch


def test_typed_channel_cache_roundtrip_matches_fusion(tmp_path) -> None:
    torch.manual_seed(7)
    features, constraints, batch = _inputs()
    encoder = TypedGraphEncoder(
        hidden=16,
        heads=4,
        graph_layers=1,
        relation_specific=True,
        global_spec_conditioning=True,
        node_semantic_dim=2,
        port_semantic_dim=2,
    )
    graph_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {"boxes", "constraints"}
    }
    state = encoder.encode_channels(features, constraints, **graph_inputs)
    expected = encoder.fuse_channels(state).block_tokens
    path = tmp_path / "case.npz"
    write_typed_channel_state(path, state, metadata={"case_id": "case"})
    restored, metadata = read_typed_channel_state(path, device="cpu")
    actual = encoder.fuse_channels(restored).block_tokens
    torch.testing.assert_close(actual, expected)
    assert metadata["case_id"] == "case"
    assert restored.group_sizes == state.group_sizes


def test_fingerprints_are_content_and_state_sensitive() -> None:
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([1.0, 3.0])
    assert input_fingerprint({"x": first}) != input_fingerprint({"x": second})
    encoder = TypedGraphEncoder(hidden=16, heads=4, graph_layers=1)
    before = encoder_state_fingerprint(encoder)
    with torch.no_grad():
        next(encoder.parameters()).add_(1.0)
    after = encoder_state_fingerprint(encoder)
    assert before != after
    assert cache_key(before, "sample") != cache_key(after, "sample")


def test_cached_states_collate_with_variable_port_counts() -> None:
    encoder = TypedGraphEncoder(hidden=16, heads=4, graph_layers=1)
    states = []
    expected = []
    cases = []
    for pin_count in (1, 3):
        torch.manual_seed(10 + pin_count)
        features = torch.randn(1, 4, 19)
        constraints = torch.zeros(1, 4, 5)
        constraints[:, :, 3] = torch.tensor([1, 1, 2, 2])
        pins = torch.rand(1, pin_count, 2)
        p2b = torch.stack(
            [
                torch.arange(pin_count).float(),
                torch.arange(pin_count).float() % 4,
                torch.ones(pin_count),
            ],
            dim=-1,
        )[None]
        state = encoder.encode_channels(
            features,
            constraints,
            areas=torch.ones(1, 4),
            b2b=torch.empty(1, 0, 3),
            p2b=p2b,
            pins=pins,
            n_b2b=torch.zeros(1, dtype=torch.long),
            n_p2b=torch.full((1,), pin_count),
            n_pins=torch.full((1,), pin_count),
        )
        states.append(state)
        expected.append(encoder.fuse_channels(state).block_tokens)
        cases.append((features, constraints, pins, p2b, pin_count))
    collated = collate_typed_channel_states(states)
    fused = encoder.fuse_channels(collated).block_tokens
    assert fused.shape == (2, 4, 16)
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused[0:1], expected[0])
    torch.testing.assert_close(fused[1:2], expected[1])
    max_pins = max(case[-1] for case in cases)
    features = torch.cat([case[0] for case in cases])
    constraints = torch.cat([case[1] for case in cases])
    pins = torch.cat(
        [
            torch.nn.functional.pad(
                case[2],
                (0, 0, 0, max_pins - case[-1]),
            )
            for case in cases
        ]
    )
    p2b = torch.cat(
        [
            torch.nn.functional.pad(
                case[3],
                (0, 0, 0, max_pins - case[-1]),
            )
            for case in cases
        ]
    )
    online = encoder(
        features,
        constraints,
        areas=torch.ones(2, 4),
        b2b=torch.empty(2, 0, 3),
        p2b=p2b,
        pins=pins,
        n_b2b=torch.zeros(2, dtype=torch.long),
        n_p2b=torch.tensor([case[-1] for case in cases]),
        n_pins=torch.tensor([case[-1] for case in cases]),
    ).block_tokens
    torch.testing.assert_close(fused, online)


def test_case_cache_loader_collates_saved_states(tmp_path) -> None:
    encoder = TypedGraphEncoder(hidden=16, heads=4, graph_layers=1)
    states = []
    for index in range(2):
        constraints = torch.zeros(1, 4, 5)
        constraints[:, :, 3] = torch.tensor([1, 1, 2, 2])
        states.append(
            encoder.encode_channels(
                torch.randn(1, 4, 19),
                constraints,
                areas=torch.ones(1, 4),
                b2b=torch.empty(1, 0, 3),
                p2b=torch.empty(1, 0, 3),
                pins=torch.empty(1, 0, 2),
                n_b2b=torch.zeros(1, dtype=torch.long),
                n_p2b=torch.zeros(1, dtype=torch.long),
                n_pins=torch.zeros(1, dtype=torch.long),
            )
        )
        write_typed_channel_state(
            typed_channel_case_path(tmp_path, "train", index),
            states[-1],
            metadata={"case_index": index},
        )
    loaded = load_typed_channel_states(
        tmp_path,
        "train",
        [0, 1],
        device="cpu",
    )
    assert loaded.block_tokens.shape == (2, 4, 16)


def test_hierarchy_context_uses_cache_when_channel_encoding_is_frozen(
    tmp_path,
) -> None:
    model = HierarchicalTokenFlowModel(
        hidden=16,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
    )
    _features, _constraints, batch = _inputs()
    state = encode_typed_channel_state(model, batch)
    write_typed_channel_state(
        typed_channel_case_path(tmp_path, "train", 7),
        state,
        metadata={"case_index": 7},
    )
    apply_training_scope(model, parse_training_scope("fusion"))
    model.typed_channel_cache_root = str(tmp_path)
    model.typed_channel_cache_split = "train"
    batch["_indices"] = np.asarray([7])

    def fail_online(*_args, **_kwargs):
        raise AssertionError("online channel encoding should be bypassed")

    model.typed.encode_channels = fail_online
    context = model.prepare_context(batch)
    assert context.typed.block_tokens.shape == (1, 4, 16)


def test_sharded_cache_loader_restores_selected_cases(tmp_path) -> None:
    encoder = TypedGraphEncoder(hidden=16, heads=4, graph_layers=1)
    states = []
    for pin_count in (1, 2):
        constraints = torch.zeros(1, 4, 5)
        constraints[:, :, 3] = torch.tensor([1, 1, 2, 2])
        p2b = torch.stack(
            [
                torch.arange(pin_count).float(),
                torch.arange(pin_count).float() % 4,
                torch.ones(pin_count),
            ],
            dim=-1,
        )[None]
        states.append(
            encoder.encode_channels(
                torch.randn(1, 4, 19),
                constraints,
                areas=torch.ones(1, 4),
                b2b=torch.empty(1, 0, 3),
                p2b=p2b,
                pins=torch.rand(1, pin_count, 2),
                n_b2b=torch.zeros(1, dtype=torch.long),
                n_p2b=torch.full((1,), pin_count),
                n_pins=torch.full((1,), pin_count),
            )
        )
    write_typed_channel_shard(
        tmp_path,
        "train",
        4,
        [3, 9],
        states,
        metadata={"source": "test"},
    )
    (tmp_path / "train" / "shards.json").write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "block_count": 4,
                        "case_ids": ["3", "9"],
                        "path": str(tmp_path / "train" / "shard_0004.npz"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_typed_channel_states(
        tmp_path,
        "train",
        [3, 9],
        device="cpu",
    )
    expected = torch.cat(
        [encoder.fuse_channels(state).block_tokens for state in states]
    )
    actual = encoder.fuse_channels(loaded).block_tokens
    torch.testing.assert_close(actual, expected)
