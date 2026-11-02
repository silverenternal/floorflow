from __future__ import annotations

import json
import sys

import pytest
import torch

from floorflow.data.floorplan_token_targets import (
    TokenTargetOverflow,
    build_floorplan_token_vocabulary,
    token_batch_from_plan,
    training_token_sequence,
)
from floorflow.data.rplan_boundary_adapter import rplan_taxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalOpening,
    CanonicalRoom,
    CanonicalWall,
)
from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.heads.query_aware_pool import QueryAwarePool
from floorflow.heads.typed_token_memory import TypedTokenMemory
from floorflow.inference.floorplan_token_generation import (
    generate_floorplan_tokens,
)
from floorflow.objectives.token_grammar import (
    TOKEN_STAGES,
    FloorplanTokenVocabulary,
    grammar_allowed_mask,
    grammar_masks_for_ids,
    stage_ids_for_tokens,
    token_sequence_loss,
)
from floorflow.objectives.token_structure_consistency import (
    token_structure_consistency_loss,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken
from floorflow.training.floorplan_token_training_runtime import (
    build_token_training_runtime,
)
from floorflow.training.rplan_generator_cli import parse_args


def _plan() -> CanonicalFloorplan:
    living = ((1, 1), (15, 1), (15, 15), (1, 15))
    kitchen = ((15, 1), (31, 1), (31, 15), (15, 15))
    return CanonicalFloorplan(
        case_id="case-specific-id",
        track="rplan_generation",
        width=32,
        height=32,
        footprint=((0, 0), (32, 0), (32, 16), (0, 16)),
        rooms=(
            CanonicalRoom("room_1", "room:LivingRoom", living, living),
            CanonicalRoom("room_2", "room:Kitchen", kitchen, kitchen),
        ),
        walls=(
            CanonicalWall("wall_1", (0, 0), (32, 0), 1),
            CanonicalWall("wall_2", (32, 0), (32, 16), 1),
            CanonicalWall("wall_3", (32, 16), (0, 16), 1),
            CanonicalWall("wall_4", (0, 16), (0, 0), 1),
            CanonicalWall("wall_5", (15, 1), (15, 15), 1),
        ),
        openings=(
            CanonicalOpening("opening_1", "door:Door", (15, 8), 3, "wall_5"),
        ),
        furniture=(),
        adjacency=(("room_1", "room_2"),),
        source_hash="private-source-hash",
    )


def test_training_targets_remove_identity_and_satisfy_grammar() -> None:
    taxonomy = rplan_taxonomy()
    tokens = training_token_sequence(_plan(), taxonomy)
    rendered = "\t".join(token.render() for token in tokens)
    assert "case-specific-id" not in rendered
    assert "private-source-hash" not in rendered
    assert "VERTEX|<REF>|<REF>" in rendered
    assert "JUNCTION|junction_" in rendered
    assert "WALL|wall_" in rendered
    vocabulary, statistics = build_floorplan_token_vocabulary([tokens])
    assert vocabulary.model_vocab_size < 100
    try:
        token_batch_from_plan(
            _plan(),
            taxonomy,
            vocabulary,
            max_length=len(tokens) - 1,
        )
    except TokenTargetOverflow:
        pass
    else:
        raise AssertionError("long token targets must fail closed")
    batch = token_batch_from_plan(
        _plan(),
        taxonomy,
        vocabulary,
        max_length=statistics.recommended_max_length,
    )
    assert batch.oov_count == 0
    assert batch.length == len(tokens)
    assert batch.mask.sum() == batch.length
    assert batch.loss_mask.sum() == batch.length
    assert set(batch.stage_ids[: batch.length].tolist()) == set(
        stage_ids_for_tokens(tokens)
    )
    grammar_mask = grammar_masks_for_ids(
        batch.ids[None],
        vocabulary,
        batch.loss_mask[None],
    )
    valid_ids = batch.ids[batch.loss_mask]
    valid_grammar = grammar_mask[0, batch.loss_mask]
    allowed = valid_grammar.gather(-1, valid_ids[:, None]).squeeze(-1)
    assert allowed.all()
    logits = torch.zeros(
        1,
        batch.ids.shape[0],
        vocabulary.model_vocab_size,
        requires_grad=True,
    )
    loss = token_sequence_loss(
        logits,
        batch.ids[None],
        batch.loss_mask[None],
        grammar_mask=grammar_mask,
        token_weights=batch.loss_mask[None].float(),
    )
    assert torch.isfinite(loss)
    structure_loss, structure_components = token_structure_consistency_loss(
        batch.ids[None],
        batch.loss_mask[None],
        vocabulary,
        room_exists_logits=torch.zeros(1, 4, requires_grad=True),
        wall_exists_logits=torch.zeros(1, 8, requires_grad=True),
        opening_exists_logits=torch.zeros(1, 2, requires_grad=True),
        room_mask_logits=torch.zeros(1, 4, 8, 8, requires_grad=True),
    )
    assert torch.isfinite(structure_loss)
    assert structure_components["token_room_partition_excess"] >= 0


def test_vocabulary_roundtrip_and_unknown_contract(tmp_path) -> None:
    vocabulary = FloorplanTokenVocabulary(
        ("SCHEMA|v", "END"),
    )
    path = tmp_path / "vocabulary.json"
    vocabulary.write(path)
    restored = FloorplanTokenVocabulary.read(path)
    assert restored.sha256() == vocabulary.sha256()
    assert restored.index("not-in-vocabulary") == restored.unk_id
    assert restored.decode(restored.pad_id) == restored.pad_token
    payload = json.loads(path.read_text())
    assert payload["sha256"] == vocabulary.sha256()


def test_query_pool_ignores_inactive_queries() -> None:
    pool = QueryAwarePool(4)
    with torch.no_grad():
        pool.score.weight.zero_()
        pool.score.bias.zero_()
    present = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    inactive = torch.full((1, 1, 4), 1.0e4)
    features = torch.cat([present, inactive], dim=1)
    pooled = pool(
        features,
        torch.tensor([[20.0, -20.0]]),
    )
    assert torch.allclose(pooled, present[:, 0])


def test_typed_memory_and_generation_respect_grammar() -> None:
    memory = TypedTokenMemory(8, ("envelope", "rooms", "walls", "openings"))
    boundary = torch.randn(1, 3, 8)
    rooms = torch.randn(1, 2, 8)
    stages = torch.randn(1, len(TOKEN_STAGES), 8)
    vector = {name: torch.randn(1, 8) for name in memory.vector_stage_names}
    context, valid = memory(
        boundary,
        torch.tensor([[True, True, False]]),
        rooms,
        torch.tensor([[True, False]]),
        stages,
        vector,
    )
    assert context.shape == (1, 17, 8)
    assert valid.sum() == 15

    vocabulary = FloorplanTokenVocabulary(
        (
            "QUANTIZER|8",
            "FOOTPRINT_BEGIN",
            "VERTEX|1|1",
            "RING_END",
            "END",
        )
    )
    head = FloorplanTokenHead(8, vocabulary.model_vocab_size, 8, hidden_dim=8)
    head.eval()
    prompt = (
        FloorplanToken("SCHEMA", ("v",)),
        FloorplanToken("CASE", ("<CASE>",)),
        FloorplanToken("TRACK", ("rplan_generation",)),
        FloorplanToken("SIZE", ("32", "32")),
        FloorplanToken("SOURCE_HASH", ("<SOURCE>",)),
    )
    allowed = grammar_allowed_mask(prompt, vocabulary)
    assert allowed[vocabulary.index("QUANTIZER|8")]
    with pytest.raises(ValueError, match="without END"):
        generate_floorplan_tokens(
            head,
            context,
            valid,
            vocabulary,
            prompt,
            max_new_tokens=1,
        )


def test_token_runtime_contract_and_checkpoint(tmp_path) -> None:
    vocabulary = FloorplanTokenVocabulary(("SCHEMA|v", "END"))
    vocabulary.write(tmp_path / "vocabulary.json")
    runtime = build_token_training_runtime(
        {
            "enabled": True,
            "vocabulary_path": str(tmp_path / "vocabulary.json"),
            "max_length": 32,
            "weight": 0.2,
            "structure_weight": 0.1,
            "kind_weights": {"END": 2.0},
            "stage_weights": {"validation": 1.5},
            "warmup_steps": 2,
            "structure_start_step": 2,
            "structure_ramp_steps": 2,
            "conflict_projection": True,
            "conflict_pairs": [["token_structure", "wall"]],
            "generation_budget_multiplier": 1.25,
            "generation_efficiency_budget": 64,
            "scheduled_sampling_start": 0,
            "scheduled_sampling_ramp": 1,
            "token_dropout_prob": 1.0,
            "prefix_corruption_prob": 1.0,
            "token_rollout_steps": 1,
            "loss_balance": {"enabled": True, "update_interval": 2},
        },
        root=tmp_path,
        loss_weights={"token": 0.1},
    )
    model_config: dict[str, object] = {}
    runtime.apply_model_config(model_config)
    assert model_config["token_vocab_size"] == vocabulary.model_vocab_size
    assert runtime.loss_balancer is not None
    assert runtime.scheduled_token_weight() == pytest.approx(0.1)
    runtime.step = 1
    assert runtime.scheduled_structure_weight() == 0.0
    runtime.step = 2
    assert runtime.scheduled_structure_weight() > 0.0
    assert runtime.config.conflict_projection
    assert runtime.generation_budget(100) == 32
    token_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    kwargs = runtime.forward_kwargs(
        {
            "token_ids": token_ids,
            "token_mask": token_mask,
            "token_stage_ids": torch.zeros_like(token_ids),
        }
    )
    assert torch.equal(kwargs["token_input_ids"][:, :5], token_ids[:, :5])
    assert kwargs["token_rollout_steps"] == 1
    eval_kwargs = runtime.forward_kwargs(
        {
            "token_ids": token_ids,
            "token_mask": token_mask,
            "token_stage_ids": torch.zeros_like(token_ids),
        },
        training=False,
    )
    assert torch.equal(eval_kwargs["token_input_ids"], token_ids)
    assert eval_kwargs["token_rollout_steps"] == 0
    payload = runtime.checkpoint_payload()
    assert payload is not None
    assert payload["vocabulary_sha256"] == vocabulary.sha256()


def test_token_training_cli_overrides(tmp_path, monkeypatch) -> None:
    vocabulary_path = tmp_path / "vocabulary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--enable-token-head",
            "--token-vocabulary",
            str(vocabulary_path),
            "--token-max-length",
            "384",
            "--token-weight",
            "0.2",
            "--joint-mask-match-weight",
            "0.0",
            "--enable-joint-room-assignment",
        ],
    )
    config = parse_args()
    assert config.enable_token_head
    assert config.token_vocabulary == vocabulary_path
    assert config.token_max_length == 384
    assert config.token_weight == 0.2
    assert config.joint_mask_match_weight == 0.0
    assert config.enable_joint_room_assignment


__all__ = []
