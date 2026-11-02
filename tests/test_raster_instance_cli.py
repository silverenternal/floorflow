import sys

import pytest

from floorflow.training.raster_instance_cli import parse_args, validate_args


def test_enhanced_room_head_options_are_available(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--mask-pooled-class-fusion",
            "--full-resolution-class-sampler",
            "--owner-semantic-reassignment",
            "--room-boundary-owner-refiner",
            "--room-boundary-weight",
            "0.5",
            "--dense-identity-refiner",
            "--query-set-head",
            "--query-count-gate",
            "--query-gate-start-step",
            "100",
            "--room-program-head",
            "--program-token-conditioning",
            "--program-class-prior-weight",
            "0.25",
            "--partition-consistent",
            "--partition-confidence-weighting",
            "--adaptive-query-refiner",
            "--query-separation-mode",
            "supervised_same_class",
            "--query-separation-weight",
            "0.1",
            "--query-set-weight",
            "0.5",
            "--selection-count-weight",
            "0.1",
            "--selection-mode",
            "dense_relative_identity",
            "--train-sampling",
            "curriculum",
            "--train-pool-size",
            "1024",
        ],
    )
    args = parse_args()
    assert args.mask_pooled_class_fusion
    assert args.full_resolution_class_sampler
    assert args.owner_semantic_reassignment
    assert args.room_boundary_owner_refiner
    assert args.room_boundary_weight == 0.5
    assert args.dense_identity_refiner
    assert args.query_set_head
    assert args.query_count_gate
    assert args.query_gate_start_step == 100
    assert args.room_program_head
    assert args.program_token_conditioning
    assert args.program_class_prior_weight == 0.25
    assert args.partition_consistent
    assert args.partition_confidence_weighting
    assert args.adaptive_query_refiner
    assert args.query_separation_mode == "supervised_same_class"
    assert args.query_set_weight == 0.5
    assert args.selection_mode == "dense_relative_identity"
    assert args.train_sampling == "curriculum"
    assert args.train_pool_size == 1024


def test_dependent_room_head_options_are_validated(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--no-query-set-head",
            "--query-count-gate",
            "--query-set-weight",
            "0.5",
        ],
    )
    args = parse_args()
    with pytest.raises(ValueError):
        validate_args(args)


def test_joint_room_slot_state_replaces_legacy_count_head(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--no-query-set-head",
            "--no-query-count-gate",
            "--joint-room-slot-state",
            "--joint-slot-owner-fusion",
            "--slot-class-anchor-weight",
            "0.1",
            "--slot-owner-weight",
            "1.0",
            "--query-set-weight",
            "0.5",
        ],
    )
    args = parse_args()
    assert args.joint_room_slot_state
    assert args.joint_slot_owner_fusion
    assert args.slot_class_anchor_weight == 0.1
    assert args.slot_owner_weight == 1.0
    validate_args(args)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--no-query-set-head",
            "--no-query-count-gate",
            "--joint-room-slot-state",
            "--query-set-head",
        ],
    )
    conflict = parse_args()
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_args(conflict)


def test_slot_state_only_requires_joint_state(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--slot-state-only"],
    )
    args = parse_args()
    with pytest.raises(ValueError, match="joint-room-slot-state"):
        validate_args(args)


def test_page_floor_crop_and_untrusted_teacher_class_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--page-floor-crop"],
    )
    args = parse_args()
    assert args.page_floor_crop
    assert args.teacher_class_weight == 0.0
    validate_args(args)


def test_raster_training_exposes_shared_runtime_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--precision",
            "bf16",
            "--compile-model",
            "--compile-mode",
            "reduce-overhead",
            "--channels-last",
            "--no-tf32",
            "--triton-kernels",
        ],
    )
    args = parse_args()
    assert args.precision == "bf16"
    assert args.compile_model
    assert args.compile_mode == "reduce-overhead"
    assert args.channels_last
    assert not args.tf32
    assert args.triton_kernels


def test_raster_training_exposes_full_split_and_offsets(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--full-train",
            "--train-offset",
            "0",
            "--val-offset",
            "384",
        ],
    )
    args = parse_args()
    assert args.full_train
    assert args.train_offset == 0
    assert args.val_offset == 384
    validate_args(args)


def test_raster_training_defaults_to_balanced_room_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train"],
    )
    args = parse_args()
    assert args.limit_train == 512
    assert args.epochs == 16
    assert args.hidden_dim == 192
    assert args.queries == 32
    assert args.layers == 5
    assert args.query_set_head
    assert args.query_count_gate
    assert args.query_set_hidden == 192
    assert args.query_set_pool == 64
    assert args.query_set_layers == 2
    assert args.query_set_routing_attention
    assert args.query_set_route_router_count == 12
    assert args.query_set_route_top_k == 4
