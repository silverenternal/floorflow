"""Adapt predicted-control CAD outputs to the shared typed edit state."""

from __future__ import annotations

import torch

from floorflow.representation.typed_floorplan_state import TypedFloorplanState


def predicted_control_typed_state(
    output: dict,
) -> TypedFloorplanState:
    """Build an editable typed state from predicted room/action outputs."""

    payload = dict(output)
    room_exists = output.get("room_instance_query_existence_logits")
    selection = output.get(
        "query_count_candidate_mask",
        output.get("query_count_selection_mask"),
    )
    if room_exists is not None:
        active = (
            torch.ones_like(room_exists, dtype=torch.bool)
            if selection is None
            else selection.bool()
        )
        if active.shape != room_exists.shape:
            raise ValueError(
                "query count selection must match room existence slots"
            )
        _mask_inactive_query_slots(payload, active)
    payload.setdefault(
        "room_exists",
        payload.get("room_instance_query_existence_logits", room_exists),
    )
    payload.setdefault(
        "room_class",
        payload.get("room_instance_class_logits"),
    )
    required = (
        "room_identity_tokens",
        "room_exists",
        "room_class",
        "room_adjacency_logits",
        "action_stage_tokens",
    )
    missing = [name for name in required if payload.get(name) is None]
    if missing:
        raise ValueError(
            f"predicted-control edit state is missing: {missing}"
        )
    _validate_predicted_state_contract(payload)
    state = TypedFloorplanState.from_predictions(payload)
    if state.room_instance_ids is None:
        raise ValueError("predicted-control edit state has no room identity slots")
    return state


def _mask_inactive_query_slots(
    payload: dict[str, torch.Tensor],
    active: torch.Tensor,
) -> None:
    """Keep every query-indexed edit tensor on the same active object set."""

    existence = payload["room_instance_query_existence_logits"]
    payload["room_instance_query_existence_logits"] = existence.masked_fill(
        ~active,
        torch.finfo(existence.dtype).min,
    )
    instance_ids = payload.get("room_instance_ids")
    if instance_ids is None:
        instance_ids = torch.arange(
            active.shape[1],
            device=active.device,
        )[None].expand_as(active)
    if instance_ids.shape != active.shape:
        raise ValueError("room instance ids must match active query slots")
    payload["room_instance_ids"] = instance_ids.masked_fill(~active, -1)
    tokens = payload.get("room_identity_tokens")
    if tokens is not None:
        payload["room_identity_tokens"] = tokens.masked_fill(
            ~active[..., None],
            0.0,
        )
    classes = payload.get("room_instance_class_logits")
    if classes is not None:
        if not classes.is_floating_point():
            raise ValueError("room class logits must be floating point")
        if classes.ndim != 3 or classes.shape[:2] != active.shape:
            raise ValueError("room classes must align to active query slots")
        classes = classes.clone()
        classes[~active, :-1] = torch.finfo(classes.dtype).min
        classes[~active, -1] = 0.0
        payload["room_instance_class_logits"] = classes
    adjacency = payload.get("room_adjacency_logits")
    if adjacency is not None:
        if adjacency.shape != (
            active.shape[0],
            active.shape[1],
            active.shape[1],
        ):
            raise ValueError("room adjacency must be [B,Q,Q]")
        invalid = ~active[:, :, None] | ~active[:, None, :]
        payload["room_adjacency_logits"] = adjacency.masked_fill(
            invalid,
            torch.finfo(adjacency.dtype).min,
        )
    _mask_inactive_query_tensor(
        payload,
        "room_instance_mask_logits",
        active,
    )
    auxiliary = payload.get("room_instance_aux_mask_logits")
    if isinstance(auxiliary, (list, tuple)):
        payload["room_instance_aux_mask_logits"] = [
            _inactive_query_value(value, active)
            for value in auxiliary
        ]
    for name in (
        "program_room_assignment_logits",
        "program_room_scheduled_logits",
    ):
        _mask_inactive_query_tensor(payload, name, active)


def _inactive_query_value(
    value: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if value.ndim < 2 or value.shape[:2] != active.shape:
        raise ValueError(
            "query-indexed tensor must begin with [B,Q]"
        )
    if not value.is_floating_point():
        raise ValueError("query-indexed logits must be floating point")
    inactive = ~active
    while inactive.ndim < value.ndim:
        inactive = inactive.unsqueeze(-1)
    return value.masked_fill(
        inactive,
        torch.finfo(value.dtype).min,
    )


def _mask_inactive_query_tensor(
    payload: dict[str, torch.Tensor],
    name: str,
    active: torch.Tensor,
) -> None:
    value = payload.get(name)
    if value is not None:
        payload[name] = _inactive_query_value(value, active)


def _validate_predicted_state_contract(payload: dict) -> None:
    tokens = payload["room_identity_tokens"]
    existence = payload["room_exists"]
    classes = payload["room_class"]
    adjacency = payload["room_adjacency_logits"]
    instance_ids = payload["room_instance_ids"]
    if tokens.ndim != 3 or existence.ndim != 2:
        raise ValueError("room tokens/existence must be [B,Q,D]/[B,Q]")
    queries = tokens.shape[1]
    if existence.shape != (tokens.shape[0], queries):
        raise ValueError("room existence query width is inconsistent")
    if classes.ndim != 3 or classes.shape[:2] != existence.shape:
        raise ValueError("room classes must be [B,Q,C] aligned to existence")
    if adjacency.shape != (tokens.shape[0], queries, queries):
        raise ValueError("room adjacency must be [B,Q,Q]")
    if instance_ids.shape != existence.shape:
        raise ValueError("room instance ids must be [B,Q]")
    stages = payload["action_stage_tokens"]
    if stages.ndim != 3 or stages.shape[0] != tokens.shape[0]:
        raise ValueError("action stage tokens must be [B,S,D]")


__all__ = ["predicted_control_typed_state"]
