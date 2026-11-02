"""Exact disk cache for pre-fusion typed channel states."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from floorflow.chip_flow.typed_channel_state import TypedChannelState

_TENSOR_FIELDS = (
    "block_tokens",
    "global_token",
    "structural_tokens",
    "constraint_tokens",
    "port_tokens",
    "structural_mask",
    "constraint_mask",
    "port_mask",
    "port_block_incidence",
    "allowed",
)


def _hash_update(digest, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())


def encoder_state_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        _hash_update(digest, name, tensor)
    return digest.hexdigest()


def input_fingerprint(tensors: dict[str, torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        if value is not None:
            _hash_update(digest, name, value)
    return digest.hexdigest()


def cache_key(encoder_fingerprint: str, sample_fingerprint: str) -> str:
    return hashlib.sha256(
        f"{encoder_fingerprint}:{sample_fingerprint}".encode("ascii")
    ).hexdigest()


def typed_channel_case_path(
    root: str | Path,
    split: str,
    case_id: str | int,
) -> Path:
    return Path(root) / split / f"{case_id}.npz"


def typed_channel_shard_path(
    root: str | Path,
    split: str,
    block_count: int,
) -> Path:
    return Path(root) / split / f"shard_{int(block_count):04d}.npz"


def write_typed_channel_state(
    path: Path,
    state: TypedChannelState,
    *,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: getattr(state, name).detach().cpu().numpy()
        for name in _TENSOR_FIELDS
    }
    if state.relation_memories is not None:
        for index, tensor in enumerate(state.relation_memories):
            arrays[f"relation_memory_{index}"] = tensor.detach().cpu().numpy()
    if state.relation_masks is not None:
        for index, tensor in enumerate(state.relation_masks):
            arrays[f"relation_mask_{index}"] = tensor.detach().cpu().numpy()
    for index, (name, tensor) in enumerate(sorted(state.diagnostics.items())):
        arrays[f"diagnostic_{index}"] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
    sidecar = {
        **metadata,
        "group_sizes": state.group_sizes,
        "diagnostic_names": sorted(state.diagnostics),
        "relation_memory_count": len(state.relation_memories or ()),
        "relation_mask_count": len(state.relation_masks or ()),
    }
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )


def write_typed_channel_shard(
    root: str | Path,
    split: str,
    block_count: int,
    case_ids: list[int],
    states: list[TypedChannelState],
    *,
    metadata: dict[str, object],
) -> Path:
    if len(case_ids) != len(states):
        raise ValueError("case ids and states must have equal length")
    block_counts = {state.block_tokens.shape[1] for state in states}
    if block_counts != {int(block_count)}:
        raise ValueError("shard states must share the declared block count")
    path = typed_channel_shard_path(root, split, block_count)
    collated = collate_typed_channel_states(states)
    write_typed_channel_state(
        path,
        collated,
        metadata={
            **metadata,
            "block_count": int(block_count),
            "case_ids": [str(case_id) for case_id in case_ids],
            "port_counts": [
                int(state.port_tokens.shape[1]) for state in states
            ],
        },
    )
    return path


def read_typed_channel_state(
    path: Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> tuple[TypedChannelState, dict[str, object]]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    with np.load(path) as payload:
        def load(name: str, *, floating: bool = True) -> torch.Tensor:
            value = torch.from_numpy(payload[name])
            if floating and dtype is not None:
                value = value.to(dtype)
            return value.to(device)

        values = {
            name: load(name, floating=not name.endswith("mask") and not name in {
                "port_block_incidence",
                "allowed",
            })
            for name in _TENSOR_FIELDS
        }
        memory_count = int(metadata["relation_memory_count"])
        mask_count = int(metadata["relation_mask_count"])
        relation_memories = (
            tuple(load(f"relation_memory_{index}") for index in range(memory_count))
            if memory_count
            else None
        )
        relation_masks = (
            tuple(
                load(f"relation_mask_{index}", floating=False)
                for index in range(mask_count)
            )
            if mask_count
            else None
        )
        diagnostics = {
            name: load(f"diagnostic_{index}")
            for index, name in enumerate(metadata["diagnostic_names"])
        }
    state = TypedChannelState(
        **values,
        group_sizes=(
            tuple(metadata["group_sizes"])
            if metadata.get("group_sizes") is not None
            else None
        ),
        relation_memories=relation_memories,
        relation_masks=relation_masks,
        diagnostics=diagnostics,
    )
    return state, metadata


def collate_typed_channel_states(
    states: list[TypedChannelState],
) -> TypedChannelState:
    """Collate cached cases with equal block but variable port counts."""

    if not states:
        raise ValueError("at least one typed channel state is required")
    block_counts = {state.block_tokens.shape[1] for state in states}
    if len(block_counts) != 1:
        raise ValueError("cached states must share a block count")
    max_ports = max(state.port_tokens.shape[1] for state in states)

    def pad_ports(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        missing = max_ports - tensor.shape[1]
        if missing == 0:
            return tensor
        padding = (0, missing) if tensor.ndim == 2 else (0, 0, 0, missing)
        return F.pad(tensor, padding, value=value)

    def pad_port_last(tensor: torch.Tensor, value: bool = False) -> torch.Tensor:
        missing = max_ports - tensor.shape[-1]
        if missing == 0:
            return tensor
        return F.pad(tensor, (0, missing), value=value)

    port_tokens = torch.cat([pad_ports(state.port_tokens) for state in states])
    port_masks = torch.cat(
        [pad_ports(state.port_mask, value=False) for state in states]
    )
    incidence = torch.cat(
        [pad_port_last(state.port_block_incidence) for state in states]
    )
    allowed = torch.cat(
        [
            F.pad(
                state.allowed,
                (0, max_ports - state.port_tokens.shape[1]),
                value=False,
            )
            for state in states
        ]
    )
    diagnostics = {
        name: torch.stack([state.diagnostics[name] for state in states]).mean()
        for name in states[0].diagnostics
    }
    group_sizes = states[0].group_sizes
    if group_sizes is not None:
        group_sizes = (*group_sizes[:-1], max_ports)
    relation_memories = None
    relation_masks = None
    if states[0].relation_memories is not None:
        memories = []
        for index in range(len(states[0].relation_memories)):
            values = [state.relation_memories[index] for state in states]
            memories.append(
                torch.cat(
                    [
                        pad_ports(value) if index == 3 else value
                        for value in values
                    ]
                )
            )
        relation_memories = tuple(memories)
    if states[0].relation_masks is not None:
        masks = []
        for index in range(len(states[0].relation_masks)):
            values = [state.relation_masks[index] for state in states]
            masks.append(
                torch.cat(
                    [
                        pad_port_last(value) if index == 3 else value
                        for value in values
                    ]
                )
            )
        relation_masks = tuple(masks)
    return TypedChannelState(
        block_tokens=torch.cat([state.block_tokens for state in states]),
        global_token=torch.cat([state.global_token for state in states]),
        structural_tokens=torch.cat(
            [state.structural_tokens for state in states]
        ),
        constraint_tokens=torch.cat(
            [state.constraint_tokens for state in states]
        ),
        port_tokens=port_tokens,
        structural_mask=torch.cat(
            [state.structural_mask for state in states]
        ),
        constraint_mask=torch.cat(
            [state.constraint_mask for state in states]
        ),
        port_mask=port_masks,
        port_block_incidence=incidence,
        allowed=allowed,
        group_sizes=group_sizes,
        relation_memories=relation_memories,
        relation_masks=relation_masks,
        diagnostics=diagnostics,
    )


def load_typed_channel_states(
    root: str | Path,
    split: str,
    case_ids: list[str | int],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> TypedChannelState:
    manifest_path = Path(root) / split / "shards.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        requested = {str(case_id) for case_id in case_ids}
        matching = [
            shard
            for shard in manifest["shards"]
            if requested.intersection(shard["case_ids"])
        ]
        if len(matching) != 1 or not requested.issubset(
            set(matching[0]["case_ids"])
        ):
            raise ValueError("requested cases must belong to one cache shard")
        shard = matching[0]
        state, metadata = _read_shard_cached(str(shard["path"]))
        positions = {case: index for index, case in enumerate(metadata["case_ids"])}
        selected = [
            _select_shard_case(
                state,
                metadata,
                positions[str(case_id)],
            )
            for case_id in case_ids
        ]
        return _state_to(
            collate_typed_channel_states(selected),
            device=device,
            dtype=dtype,
        )
    states = [
        read_typed_channel_state(
            typed_channel_case_path(root, split, case_id),
            device=device,
            dtype=dtype,
        )[0]
        for case_id in case_ids
    ]
    return collate_typed_channel_states(states)


@lru_cache(maxsize=8)
def _read_shard_cached(path: str) -> tuple[TypedChannelState, dict]:
    return read_typed_channel_state(Path(path), device="cpu")


def _state_to(
    state: TypedChannelState,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None,
) -> TypedChannelState:
    def move(tensor: torch.Tensor, *, floating: bool = True) -> torch.Tensor:
        value = tensor.to(device)
        return value.to(dtype) if floating and dtype is not None else value

    return TypedChannelState(
        block_tokens=move(state.block_tokens),
        global_token=move(state.global_token),
        structural_tokens=move(state.structural_tokens),
        constraint_tokens=move(state.constraint_tokens),
        port_tokens=move(state.port_tokens),
        structural_mask=move(state.structural_mask, floating=False),
        constraint_mask=move(state.constraint_mask, floating=False),
        port_mask=move(state.port_mask, floating=False),
        port_block_incidence=move(
            state.port_block_incidence,
            floating=False,
        ),
        allowed=move(state.allowed, floating=False),
        group_sizes=state.group_sizes,
        relation_memories=(
            tuple(move(value) for value in state.relation_memories)
            if state.relation_memories is not None
            else None
        ),
        relation_masks=(
            tuple(
                move(value, floating=False)
                for value in state.relation_masks
            )
            if state.relation_masks is not None
            else None
        ),
        diagnostics={
            name: move(value)
            for name, value in state.diagnostics.items()
        },
    )


def _select_shard_case(
    state: TypedChannelState,
    metadata: dict,
    position: int,
) -> TypedChannelState:
    port_count = int(metadata["port_counts"][position])
    max_ports = max(int(value) for value in metadata["port_counts"])
    base_allowed = state.allowed.shape[-1] - max_ports

    def select(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[position : position + 1]

    def trim_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        value = select(tensor)
        index = [slice(None)] * value.ndim
        index[dim] = slice(0, port_count)
        return value[tuple(index)]

    allowed = select(state.allowed)
    allowed = torch.cat(
        [
            allowed[..., :base_allowed],
            allowed[..., base_allowed : base_allowed + port_count],
        ],
        dim=-1,
    )
    relation_memories = None
    relation_masks = None
    if state.relation_memories is not None:
        relation_memories = tuple(
            trim_dim(value, 1) if index == 3 else select(value)
            for index, value in enumerate(state.relation_memories)
        )
    if state.relation_masks is not None:
        relation_masks = tuple(
            trim_dim(value, -1) if index == 3 else select(value)
            for index, value in enumerate(state.relation_masks)
        )
    group_sizes = state.group_sizes
    if group_sizes is not None:
        group_sizes = (*group_sizes[:-1], port_count)
    return TypedChannelState(
        block_tokens=select(state.block_tokens),
        global_token=select(state.global_token),
        structural_tokens=select(state.structural_tokens),
        constraint_tokens=select(state.constraint_tokens),
        port_tokens=trim_dim(state.port_tokens, 1),
        structural_mask=select(state.structural_mask),
        constraint_mask=select(state.constraint_mask),
        port_mask=trim_dim(state.port_mask, 1),
        port_block_incidence=trim_dim(state.port_block_incidence, -1),
        allowed=allowed,
        group_sizes=group_sizes,
        relation_memories=relation_memories,
        relation_masks=relation_masks,
        diagnostics=state.diagnostics,
    )


__all__ = [
    "cache_key",
    "collate_typed_channel_states",
    "encoder_state_fingerprint",
    "input_fingerprint",
    "load_typed_channel_states",
    "read_typed_channel_state",
    "typed_channel_case_path",
    "typed_channel_shard_path",
    "write_typed_channel_shard",
    "write_typed_channel_state",
]
