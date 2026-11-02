"""Gate room geometry separately from room semantic assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import label

from floorflow.inference.room_repair import fill_room_support
from floorflow.recursion.trust_gate import (
    RoomGraphTrustGate,
    room_logit_trust_features,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan


@dataclass(frozen=True)
class GatedRoomOutput:
    room_labels: torch.Tensor
    support: torch.Tensor
    diagnostics: dict[str, float]


def compose_gated_room_layers(
    base_layers: LayeredFloorplan,
    graph_layers: LayeredFloorplan,
    base_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
    *,
    graph_confidence_threshold: float = 0.5,
    trust_gate: RoomGraphTrustGate | None = None,
    trust_threshold: float = 0.5,
) -> tuple[LayeredFloorplan, dict[str, float]]:
    """Keep base objects and replace only the gated room layer."""

    gate = gated_room_decoupling(
        base_layers.room_labels,
        graph_layers.room_labels,
        base_logits,
        graph_logits,
        footprint,
        taxonomy,
        graph_confidence_threshold=graph_confidence_threshold,
        trust_gate=trust_gate,
        trust_threshold=trust_threshold,
    )
    layers = LayeredFloorplan(
        room_labels=gate.room_labels,
        wall=base_layers.wall,
        opening=base_layers.opening,
        furniture=base_layers.furniture,
        footprint=footprint[:, 0] if footprint.ndim == 4 else footprint,
        taxonomy=taxonomy,
    )
    return layers, gate.diagnostics


def _room_confidence(
    logits: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim == 4:
        return logits.float().softmax(dim=1).amax(dim=1)
    if logits.ndim == 3:
        return logits.float().softmax(dim=0).amax(dim=0)
    raise ValueError("room logits must be [C,H,W] or [B,C,H,W]")


def gated_room_decoupling(
    base_labels: torch.Tensor,
    graph_labels: torch.Tensor,
    base_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
    *,
    graph_confidence_threshold: float = 0.5,
    reassign_graph_components: bool = True,
    trust_gate: RoomGraphTrustGate | None = None,
    trust_threshold: float = 0.5,
) -> GatedRoomOutput:
    """Keep semantic assignment and adopt only confidently improved geometry."""

    if base_labels.ndim == 4:
        base_labels = base_labels[:, 0]
    if graph_labels.ndim == 4:
        graph_labels = graph_labels[:, 0]
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    if base_labels.shape != graph_labels.shape:
        raise ValueError("base and graph room labels must share shape")
    if base_labels.shape != footprint.shape:
        raise ValueError("footprint must match room labels")
    if not 0.0 <= float(graph_confidence_threshold) <= 1.0:
        raise ValueError("graph_confidence_threshold must be in [0,1]")
    device = base_labels.device
    room_indices = torch.tensor(
        taxonomy.room_ids,
        device=device,
        dtype=torch.long,
    )
    base_support = torch.isin(base_labels, room_indices) & footprint.bool()
    graph_support = torch.isin(graph_labels, room_indices) & footprint.bool()
    graph_confidence = _room_confidence(graph_logits)
    trust = None
    if trust_gate is not None:
        trust = trust_gate.trust(
            room_logit_trust_features(
                base_logits,
                graph_logits,
                footprint,
            )
        )
    trusted_case = (
        torch.ones(base_labels.shape[0], device=device, dtype=torch.bool)
        if trust is None
        else trust >= float(trust_threshold)
    )
    accepted_geometry = graph_support & trusted_case[:, None, None] & (
        graph_confidence >= float(graph_confidence_threshold)
    )
    geometry_support = base_support | accepted_geometry
    composed = []
    room_ids_np = np.asarray(taxonomy.room_ids, dtype=np.int64)
    for index in range(base_labels.shape[0]):
        base_np = base_labels[index].cpu().numpy()
        graph_np = graph_labels[index].cpu().numpy()
        accepted_np = accepted_geometry[index].cpu().numpy()
        output = np.where(accepted_np, graph_np, 0).astype(base_np.dtype)
        if reassign_graph_components:
            for class_id in room_ids_np:
                components, count = label(
                    accepted_np & (graph_np == class_id)
                )
                for component_id in range(1, int(count) + 1):
                    component = components == component_id
                    if base_logits.ndim == 4:
                        semantic_scores = base_logits[index].float()[
                            :, component
                        ].mean(dim=1)
                    else:
                        semantic_scores = base_logits.float()[
                            :, component
                        ].mean(dim=1)
                    output[component] = room_ids_np[
                        int(semantic_scores.argmax())
                    ]
        outside = ~accepted_np & base_support[index].cpu().numpy()
        output[outside] = base_np[outside]
        output = fill_room_support(
            output,
            footprint[index].cpu().numpy().astype(bool),
            taxonomy,
            support_mask=geometry_support[index].cpu().numpy(),
        )
        composed.append(output)
    room_labels = torch.from_numpy(
        np.stack(composed)
    ).to(device=device, dtype=base_labels.dtype)
    denominator = max(1, int(footprint.sum()))
    confidence_rows = [
        graph_confidence[index][graph_support[index]]
        for index in range(base_labels.shape[0])
        if graph_support[index].any()
    ]
    confidence_mean = (
        torch.cat(confidence_rows).mean()
        if confidence_rows
        else graph_confidence.sum() * 0.0
    )
    diagnostics = {
        "accepted_graph_fraction": float(
            (accepted_geometry & ~base_support).sum()
        )
        / denominator,
        "base_support_fraction": float(base_support.sum()) / denominator,
        "gated_support_fraction": float(
            torch.isin(room_labels, room_indices).sum()
        )
        / denominator,
        "graph_confidence_mean": float(confidence_mean),
        "trust_gate_mean": float(trust.mean()) if trust is not None else 1.0,
        "trusted_case_fraction": float(trusted_case.float().mean()),
    }
    return GatedRoomOutput(
        room_labels=room_labels,
        support=torch.isin(room_labels, room_indices),
        diagnostics=diagnostics,
    )


__all__ = [
    "GatedRoomOutput",
    "compose_gated_room_layers",
    "gated_room_decoupling",
]
