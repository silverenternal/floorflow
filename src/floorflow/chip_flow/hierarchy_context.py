"""Context construction for the typed chip hierarchy model."""

from __future__ import annotations

import torch
from trinity.models.backbone import DenoiserCond

from floorflow.chip_flow.graph_conditioning import graph_conditioned_features
from floorflow.chip_flow.hierarchy_state import (
    HierarchyContext,
    StateEdgeContext,
)
from floorflow.chip_flow.hierarchy_structure import structural_membership
from floorflow.chip_flow.netlist_hypergraph import netlist_context_from_batch
from floorflow.chip_flow.sparse_hierarchy import graph_complexity
from floorflow.chip_flow.sparse_layers import prepare_state_edge_cache
from floorflow.chip_flow.training_scopes import (
    typed_channel_encoding_requires_grad,
)
from floorflow.chip_flow.typed_channel_cache import load_typed_channel_states


def _typed_batch_inputs(model, batch: dict):
    features, adjacency, key_mask = graph_conditioned_features(batch)
    graph_inputs = {
        "areas": batch["areas"],
        "b2b": batch["b2b"],
        "p2b": batch["p2b"],
        "pins": batch["pins"],
        "n_b2b": batch["n_b2b"],
        "n_p2b": batch["n_p2b"],
        "n_pins": batch["n_pins"],
    }
    if not model.use_netlist:
        features = features.clone()
        features[..., 3] = 0.0
        features[..., 16:19] = 0.0
        adjacency = torch.zeros_like(adjacency)
        graph_inputs["b2b"] = None
        graph_inputs["p2b"] = None
    global_spec = batch.get("global_spec")
    global_spec_mask = batch.get("global_spec_mask")
    node_features = batch.get("node_features")
    node_feature_mask = batch.get("node_feature_mask")
    port_features = batch.get("port_features")
    port_feature_mask = batch.get("port_feature_mask")
    typed_inputs = {
        "global_spec": global_spec,
        "global_spec_mask": global_spec_mask,
        "node_features": node_features,
        "node_feature_mask": node_feature_mask,
        "port_features": port_features,
        "port_feature_mask": port_feature_mask,
        **graph_inputs,
    }
    return (
        features,
        adjacency,
        key_mask,
        typed_inputs,
        global_spec,
        global_spec_mask,
        node_features,
        node_feature_mask,
        port_features,
        port_feature_mask,
    )


def _encode_typed_state(model, batch, features, typed_inputs):
    if typed_channel_encoding_requires_grad(model):
        return model.typed.encode_channels(
            features,
            batch["constraints"],
            **typed_inputs,
        )
    with torch.no_grad():
        return model.typed.encode_channels(
            features,
            batch["constraints"],
            **typed_inputs,
        )


def encode_typed_channel_state(model, batch: dict):
    """Return the cacheable pre-fusion typed channel state for one batch."""

    (
        features,
        _adjacency,
        _key_mask,
        typed_inputs,
        _global_spec,
        _global_spec_mask,
        _node_features,
        _node_feature_mask,
        _port_features,
        _port_feature_mask,
    ) = _typed_batch_inputs(
        model,
        batch,
    )
    return _encode_typed_state(model, batch, features, typed_inputs)


def prepare_hierarchy_context(model, batch: dict) -> HierarchyContext:
    """Build target-free typed graph and adapter context once per sample."""

    (
        features,
        adjacency,
        key_mask,
        typed_inputs,
        global_spec,
        global_spec_mask,
        node_features,
        node_feature_mask,
        port_features,
        port_feature_mask,
    ) = _typed_batch_inputs(
        model,
        batch,
    )
    cache_root = getattr(model, "typed_channel_cache_root", None)
    cache_indices = batch.get("_indices")
    if (
        cache_root is not None
        and cache_indices is not None
        and not typed_channel_encoding_requires_grad(model)
    ):
        channel_state = load_typed_channel_states(
            cache_root,
            getattr(model, "typed_channel_cache_split", "train"),
            [int(index) for index in cache_indices],
            device=features.device,
        )
    else:
        channel_state = _encode_typed_state(
            model,
            batch,
            features,
            typed_inputs,
        )
    typed = model.typed.fuse_channels(channel_state)
    condition = DenoiserCond(
        features=features,
        adjacency=adjacency,
        key_pad_mask=key_mask,
        graph_pe=model.graph_pe.batched(adjacency, key_mask),
    )
    outline_state = None
    if model.outline_conditioning:
        if "outline_state" not in batch:
            raise ValueError("outline conditioning requires batch['outline_state']")
        outline_state = batch["outline_state"].float()
        if outline_state.shape != (batch["areas"].shape[0], 4):
            raise ValueError("outline_state must have shape (batch, 4)")
    edge_state = None
    if (
        model.dynamic_relay is not None
        or model.state_conditioned_edge_bias is not None
        or model.spatial_route_field
    ):
        edge_state = StateEdgeContext(
            cache=prepare_state_edge_cache(
                areas=batch["areas"],
                b2b=batch.get("b2b"),
                p2b=batch.get("p2b"),
                pins=batch.get("pins"),
                n_b2b=batch.get("n_b2b"),
                n_p2b=batch.get("n_p2b"),
                n_pins=batch.get("n_pins"),
            )
        )
    order_cluster_ids = None
    order_cluster_mask = None
    pin_positions = None
    pin_mask = None
    if batch.get("pins") is not None:
        flow_scale = batch["areas"].sum(dim=1).sqrt().clamp_min(1.0e-6)
        pin_positions = (
            batch["pins"].float() / flow_scale[:, None, None]
        )
        pin_rows = torch.arange(
            pin_positions.shape[1],
            device=pin_positions.device,
        )[None]
        pin_counts = batch.get("n_pins")
        pin_mask = (
            torch.ones(
                pin_positions.shape[:2],
                dtype=torch.bool,
                device=pin_positions.device,
            )
            if pin_counts is None
            else pin_rows < pin_counts.to(pin_positions.device)[:, None]
        )
    if model.order_staging is not None:
        raw_ids = batch["constraints"][..., 3].long()
        order_cluster_ids = torch.where(
            (raw_ids >= 1) & (raw_ids <= 4),
            raw_ids,
            torch.zeros_like(raw_ids),
        )
        order_cluster_mask = torch.stack(
            [order_cluster_ids.eq(group) for group in range(1, 5)],
            dim=-1,
        ).any(dim=1)
    return HierarchyContext(
        typed=typed,
        denoiser_cond=condition,
        outline_state=outline_state,
        edge_state=edge_state,
        order_cluster_ids=order_cluster_ids,
        order_cluster_mask=order_cluster_mask,
        coarse_membership=structural_membership(batch["constraints"]),
        block_areas=batch["areas"],
        complexity=graph_complexity(
            batch["constraints"],
            batch.get("b2b"),
            batch.get("n_b2b"),
        ),
        global_spec=global_spec,
        global_spec_mask=global_spec_mask,
        node_features=node_features,
        node_feature_mask=node_feature_mask,
        port_features=port_features,
        port_feature_mask=port_feature_mask,
        pin_positions=pin_positions,
        pin_mask=pin_mask,
        netlist=netlist_context_from_batch(batch),
        native_group_moment_state=model.group_moment_coupler is not None,
        constraints=batch["constraints"],
    )


__all__ = ["encode_typed_channel_state", "prepare_hierarchy_context"]
