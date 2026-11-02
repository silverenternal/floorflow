"""Backbone encode/decode operations kept separate from the module definition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.representation.conditioning import prepare_condition


def encode_condition_features(model, footprint, t, floor_ids=None):
    """Encode contour and floor conditions independently of the flow state."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if model.training and model.condition_dropout > 0.0:
        keep = (
            torch.rand(
                footprint.shape[0],
                1,
                1,
                1,
                device=footprint.device,
                dtype=footprint.dtype,
            )
            >= model.condition_dropout
        ).float()
        conditioned_footprint = footprint * keep
    else:
        conditioned_footprint = footprint
        keep = torch.ones(
            footprint.shape[0],
            1,
            1,
            1,
            device=footprint.device,
            dtype=footprint.dtype,
        )
    condition = prepare_condition(
        conditioned_footprint,
        fourier_conditioning=model.use_fourier_conditioning,
    )
    time_embedding = model.time_embedding(t)
    if model.use_floor_conditioning:
        if floor_ids is None:
            floor_ids = torch.ones(
                footprint.shape[0],
                device=footprint.device,
                dtype=torch.long,
            )
        floor_ids = floor_ids.to(
            device=footprint.device,
            dtype=torch.long,
        ).clamp(0, model.num_floors)
        floor_ids = (
            floor_ids * keep[:, 0, 0, 0].to(torch.long)
        ).clamp(0, model.num_floors)
        time_embedding = time_embedding + model.floor_embedding(floor_ids)
    model._last_condition_keep = keep
    model._last_condition_floor_ids = floor_ids
    return model.condition_encoder(condition, time_embedding)


def encode_features(
    model,
    x_t,
    t,
    footprint,
    floor_ids=None,
    condition_features=None,
):
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if condition_features is None:
        condition_features = encode_condition_features(
            model,
            footprint,
            t,
            floor_ids,
        )
    time_embedding = model.time_embedding(t)
    if model.use_floor_conditioning:
        if floor_ids is None and getattr(
            model,
            "_last_condition_floor_ids",
            None,
        ) is not None:
            floor_ids = model._last_condition_floor_ids
        elif floor_ids is None:
            floor_ids = torch.ones(
                x_t.shape[0],
                device=x_t.device,
                dtype=torch.long,
            )
        floor_ids = floor_ids.to(x_t.device, torch.long).clamp(
            0,
            model.num_floors,
        )
        keep = getattr(model, "_last_condition_keep", None)
        if keep is not None:
            floor_ids = (
                floor_ids * keep[:, 0, 0, 0].to(torch.long)
            ).clamp(0, model.num_floors)
        time_embedding = time_embedding + model.floor_embedding(floor_ids)
    structure_token = (
        model.structure_token_encoder(x_t, footprint)
        if model.structure_token_encoder is not None
        else None
    )

    hidden = model.input_projection(x_t)
    skips: list[torch.Tensor] = []
    for level, block in enumerate(model.encoder):
        if (
            model.structure_injector is not None
            and structure_token is not None
        ):
            hidden = model.structure_injector(
                level,
                hidden,
                structure_token,
            )
        hidden = block(hidden + condition_features[level], time_embedding)
        if level < len(model.encoder_down):
            skips.append(hidden)
            hidden = model.encoder_down[level](hidden)

    hidden = model.middle(hidden, time_embedding)
    hidden = model.condition_attention(hidden, condition_features[-1])
    if model.use_slot_mixing:
        hidden, _slot_diagnostics = model.slot_mixer(
            hidden, condition_features[-1]
        )
    hidden = model.semantic_channel_attention(hidden)
    if model.use_layoutbridge_refinement:
        hidden, _ddi = model.dual_stream_injection(
            hidden, condition_features[-1], t
        )
    if model.use_advanced_streams:
        query = hidden.flatten(2).transpose(1, 2)
        condition_tokens = condition_features[-1].flatten(2).transpose(1, 2)
        streams = (
            model.family_stream_evidence(x_t, condition_features[-1])
            if model.family_stream_evidence is not None
            else {
                "room": query,
                "wall": query,
                "opening": query,
                "furniture": query,
                "condition": condition_tokens,
                "direct": condition_tokens,
            }
        )
        mixed, _stream_gates = model.layout_stream_mixer(query, streams)
        hidden = mixed.transpose(1, 2).reshape_as(hidden)
    if model.use_family_streams:
        topology_tokens = (
            F.adaptive_max_pool2d(condition_features[-1], (4, 4))
            .flatten(2)
            .transpose(1, 2)
        )
        hidden, family_diagnostics = model.family_stream_mixer(
            hidden,
            condition_features[-1],
            topology_tokens,
        )
        model._last_family_diagnostics = family_diagnostics
    model._last_condition_features = condition_features[-1]

    for up, block in zip(model.decoder_up, model.decoder):
        hidden = up(hidden)
        skip = skips.pop()
        if hidden.shape[-2:] != skip.shape[-2:]:
            hidden = F.interpolate(
                hidden,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        hidden = block(torch.cat([hidden, skip], dim=1), time_embedding)
    return F.silu(model.output_norm(hidden))


def velocity_from_features(model, features, t=None):
    velocity = model.output(features)
    if model.use_dynamic_output_head:
        velocity = velocity + model.dynamic_output_head(features)
    if model.use_advanced_streams:
        pooled = F.adaptive_avg_pool2d(features, (16, 16))
        tokens = pooled.flatten(2).transpose(1, 2)
        condition = getattr(model, "_last_condition_features", None)
        streams = (
            model.velocity_family_streams(
                pooled,
                F.adaptive_avg_pool2d(condition, (16, 16)),
            )
            if (
                model.velocity_family_streams is not None
                and condition is not None
            )
            else {
                "room": tokens,
                "wall": tokens,
                "opening": tokens,
                "furniture": tokens,
                "condition": tokens,
            }
        )
        residual, _gates = model.spatial_velocity(
            tokens,
            streams,
        )
        residual = residual.transpose(1, 2).reshape(
            features.shape[0],
            model.target_channels,
            pooled.shape[-2],
            pooled.shape[-1],
        )
        velocity = velocity + F.interpolate(
            residual,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if model.use_layoutbridge_refinement and t is not None:
        velocity, _anisotropic = model.anisotropic_velocity(
            velocity, features, t
        )
    return velocity


__all__ = [
    "encode_condition_features",
    "encode_features",
    "velocity_from_features",
]
