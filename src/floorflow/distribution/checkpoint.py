"""Save and restore the coupled distribution-CFM model."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.model.cfm_structure_adapter import enable_cfm_structure
from floorflow.training.checkpoint import load_checkpoint_model


def _compatible_state(module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = module.state_dict()
    return {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }


def save_distribution_model(
    model: DistributionCFMModel,
    path: str | Path,
    *,
    taxonomy: FloorplanTaxonomy,
    extra: dict | None = None,
) -> None:
    payload = {
        "model": model.base.state_dict(),
        "pixel_head": model.hub.pixel.state_dict(),
        "element_head": model.hub.element.state_dict(),
        "cross_channel_prior": model.hub.cross_channel_prior.state_dict(),
        "element_feedback_scale": model.hub.element_feedback_scale.detach(),
        "cfm_bridge": model.cfm_bridge.state_dict(),
        "velocity_style_adapter": (
            model.velocity_style_adapter.state_dict()
            if model.velocity_style_adapter is not None
            else None
        ),
        "velocity_refiner": model.velocity_refiner.state_dict(),
        "taxonomy": taxonomy.to_json(),
        "config": {
            "hidden_channels": model.hidden_channels,
            "pixel_grid": model.pixel_grid,
            "element_grid": model.element_grid,
            "element_resolution": model.element_resolution,
            "queries_per_group": model.queries_per_group,
            "element_mask_version": model.element_mask_version,
            "num_styles": model.num_styles,
            "style_rank": model.style_rank,
            "velocity_refiner_hidden": model.velocity_refiner_hidden,
            "hierarchy_support_strength": model.hierarchy_support_strength,
            "element_blend_init": model.element_blend_init,
            "element_feedback_init": model.element_feedback_init,
            "bridge_mode": model.bridge_mode,
            "dynamic_head_variant": model.dynamic_head_variant,
            "dynamic_element_queries": model.dynamic_element_queries,
            "static_condition_encoder": bool(
                getattr(model.base, "static_condition_encoder", False)
            ),
            "typed_family_streams": bool(
                getattr(model.base, "typed_family_streams", False)
            ),
            "multiscale_structure_injection": bool(
                getattr(model.base, "multiscale_structure_injection", False)
            ),
            "architecture": "coupled_distribution_cfm",
            "bridge_blend_mode": "tanh",
        },
        **(extra or {}),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def restore_distribution_model(
    path: str | Path,
    device: str | torch.device,
    *,
    element_mask_version: str | None = None,
    num_styles: int | None = None,
    style_rank: int | None = None,
    hierarchy_support_strength: float | None = None,
    element_blend_init: float | None = None,
    element_feedback_init: float | None = None,
    dynamic_head_variant: str | None = None,
    dynamic_element_queries: bool | None = None,
    bridge_mode: str | None = None,
) -> tuple[DistributionCFMModel, FloorplanTaxonomy, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    base, _base_config, _taxonomy, _state = load_checkpoint_model(
        path,
        device,
        taxonomy=taxonomy,
    )
    config = payload.get("config", {})
    use_structure_fusion = bool(
        config.get("typed_family_streams", False)
        or config.get("multiscale_structure_injection", False)
    )
    if use_structure_fusion:
        enable_cfm_structure(
            base,
            taxonomy,
            static_condition_encoder=bool(
                config.get("static_condition_encoder", True)
            ),
            typed_family_streams=bool(
                config.get("typed_family_streams", True)
            ),
            multiscale_structure_injection=bool(
                config.get("multiscale_structure_injection", True)
            ),
        )
        base.load_state_dict(
            _compatible_state(base, _state),
            strict=False,
        )
    mask_version = element_mask_version or str(
        config.get("element_mask_version", "v1")
    )
    model = DistributionCFMModel(
        base,
        ChannelHierarchy.from_taxonomy(taxonomy),
        hidden_channels=int(config.get("hidden_channels", 96)),
        pixel_grid=int(config.get("pixel_grid", 16)),
        element_grid=int(config.get("element_grid", 16)),
        element_resolution=int(config.get("element_resolution", 64)),
        queries_per_group=config.get("queries_per_group"),
        element_mask_version=mask_version,
        num_styles=(
            int(num_styles)
            if num_styles is not None
            else int(config.get("num_styles", 0))
        ),
        style_rank=(
            int(style_rank)
            if style_rank is not None
            else int(config.get("style_rank", 8))
        ),
        velocity_refiner_hidden=int(
            config.get("velocity_refiner_hidden", 64)
        ),
        hierarchy_support_strength=(
            float(hierarchy_support_strength)
            if hierarchy_support_strength is not None
            else float(config.get("hierarchy_support_strength", 0.0))
        ),
        element_blend_init=(
            float(element_blend_init)
            if element_blend_init is not None
            else float(config.get("element_blend_init", 0.0))
        ),
        element_feedback_init=(
            float(element_feedback_init)
            if element_feedback_init is not None
            else float(config.get("element_feedback_init", 0.0))
        ),
        dynamic_head_variant=(
            str(dynamic_head_variant)
            if dynamic_head_variant is not None
            else str(config.get("dynamic_head_variant", "generated"))
        ),
        dynamic_element_queries=(
            bool(dynamic_element_queries)
            if dynamic_element_queries is not None
            else bool(config.get("dynamic_element_queries", False))
        ),
        bridge_mode=(
            str(bridge_mode)
            if bridge_mode is not None
            else str(config.get("bridge_mode", "blend"))
        ),
    ).to(device)
    model.hub.pixel.load_state_dict(
        _compatible_state(model.hub.pixel, payload["pixel_head"]),
        strict=False,
    )
    model.hub.element.load_state_dict(
        _compatible_state(model.hub.element, payload["element_head"]),
        strict=False,
    )
    if payload.get("cross_channel_prior") is not None:
        model.hub.cross_channel_prior.load_state_dict(
            _compatible_state(
                model.hub.cross_channel_prior,
                payload["cross_channel_prior"],
            ),
            strict=False,
        )
    if "element_feedback_scale" in payload:
        model.hub.element_feedback_scale.data.copy_(
            payload["element_feedback_scale"].to(device)
        )
    if element_feedback_init is not None:
        model.hub.element_feedback_scale.data.fill_(
            float(element_feedback_init)
        )
    model.cfm_bridge.load_state_dict(payload["cfm_bridge"], strict=False)
    if element_blend_init is not None:
        model.cfm_bridge.element_blend.data.fill_(
            float(element_blend_init)
        )
    if (
        model.velocity_style_adapter is not None
        and payload.get("velocity_style_adapter") is not None
    ):
        model.velocity_style_adapter.load_state_dict(
            _compatible_state(
                model.velocity_style_adapter,
                payload["velocity_style_adapter"],
            ),
            strict=False,
        )
    if "velocity_refiner" in payload:
        model.velocity_refiner.load_state_dict(
            _compatible_state(
                model.velocity_refiner,
                payload["velocity_refiner"],
            ),
            strict=False,
        )
    if config.get("bridge_blend_mode") != "tanh":
        model.cfm_bridge.element_blend.data.zero_()
    model.eval()
    return model, taxonomy, payload


__all__ = ["restore_distribution_model", "save_distribution_model"]
