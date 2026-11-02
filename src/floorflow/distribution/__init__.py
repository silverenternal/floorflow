"""Coupled pixel and semantic-element distributions."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CoupledDistributionHub": ("floorflow.distribution.hub", "CoupledDistributionHub"),
    "DistributionCFMBridge": ("floorflow.distribution.cfm_bridge", "DistributionCFMBridge"),
    "DistributionCFMModel": ("floorflow.distribution.cfm_model", "DistributionCFMModel"),
    "HierarchicalPixelDistributionHead": (
        "floorflow.distribution.pixel_head",
        "HierarchicalPixelDistributionHead",
    ),
    "SemanticElementDistributionHead": (
        "floorflow.distribution.element_head",
        "SemanticElementDistributionHead",
    ),
    "element_set_loss": ("floorflow.distribution.element_loss", "element_set_loss"),
    "element_targets": ("floorflow.distribution.element_targets", "element_targets"),
    "joint_distribution_cfm_loss": (
        "floorflow.distribution.joint_loss",
        "joint_distribution_cfm_loss",
    ),
    "restore_distribution_model": (
        "floorflow.distribution.checkpoint",
        "restore_distribution_model",
    ),
    "save_distribution_model": (
        "floorflow.distribution.checkpoint",
        "save_distribution_model",
    ),
}

__all__ = tuple(sorted(_EXPORTS))


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
