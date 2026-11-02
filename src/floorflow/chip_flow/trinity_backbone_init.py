"""Initialize a typed chip hierarchy from the pinned Trinity backbone."""

from __future__ import annotations

import hashlib
from pathlib import Path

from safetensors.torch import load_file


def initialize_trinity_backbone(
    model: object,
    release_path: str | Path,
) -> dict[str, object]:
    """Load the release backbone strictly without touching typed modules."""

    root = Path(release_path)
    weights = root / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("model does not expose a backbone")
    state = load_file(str(weights), device="cpu")
    incompatible = backbone.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for name in incompatible.missing_keys
        if name.startswith("hierarchy_bridge.")
    }
    disallowed_missing = set(incompatible.missing_keys) - allowed_missing
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Trinity backbone state did not load exactly: "
            f"Missing keys={sorted(disallowed_missing)}, "
            f"Unexpected keys={sorted(incompatible.unexpected_keys)}"
        )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    return {
        "release_path": str(root),
        "weights_sha256": digest,
        "backbone_parameters": sum(value.numel() for value in state.values()),
        "strict_load": True,
        "fresh_bridge_parameters": sum(
            parameter.numel()
            for name, parameter in backbone.named_parameters()
            if name.startswith("hierarchy_bridge.")
        ),
    }


__all__ = ["initialize_trinity_backbone"]
