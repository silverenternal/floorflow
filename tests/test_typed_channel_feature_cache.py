from __future__ import annotations

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.model.predicted_optional_stages import resolve_field_features
from floorflow.training.typed_channel_feature_cache import (
    ensure_feature_cache_manifest,
    extract_frozen_features,
    read_frozen_batch,
    write_frozen_batch,
)


def test_frozen_feature_cache_roundtrip_preserves_model_output(tmp_path) -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Door": 2}),
            "window": Counter({"Window": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    structure = len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3
    model = build_predicted_channel_model(
        taxonomy,
        {
            "control_channels": [
                len(taxonomy.room_ids),
                structure,
                len(taxonomy.furniture_ids) + 1,
                2,
            ],
            "program_channels": 2,
            "grid": 8,
            "field_channels": 8,
            "head_channels": 16,
            "heads": 4,
            "layers": 1,
            "attention_grid": 4,
            "full_resolution_output": False,
            "typed_channel_links": True,
            "use_action_graph": False,
        },
    ).eval()
    controls = {
        "room_logits": torch.randn(1, len(taxonomy.room_ids), 8, 8),
        "structure": torch.rand(1, structure, 8, 8),
        "furniture": torch.rand(
            1,
            len(taxonomy.furniture_ids) + 1,
            8,
            8,
        ),
        "support": torch.rand(1, 2, 8, 8),
        "opening_family_probability": torch.rand(1, 2, 8, 8),
        "opening_foreground_probability": torch.rand(1, 8, 8),
        "opening_family_logits": torch.randn(1, 2, 8, 8),
        "junction_probability": torch.rand(1, 8, 8),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    footprint = torch.ones(1, 8, 8)
    features = extract_frozen_features(model, controls, footprint)
    write_frozen_batch(
        tmp_path,
        "train",
        ["case"],
        controls,
        features,
    )
    cached_controls, cached_features = read_frozen_batch(
        tmp_path,
        "train",
        ["case"],
        torch.device("cpu"),
    )
    for key in (
        "opening_family_probability",
        "opening_foreground_probability",
        "opening_family_logits",
        "junction_probability",
    ):
        assert torch.allclose(cached_controls[key], controls[key], atol=1.0e-6)
    full = model(controls, footprint)
    cached = model(
        cached_controls,
        footprint,
        cached_features=cached_features,
    )
    finite = torch.isfinite(full["logits"]) & torch.isfinite(cached["logits"])
    assert finite.any()
    assert torch.allclose(
        full["logits"][finite],
        cached["logits"][finite],
        atol=2.0e-3,
    )


def test_feature_cache_manifest_rejects_stale_root(tmp_path) -> None:
    ensure_feature_cache_manifest(tmp_path, "first")
    ensure_feature_cache_manifest(tmp_path, "first")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ensure_feature_cache_manifest(tmp_path, "second")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_features_preserve_accelerator_device() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    structure = len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3
    model = build_predicted_channel_model(
        taxonomy,
        {
            "control_channels": [
                len(taxonomy.room_ids),
                structure,
                len(taxonomy.furniture_ids) + 1,
                2,
            ],
            "program_channels": 2,
            "grid": 8,
            "field_channels": 8,
            "head_channels": 16,
            "heads": 4,
            "layers": 1,
            "attention_grid": 4,
            "full_resolution_output": False,
            "typed_channel_links": True,
            "use_action_graph": False,
        },
    ).to("cuda:0")
    controls = {
        "room_logits": torch.randn(
            1, len(taxonomy.room_ids), 8, 8, device="cuda:0"
        ),
        "structure": torch.rand(1, structure, 8, 8, device="cuda:0"),
        "furniture": torch.rand(
            1,
            len(taxonomy.furniture_ids) + 1,
            8,
            8,
            device="cuda:0",
        ),
        "support": torch.rand(1, 2, 8, 8, device="cuda:0"),
        "program": torch.tensor([[1, 0]], dtype=torch.bool, device="cuda:0"),
    }
    features = extract_frozen_features(
        model,
        controls,
        torch.ones(1, 8, 8, device="cuda:0"),
    )
    assert all(value.device.type == "cuda" for value in features.values())


def test_cached_field_features_fail_closed_on_missing_evidence() -> None:
    model = type(
        "Model",
        (),
        {"control_evidence_context": True, "field": None},
    )()
    cached = {
        "field_features": torch.zeros(2, 4, 2, 2),
        "source_tokens": torch.zeros(2, 6, 4),
    }
    with pytest.raises(ValueError, match="require evidence"):
        resolve_field_features(
            model,
            {},
            torch.ones(2, 4, 4),
            cached,
        )


def test_cached_field_features_reject_device_mismatch() -> None:
    model = type(
        "Model",
        (),
        {"control_evidence_context": False, "field": None},
    )()
    cached = {
        "field_features": torch.zeros(1, 4, 2, 2),
        "source_tokens": torch.zeros(1, 6, 4, device="meta"),
    }
    with pytest.raises(ValueError, match="footprint device"):
        resolve_field_features(
            model,
            {},
            torch.ones(1, 4, 4),
            cached,
        )
