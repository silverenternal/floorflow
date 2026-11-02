from __future__ import annotations

import pytest
import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchical_training_state,
    hierarchy_source_state,
)


def _batch() -> dict:
    count = 5
    areas = torch.ones(2, count)
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    constraints = torch.zeros(2, count, 5)
    return {
        "areas": areas,
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "source_latent": torch.arange(2 * count * 3).float().reshape(
            2,
            count,
            3,
        ),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 1.0]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 1.0]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def test_shelf_source_matches_deployment_block_state() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        head_dim=64,
        typed_layers=1,
    )
    batch = _batch()
    context = model.prepare_context(batch)

    source, _ = hierarchical_training_state(
        batch,
        context,
        source_mode="shelf",
    )

    torch.testing.assert_close(source.block_state, batch["source_latent"])


def test_noisy_shelf_source_is_reproducible() -> None:
    model = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    batch = _batch()
    context = model.prepare_context(batch)

    first, _ = hierarchical_training_state(
        batch,
        context,
        generator=torch.Generator().manual_seed(17),
        source_mode="noisy_shelf",
        source_noise=0.25,
    )
    second, _ = hierarchical_training_state(
        batch,
        context,
        generator=torch.Generator().manual_seed(17),
        source_mode="noisy_shelf",
        source_noise=0.25,
    )

    assert not torch.equal(first.block_state, batch["source_latent"])
    torch.testing.assert_close(first.block_state, second.block_state)


def test_mixed_source_is_reproducible_and_respects_probability() -> None:
    model = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    batch = _batch()
    context = model.prepare_context(batch)
    shelf, _ = hierarchical_training_state(
        batch,
        context,
        generator=torch.Generator().manual_seed(31),
        source_mode="mixed",
        source_shelf_probability=1.0,
    )
    random, _ = hierarchical_training_state(
        batch,
        context,
        generator=torch.Generator().manual_seed(31),
        source_mode="mixed",
        source_shelf_probability=0.0,
    )
    repeated, _ = hierarchical_training_state(
        batch,
        context,
        generator=torch.Generator().manual_seed(31),
        source_mode="mixed",
        source_shelf_probability=0.0,
    )
    torch.testing.assert_close(shelf.block_state, batch["source_latent"])
    assert not torch.equal(random.block_state, batch["source_latent"])
    torch.testing.assert_close(random.block_state, repeated.block_state)


def test_random_source_preserves_legacy_behavior() -> None:
    model = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    expected = torch.randn(
        target.block_state.shape,
        generator=torch.Generator().manual_seed(23),
    )

    source, _ = hierarchical_training_state(
        batch,
        context,
        generator=torch.Generator().manual_seed(23),
        source_mode="random",
    )

    torch.testing.assert_close(source.block_state, expected)


def test_source_mode_rejects_invalid_configuration() -> None:
    model = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    batch = _batch()
    context = model.prepare_context(batch)

    with pytest.raises(ValueError, match="source_mode"):
        hierarchical_training_state(batch, context, source_mode="unknown")
    with pytest.raises(ValueError, match="non-negative"):
        hierarchical_training_state(
            batch,
            context,
            source_mode="noisy_shelf",
            source_noise=-0.1,
        )


def test_derived_hierarchy_source_is_target_free_and_consistent() -> None:
    model = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    batch = _batch()
    context = model.prepare_context(batch)

    source = hierarchy_source_state(
        batch["source_latent"],
        context,
        source_mode="derived",
        batch=batch,
    )
    changed = {key: value.clone() for key, value in batch.items()}
    changed["target_latent"].add_(100.0)
    rerun = hierarchy_source_state(
        batch["source_latent"],
        context,
        source_mode="derived",
        batch=changed,
    )

    torch.testing.assert_close(source.global_state, rerun.global_state)
    torch.testing.assert_close(source.structural_state, rerun.structural_state)
    torch.testing.assert_close(source.port_state, rerun.port_state)
    assert source.global_state.shape == (2, 4)
    assert source.structural_state.shape == (2, 9, 3)


def test_checkpoint_restores_hierarchy_source_levels() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        head_dim=64,
        typed_layers=1,
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 1,
            "head_dim": 64,
            "typed_layers": 1,
            "hierarchy_source_levels": "derived",
        },
        "model": model.state_dict(),
        "ema": model.state_dict(),
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)

    assert restored.hierarchy_source_levels == "derived"
