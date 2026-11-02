from __future__ import annotations

import json

import torch

from floorflow.data.raster2seq_proposal_cache import (
    MANIFEST_VERSION,
    sha256_file,
    write_teacher_case,
)
from floorflow.training.proposal_distillation import (
    ProposalTeacherIndex,
    TeacherTargetBatch,
    proposal_distillation_loss,
)


def test_proposal_distillation_matches_masks_and_backpropagates() -> None:
    masks = torch.zeros(1, 3, 16, 16, dtype=torch.bool)
    masks[0, 0, 1:7, 1:7] = True
    masks[0, 1, 8:15, 8:15] = True
    valid = torch.tensor([[True, True, False]])
    confidence = torch.tensor([[0.9, 0.8, 0.0]])
    class_ids = torch.tensor([[0, 1, -1]])
    teacher = TeacherTargetBatch(
        masks=masks,
        valid=valid,
        confidence=confidence,
        sample_keys=("case__Floor-1",),
        class_ids=class_ids,
    )
    logits = torch.randn(1, 4, 16, 16, requires_grad=True)
    class_logits = torch.randn(1, 4, 3, requires_grad=True)
    loss, components = proposal_distillation_loss(
        logits,
        teacher,
        class_logits=class_logits,
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "proposal_distill_loss",
        "proposal_distill_matched",
        "proposal_distill_background",
        "proposal_distill_matched_iou",
        "proposal_distill_class",
        "proposal_distill_class_accuracy",
    }
    loss.backward()
    assert logits.grad is not None
    assert class_logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_teacher_index_resolves_sample_key_without_class_dependency(tmp_path) -> None:
    cache = tmp_path / "case.npz"
    write_teacher_case(
        cache,
        sample_key="case__Floor-1",
        split="train",
        split_index=0,
        image_size=16,
        polygons=[[[1, 1], [1, 8], [8, 8], [8, 1]]],
        class_ids=[2],
        class_names=["room:Bedroom"],
        existence_scores=[0.75],
        sequence_length=5,
        source_image_sha256="c" * 64,
    )
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "target_used_at_inference": False,
        "rows": [
            {
                "sample_key": "case__Floor-1",
                "cache_file": "case.npz",
                "cache_sha256": sha256_file(cache),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = ProposalTeacherIndex(manifest_path)
    targets = index.load(
        ["case__Floor-1"],
        image_size=16,
        max_proposals=4,
    )
    assert targets.masks.shape == (1, 4, 16, 16)
    assert targets.valid.sum() == 1
    assert torch.isclose(targets.confidence[0, 0], torch.tensor(0.75))
    resized = index.load(
        ["case__Floor-1"],
        image_size=32,
        max_proposals=4,
    )
    assert resized.masks.shape == (1, 4, 32, 32)
    assert resized.masks[0, 0].any()
    compact = index.load(
        ["case__Floor-1"],
        image_size=16,
        max_proposals=4,
        compact_proposals=True,
    )
    assert compact.masks.shape == (1, 1, 16, 16)
    assert compact.valid.tolist() == [[True]]
    assert len(index._case_cache) == 1
