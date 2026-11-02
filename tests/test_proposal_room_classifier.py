from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.proposal_room_classifier import (
    ProposalRoomClassifier,
    compose_proposal_labels,
    proposal_classification_loss,
)
from floorflow.training.proposal_distillation import TeacherTargetBatch


def test_proposal_classifier_is_live_and_composes_labels() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 5, "Bath": 5}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    masks = torch.zeros(1, 2, 16, 16, dtype=torch.bool)
    masks[0, 0, :, :8] = True
    masks[0, 1, :, 8:] = True
    teacher = TeacherTargetBatch(
        masks=masks,
        valid=torch.ones(1, 2, dtype=torch.bool),
        confidence=torch.ones(1, 2),
        sample_keys=("case",),
        class_ids=torch.full(
            (1, 2),
            taxonomy.room_ids[0],
            dtype=torch.long,
        ),
    )
    classifier = ProposalRoomClassifier(8, len(taxonomy.room_ids), hidden_dim=16)
    features = torch.randn(1, 8, 16, 16, requires_grad=True)
    room_logits = torch.randn(1, len(taxonomy.room_ids), 16, 16)
    output = classifier(features, room_logits, teacher, taxonomy)
    targets = torch.tensor([[0, 1]])
    loss, components = proposal_classification_loss(
        output,
        teacher,
        targets,
        teacher.valid,
    )
    loss.backward()
    assert features.grad is not None
    assert components["proposal_room_accuracy"].ndim == 0
    labels = compose_proposal_labels(
        output,
        teacher,
        taxonomy,
        torch.ones(1, 16, 16),
    )
    assert torch.isin(
        labels,
        torch.tensor(taxonomy.room_ids),
    ).any()
