from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.training.proposal_distillation import TeacherTargetBatch
from floorflow.training.room_capability_training import (
    teacher_room_targets,
)


def test_teacher_room_targets_assign_majority_semantic_room() -> None:
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
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 2:8, 2:8] = taxonomy.room_ids[1]
    masks = torch.zeros(1, 2, 16, 16, dtype=torch.bool)
    masks[0, 0, 2:8, 2:8] = True
    masks[0, 1, 10:14, 10:14] = True
    valid = torch.tensor([[True, True]])
    teacher = TeacherTargetBatch(
        masks=masks,
        valid=valid,
        confidence=torch.ones(1, 2),
        sample_keys=("case",),
    )
    classes, mask_targets, target_valid = teacher_room_targets(
        teacher,
        target,
        taxonomy,
        max_rooms=2,
    )
    assert classes[0, 0].item() == 1
    assert not target_valid[0, 1]
    assert mask_targets.shape == (1, 2, 16, 16)
