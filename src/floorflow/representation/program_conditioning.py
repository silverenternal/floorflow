"""Optional room-program control for contour-conditioned generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as component_label
from scipy.optimize import linear_sum_assignment


def apply_program_constraint(
    room_logits: torch.Tensor,
    room_program: torch.Tensor,
    *,
    absent_penalty: float = 8.0,
) -> torch.Tensor:
    """Suppress room classes declared absent by an explicit program vector."""

    if room_logits.ndim != 4:
        raise ValueError("room_logits must be [B,C,H,W]")
    program = room_program.to(device=room_logits.device, dtype=room_logits.dtype)
    if program.ndim == 1:
        program = program[None]
    if program.shape != (room_logits.shape[0], room_logits.shape[1]):
        raise ValueError("room_program must match [B,C] room logits")
    present = program > 0.5
    nonempty = present.any(dim=1)
    safe_present = present | (~nonempty)[:, None]
    penalty = (~safe_present).to(room_logits.dtype) * float(absent_penalty)
    return room_logits - penalty[:, :, None, None]


class ProgramConditioner(nn.Module):
    """Learned class bias and token from an explicit room program."""

    def __init__(self, num_room_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.encoder = nn.Sequential(
            nn.Linear(self.num_room_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.class_bias = nn.Linear(hidden_dim, self.num_room_classes)
        nn.init.zeros_(self.class_bias.weight)
        nn.init.zeros_(self.class_bias.bias)

    def forward(self, room_program: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(room_program.float())
        return {
            "class_bias": self.class_bias(hidden),
            "tokens": F.normalize(hidden, dim=-1)[:, None],
        }


def assign_instances_to_program(
    labels: torch.Tensor,
    room_logits: torch.Tensor,
    room_program_counts: torch.Tensor,
    room_class_ids: tuple[int, ...],
    *,
    min_area: int = 8,
) -> torch.Tensor:
    """Assign one program-consistent class to each room component."""

    output = labels.clone()
    ids = np.asarray(room_class_ids, dtype=np.int64)
    label_maps = labels.detach().cpu().numpy()
    logits = room_logits.detach().cpu().numpy()
    counts = room_program_counts.detach().cpu().numpy()
    for batch_index in range(label_maps.shape[0]):
        room_support = np.isin(label_maps[batch_index], ids)
        components, count = component_label(room_support)
        component_masks = [
            components == index
            for index in range(1, int(count) + 1)
            if int((components == index).sum()) >= int(min_area)
        ]
        if not component_masks:
            continue
        scores = np.stack(
            [logits[batch_index][:, mask].mean(axis=1) for mask in component_masks]
        )
        slots = [
            class_index
            for class_index, class_count in enumerate(counts[batch_index])
            for _ in range(int(class_count))
        ]
        if not slots:
            continue
        if len(slots) < len(component_masks):
            slots.extend(scores.argmax(axis=1)[len(slots):].tolist())
        elif len(slots) > len(component_masks):
            keep = np.argsort(-scores.max(axis=0))[: len(component_masks)]
            slots = [slots[index] for index in sorted(int(item) for item in keep)]
        cost = -scores[:, slots]
        rows, columns = linear_sum_assignment(cost)
        updated = label_maps[batch_index].copy()
        for row, column in zip(rows, columns):
            updated[component_masks[row]] = ids[slots[column]]
        output[batch_index] = torch.from_numpy(updated).to(
            device=labels.device,
            dtype=labels.dtype,
        )
    return output


__all__ = [
    "ProgramConditioner",
    "apply_program_constraint",
    "assign_instances_to_program",
]
