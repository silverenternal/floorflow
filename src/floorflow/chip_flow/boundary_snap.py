"""Conservative boundary-code repair for already legal boxes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundarySnapResult:
    boxes: torch.Tensor
    moves: int
    before: int
    after: int


def _overlap_pairs(boxes: torch.Tensor) -> int:
    x0 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
    x1 = torch.minimum(
        boxes[:, None, 0] + boxes[:, None, 2],
        boxes[None, :, 0] + boxes[None, :, 2],
    )
    y0 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
    y1 = torch.minimum(
        boxes[:, None, 1] + boxes[:, None, 3],
        boxes[None, :, 1] + boxes[None, :, 3],
    )
    upper = torch.triu(
        torch.ones(
            boxes.shape[0],
            boxes.shape[0],
            device=boxes.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    return int(((x1 > x0) & (y1 > y0) & upper).sum())


def boundary_violations(
    boxes: torch.Tensor,
    codes: torch.Tensor,
) -> int:
    x0 = float(boxes[:, 0].min())
    y0 = float(boxes[:, 1].min())
    x1 = float((boxes[:, 0] + boxes[:, 2]).max())
    y1 = float((boxes[:, 1] + boxes[:, 3]).max())
    code = codes.long()
    edges = (
        (1, boxes[:, 0], x0),
        (2, boxes[:, 0] + boxes[:, 2], x1),
        (4, boxes[:, 1] + boxes[:, 3], y1),
        (8, boxes[:, 1], y0),
    )
    return sum(
        int(((code & bit).ne(0) & edge.sub(target).abs().gt(1.0e-6)).sum())
        for bit, edge, target in edges
    )


def snap_boundary_boxes(
    boxes: torch.Tensor,
    codes: torch.Tensor,
    movable: torch.Tensor,
    *,
    max_moves: int = 32,
) -> BoundarySnapResult:
    """Align movable coded blocks only when the move strictly repairs legality."""

    out = boxes.float().clone()
    code = codes.long()
    movable = movable.bool()
    before = boundary_violations(out, code)
    current = before
    moves = 0
    for _ in range(max_moves):
        x0 = float(out[:, 0].min())
        y0 = float(out[:, 1].min())
        x1 = float((out[:, 0] + out[:, 2]).max())
        y1 = float((out[:, 1] + out[:, 3]).max())
        candidates = (
            (1, out[:, 0], x0, 0),
            (2, out[:, 0] + out[:, 2], x1, 0),
            (4, out[:, 1] + out[:, 3], y1, 1),
            (8, out[:, 1], y0, 1),
        )
        deficits = []
        for bit, edge, target, axis in candidates:
            rows = ((code & bit).ne(0) & movable & edge.sub(target).abs().gt(1.0e-6))
            for index in torch.nonzero(rows, as_tuple=False).flatten().tolist():
                shift = target - float(edge[index])
                deficits.append((abs(shift), index, axis, shift))
        accepted = None
        for _, index, axis, shift in sorted(deficits):
            normal = out.clone()
            normal[index, axis] += shift
            slide_axis = 1 - axis
            size = float(out[index, 2 + slide_axis])
            lower = float(out[:, slide_axis].min())
            upper = float((out[:, slide_axis] + out[:, 2 + slide_axis]).max())
            candidates = {
                float(normal[index, slide_axis]),
                lower,
                upper - size,
            }
            for other in range(out.shape[0]):
                if other == index:
                    continue
                candidates.add(float(out[other, slide_axis] + out[other, 2 + slide_axis]))
                candidates.add(float(out[other, slide_axis] - size))
            current_slide = float(out[index, slide_axis])
            ordered = sorted(
                (
                    value
                    for value in candidates
                    if lower - 1.0e-6 <= value <= upper - size + 1.0e-6
                ),
                key=lambda value: abs(value - current_slide),
            )
            for value in ordered:
                trial = normal.clone()
                trial[index, slide_axis] = value
                if _overlap_pairs(trial) > 0:
                    continue
                trial_count = boundary_violations(trial, code)
                if trial_count < current:
                    accepted = (trial, trial_count)
                    break
            if accepted is not None:
                break
        if accepted is None:
            break
        out, current = accepted
        moves += 1
        if current == 0:
            break
    return BoundarySnapResult(
        boxes=out,
        moves=moves,
        before=before,
        after=current,
    )


__all__ = [
    "BoundarySnapResult",
    "boundary_violations",
    "snap_boundary_boxes",
]
