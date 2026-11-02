"""Validation helpers for contour-conditioned distribution models."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.evaluation.room_metric_suite import room_metric_suite


@torch.inference_mode()
def validate_distribution_room_iou(
    model,
    loader,
    taxonomy,
    device,
    *,
    style_assignments: dict[str, int] | None = None,
    default_style_id: int | None = None,
):
    """Evaluate visible room and matched-instance metrics."""

    model.eval()
    rows = []
    for target, footprint, floor_ids, names in loader:
        target = target.to(device, non_blocking=True)
        footprint = footprint[:, None].to(device, non_blocking=True)
        output = model.contour_initial_state(
            footprint,
            floor_ids=floor_ids.to(device, non_blocking=True),
            style_ids=(
                torch.tensor(
                    [style_assignments[name] for name in names],
                    device=device,
                    dtype=torch.long,
                )
                if style_assignments is not None
                else (
                    torch.full(
                        (target.shape[0],),
                        default_style_id,
                        device=device,
                        dtype=torch.long,
                    )
                    if default_style_id is not None
                    else None
                )
            ),
        )
        prediction = output["pixel"]["logits"].argmax(dim=1) * footprint[:, 0].long()
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        for index in range(target.shape[0]):
            predicted_support = torch.isin(
                prediction[index],
                room_ids,
            ).cpu().numpy()
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index, 0].cpu().numpy(),
                taxonomy,
                predicted_room_support=predicted_support,
                predicted_instance_support=predicted_support,
                target_room_support=torch.isin(
                    target[index],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


__all__ = ["validate_distribution_room_iou"]
