"""Dense semantic fusion for room-query mask decoders."""

from __future__ import annotations

import torch
from scipy.ndimage import label

from floorflow.data.taxonomy import FloorplanTaxonomy


def query_gate_probability(
    existence_logits: torch.Tensor,
    selection: torch.Tensor | None,
) -> torch.Tensor:
    """Return one authoritative query gate for every downstream consumer."""

    if selection is None:
        return existence_logits.sigmoid()
    if selection.shape != existence_logits.shape:
        raise ValueError("query count selection must match [B,Q]")
    return selection.float().clamp(0.0, 1.0)


def apply_program_class_prior(
    class_logits: torch.Tensor,
    program_logits: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Calibrate foreground query classes with a contour-only program prior."""

    foreground = class_logits.shape[-1] - 1
    if program_logits.shape != (
        class_logits.shape[0],
        foreground,
    ):
        raise ValueError("program logits must be [B,num_room_classes]")
    centered = program_logits - program_logits.mean(dim=-1, keepdim=True)
    adjusted = class_logits.clone()
    adjusted[..., :foreground] = (
        adjusted[..., :foreground]
        + float(scale) * centered[:, None, :]
    )
    return adjusted


def apply_class_count_prior(
    class_logits: torch.Tensor,
    class_count_logits: torch.Tensor,
    *,
    scale: float | torch.Tensor = 0.1,
) -> torch.Tensor:
    """Calibrate query classes with predicted global per-class counts."""

    room_classes = class_logits.shape[-1] - 1
    if class_count_logits.shape[:2] != (
        class_logits.shape[0],
        room_classes,
    ):
        raise ValueError("class-count logits must be [B,num_room_classes,Q+1]")
    probability = class_count_logits.float().softmax(dim=-1)
    counts = torch.arange(
        probability.shape[-1],
        device=probability.device,
        dtype=probability.dtype,
    )
    expected = (probability * counts).sum(dim=-1)
    prior = (expected + 1.0).log()
    prior = prior - prior.mean(dim=-1, keepdim=True)
    adjusted = class_logits.clone()
    adjusted[..., :room_classes] = (
        adjusted[..., :room_classes]
        + scale * prior[:, None]
    )
    return adjusted


def fused_room_mask_logits(
    predictions: dict[str, torch.Tensor],
    num_room_classes: int,
    fusion_mode: str = "sigmoid",
) -> torch.Tensor:
    """Fuse query masks and classes into one dense room-class map."""

    if fusion_mode == "legacy":
        mask_weights = predictions["mask_logits"].softmax(dim=1).clamp_min(1.0e-8)
        room_logits = predictions["class_logits"][..., :num_room_classes]
        class_log_probability = room_logits.log_softmax(dim=-1)[:, :, :, None, None]
        fused = torch.logsumexp(
            mask_weights.log()[:, :, None] + class_log_probability,
            dim=1,
        )
        scale = predictions.get("fusion_scale")
        if scale is not None:
            fused = fused * scale
    elif fusion_mode in {"sigmoid", "sigmoid_existence"}:
        mask_probability = predictions["mask_logits"].sigmoid()
        if fusion_mode == "sigmoid_existence":
            existence = predictions.get("query_existence_logits")
            if existence is None:
                raise ValueError(
                    "sigmoid_existence fusion requires query_existence_logits"
                )
            gate = query_gate_probability(
                existence,
                predictions.get("query_count_selection_mask"),
            )
            mask_probability = mask_probability * gate[..., None, None]
        else:
            selection = predictions.get("query_count_selection_mask")
            if selection is not None:
                if selection.shape != mask_probability.shape[:2]:
                    raise ValueError(
                        "query_count_selection_mask must match [B,Q]"
                    )
                mask_probability = mask_probability * selection[
                    :, :, None, None
                ].to(mask_probability.dtype)
        class_probability = predictions["class_logits"][
            ..., :num_room_classes
        ].softmax(dim=-1)
        class_mass = torch.einsum(
            "bqhw,bqk->bkhw",
            mask_probability,
            class_probability,
        )
        fused_probability = class_mass / class_mass.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        fused = fused_probability.clamp_min(1.0e-8).log()
    elif fusion_mode in {"query_mixture", "hard_query"}:
        mask_probability = predictions["mask_logits"].sigmoid()
        query_weight = mask_probability / mask_probability.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        room_logits = predictions["class_logits"][..., :num_room_classes]
        if fusion_mode == "query_mixture":
            class_probability = room_logits.softmax(dim=-1)
        else:
            class_probability = torch.nn.functional.one_hot(
                room_logits.argmax(dim=-1),
                num_classes=num_room_classes,
            ).to(query_weight.dtype)
        fused_probability = torch.einsum(
            "bqhw,bqk->bkhw",
            query_weight,
            class_probability,
        )
        fused_probability = fused_probability / fused_probability.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        fused = fused_probability.clamp_min(1.0e-8).log()
    elif fusion_mode == "winner_query":
        soft = fused_room_mask_logits(
            predictions,
            num_room_classes,
            fusion_mode="sigmoid",
        )
        mask_probability = predictions["mask_logits"].sigmoid()
        room_logits = predictions["class_logits"][..., :num_room_classes]
        class_score, class_index = room_logits.max(dim=-1)
        owner_score = torch.nn.functional.logsigmoid(
            predictions["mask_logits"]
        ) + 0.25 * class_score[..., None, None]
        owner = owner_score.argmax(dim=1)
        flat_owner = owner.flatten(1)
        owner_mask = torch.gather(
            mask_probability.flatten(2),
            1,
            flat_owner[:, None],
        )[:, 0].reshape(owner.shape)
        owner_class = torch.gather(class_index, 1, flat_owner)
        one_hot = torch.nn.functional.one_hot(
            owner_class.reshape(owner.shape),
            num_classes=num_room_classes,
        ).permute(0, 3, 1, 2).to(soft.dtype)
        hard = 12.0 * one_hot - 12.0 * (1.0 - one_hot)
        fused = torch.where(owner_mask[:, None] > 0.5, hard, soft)
    elif fusion_mode == "slot_owner":
        owner_probability = predictions.get("slot_owner_probability")
        if owner_probability is None:
            raise ValueError(
                "slot_owner fusion requires slot_owner_probability"
            )
        class_probability = predictions["class_logits"][
            ..., :num_room_classes
        ].softmax(dim=-1)
        class_mass = torch.einsum(
            "bqhw,bqk->bkhw",
            owner_probability,
            class_probability,
        )
        fused_probability = class_mass / class_mass.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        fused = fused_probability.clamp_min(1.0e-8).log()
    else:
        raise ValueError(f"unsupported room-mask fusion mode: {fusion_mode}")
    prior = predictions.get("prior_logits")
    if prior is not None and prior.shape == fused.shape:
        fused = fused + prior
    return fused


@torch.no_grad()
def render_room_masks(
    predictions: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    footprint: torch.Tensor | None = None,
) -> torch.Tensor:
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=predictions["mask_logits"].device,
    )
    fused = predictions.get("fused_logits")
    if fused is None or fused.shape[1] != len(room_ids):
        fused = fused_room_mask_logits(predictions, len(room_ids))
    labels = room_ids[fused.argmax(dim=1)]
    if footprint is not None:
        labels = labels * footprint.long()
    return labels


@torch.no_grad()
def mask_region_class_logits(
    predictions: dict[str, torch.Tensor],
    external_class_logits: torch.Tensor,
    footprint: torch.Tensor,
    *,
    threshold: float = 0.5,
    strength: float = 8.0,
) -> torch.Tensor:
    """Use query masks as regions and external logits as their class evidence."""

    mask_probability = predictions["mask_logits"].sigmoid()
    owner = mask_probability.argmax(dim=1)
    support = (
        mask_probability.max(dim=1).values >= float(threshold)
    ) & footprint.bool()
    output = external_class_logits.clone()
    owner_np = owner.detach().cpu().numpy()
    support_np = support.detach().cpu().numpy()
    for batch_index in range(support.shape[0]):
        for query_index in range(owner.shape[1]):
            query_support = support_np[batch_index] & (
                owner_np[batch_index] == query_index
            )
            regions, count = label(query_support)
            for region_index in range(1, int(count) + 1):
                region = torch.from_numpy(
                    regions == region_index
                ).to(external_class_logits.device)
                scores = external_class_logits[batch_index][
                    :, region
                ].mean(dim=1)
                class_index = int(scores.argmax())
                output[batch_index, :, region] = -float(strength)
                output[batch_index, class_index, region] = float(strength)
    return output


__all__ = [
    "apply_class_count_prior",
    "apply_program_class_prior",
    "fused_room_mask_logits",
    "mask_region_class_logits",
    "query_gate_probability",
    "render_room_masks",
]
