"""Single typed state shared by room, graph, geometry, and token branches."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from scipy.optimize import linear_sum_assignment

from floorflow.objectives.gpu_assignment import soft_room_assignment


@dataclass
class TypedFloorplanState:
    room_instance_ids: torch.Tensor | None = None
    room_features: torch.Tensor | None = None
    room_exists: torch.Tensor | None = None
    room_class: torch.Tensor | None = None
    room_polygons: torch.Tensor | None = None
    room_adjacency_logits: torch.Tensor | None = None
    wall_endpoints: torch.Tensor | None = None
    wall_exists: torch.Tensor | None = None
    wall_exterior_logits: torch.Tensor | None = None
    opening_centers: torch.Tensor | None = None
    opening_widths: torch.Tensor | None = None
    opening_exists: torch.Tensor | None = None
    opening_host_logits: torch.Tensor | None = None
    junction_coordinates: torch.Tensor | None = None
    junction_degree_logits: torch.Tensor | None = None
    stage_tokens: torch.Tensor | None = None
    token_context_tokens: torch.Tensor | None = None
    token_context_valid: torch.Tensor | None = None

    @classmethod
    def from_predictions(cls, predictions: dict) -> TypedFloorplanState:
        room_exists = predictions.get("room_exists")
        room_instance_ids = predictions.get("room_instance_ids")
        if room_instance_ids is None and room_exists is not None:
            room_instance_ids = torch.arange(
                room_exists.shape[1],
                device=room_exists.device,
            )[None].expand(room_exists.shape[0], -1)
        return cls(
            room_instance_ids=room_instance_ids,
            room_features=predictions.get(
                "room_identity_tokens",
                predictions.get("room_features"),
            ),
            room_exists=predictions.get("room_exists"),
            room_class=predictions.get("room_class"),
            room_polygons=predictions.get("room_polygons"),
            room_adjacency_logits=predictions.get("room_adjacency_logits"),
            wall_endpoints=predictions.get("wall_endpoints"),
            wall_exists=predictions.get("wall_exists"),
            wall_exterior_logits=predictions.get("wall_exterior_logits"),
            opening_centers=predictions.get("opening_centers"),
            opening_widths=predictions.get("opening_widths"),
            opening_exists=predictions.get("opening_exists"),
            opening_host_logits=predictions.get("opening_host_logits"),
            junction_coordinates=predictions.get("token_junction_coordinates"),
            junction_degree_logits=predictions.get("token_junction_degree_logits"),
            stage_tokens=predictions.get("action_stage_tokens"),
            token_context_tokens=predictions.get("token_context_tokens"),
            token_context_valid=predictions.get("token_context_valid"),
        )

    def to_predictions(self) -> dict[str, torch.Tensor]:
        mapping = {
            "room_instance_ids": self.room_instance_ids,
            "room_features": self.room_features,
            "room_exists": self.room_exists,
            "room_class": self.room_class,
            "room_polygons": self.room_polygons,
            "room_adjacency_logits": self.room_adjacency_logits,
            "wall_endpoints": self.wall_endpoints,
            "wall_exists": self.wall_exists,
            "wall_exterior_logits": self.wall_exterior_logits,
            "opening_centers": self.opening_centers,
            "opening_widths": self.opening_widths,
            "opening_exists": self.opening_exists,
            "opening_host_logits": self.opening_host_logits,
            "token_junction_coordinates": self.junction_coordinates,
            "token_junction_degree_logits": self.junction_degree_logits,
            "action_stage_tokens": self.stage_tokens,
            "token_context_tokens": self.token_context_tokens,
            "token_context_valid": self.token_context_valid,
        }
        return {
            name: value
            for name, value in mapping.items()
            if value is not None
        }

    def clone(self) -> TypedFloorplanState:
        values = {
            field.name: (
                getattr(self, field.name).clone()
                if torch.is_tensor(getattr(self, field.name))
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
        return TypedFloorplanState(**values)

    def room_boxes(self) -> torch.Tensor | None:
        if self.room_polygons is None:
            return None
        return torch.cat(
            [
                self.room_polygons.amin(dim=2),
                self.room_polygons.amax(dim=2),
            ],
            dim=-1,
        )

    def align_rooms_to(
        self,
        reference: TypedFloorplanState,
        *,
        differentiable: bool | None = None,
    ) -> TypedFloorplanState:
        boxes = self.room_boxes()
        reference_boxes = reference.room_boxes()
        if boxes is None or reference_boxes is None:
            return self
        use_soft = (
            torch.is_grad_enabled()
            if differentiable is None
            else bool(differentiable)
        )
        if use_soft and (
            differentiable is True
            or boxes.requires_grad
            or reference_boxes.requires_grad
        ):
            return self._soft_align_rooms_to(reference, boxes, reference_boxes)
        valid = (
            reference.room_exists.sigmoid() >= 0.5
            if reference.room_exists is not None
            else torch.ones(reference_boxes.shape[:2], dtype=torch.bool)
        )
        permutation = []
        for batch in range(boxes.shape[0]):
            count = int(valid[batch].sum())
            if count == 0:
                permutation.append(torch.arange(boxes.shape[1]))
                continue
            cost = torch.cdist(
                boxes[batch],
                reference_boxes[batch, :count],
            )
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            order = torch.arange(boxes.shape[1])
            order[torch.as_tensor(columns)] = torch.as_tensor(rows)
            permutation.append(order)
        index = torch.stack(permutation).to(boxes.device)
        return self._reorder_rooms(index)

    def _soft_align_rooms_to(
        self,
        reference: TypedFloorplanState,
        boxes: torch.Tensor,
        reference_boxes: torch.Tensor,
    ) -> TypedFloorplanState:
        predicted_valid = (
            self.room_exists.sigmoid() >= 0.5
            if self.room_exists is not None
            else torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        )
        reference_valid = (
            reference.room_exists.sigmoid() >= 0.5
            if reference.room_exists is not None
            else torch.ones(
                reference_boxes.shape[:2],
                dtype=torch.bool,
                device=boxes.device,
            )
        )
        assignment = soft_room_assignment(
            boxes,
            reference_boxes,
            predicted_valid=predicted_valid,
            reference_valid=reference_valid,
        )
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if field.name in {"room_features", "room_class"}:
                value = torch.einsum("bst,bsd->btd", assignment, value)
            elif field.name == "room_exists":
                value = torch.einsum("bst,bs->bt", assignment, value)
            elif field.name == "room_polygons":
                value = torch.einsum("bst,bspd->btpd", assignment, value)
            elif field.name == "room_adjacency_logits":
                value = (
                    assignment.transpose(1, 2)
                    @ value
                    @ assignment
                )
            elif field.name == "room_instance_ids":
                value = reference.room_instance_ids
            values[field.name] = value
        return replace(self, **values)

    def _reorder_rooms(self, index: torch.Tensor) -> TypedFloorplanState:
        gather_index = index[..., None]
        polygon_index = index[..., None, None].expand(
            -1,
            -1,
            self.room_polygons.shape[2],
            self.room_polygons.shape[3],
        )
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                values[field.name] = None
                continue
            if field.name in {"room_features", "room_class"}:
                value = value.gather(1, gather_index.expand_as(value))
            elif field.name in {"room_exists", "room_instance_ids"}:
                value = value.gather(1, index)
            elif field.name == "room_polygons":
                value = value.gather(1, polygon_index)
            elif field.name == "room_adjacency_logits":
                value = value.gather(1, gather_index.expand_as(value))
                value = value.gather(
                    2,
                    index[..., None].expand_as(value),
                )
            values[field.name] = value
        return replace(self, **values)


__all__ = ["TypedFloorplanState"]
