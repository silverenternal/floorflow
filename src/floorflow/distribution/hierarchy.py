"""CAD family hierarchy shared by pixel and element distribution heads."""

from __future__ import annotations

from dataclasses import dataclass

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class ChannelHierarchy:
    num_classes: int
    wall_ids: tuple[int, ...]
    opening_ids: tuple[int, ...]
    room_ids: tuple[int, ...]
    furniture_ids: tuple[int, ...]
    support_ids: tuple[int, ...]

    @classmethod
    def from_taxonomy(cls, taxonomy: FloorplanTaxonomy) -> "ChannelHierarchy":
        wall = tuple(
            index
            for index, name in enumerate(taxonomy.class_names)
            if name == "wall"
        )
        support = tuple(
            index
            for index, name in enumerate(taxonomy.class_names)
            if name == "structure"
        )
        return cls(
            num_classes=taxonomy.num_classes,
            wall_ids=wall,
            opening_ids=tuple(taxonomy.opening_ids),
            room_ids=tuple(taxonomy.room_ids),
            furniture_ids=tuple(taxonomy.furniture_ids),
            support_ids=support,
        )

    @staticmethod
    def class_indices(class_ids: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(class_id) for class_id in class_ids)

    @staticmethod
    def foreground_indices(class_ids: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(class_id) - 1 for class_id in class_ids)

    @property
    def groups(self) -> dict[str, tuple[int, ...]]:
        return {
            "wall": self.class_indices(self.wall_ids),
            "opening": self.class_indices(self.opening_ids),
            "room": self.class_indices(self.room_ids),
            "furniture": self.class_indices(self.furniture_ids),
            "support": self.class_indices(self.support_ids),
        }


__all__ = ["ChannelHierarchy"]
