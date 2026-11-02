from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.opening_readout import OpeningDetection
from floorflow.geometry.wall_graph_readout import ReadoutFurniture, ReadoutWall
from floorflow.inference.cad_layer_vectorizer import (
    project_detections_to_walls,
    vectorize_layers,
)
from floorflow.inference.furniture_vector_readout import (
    clip_furniture_to_rooms,
)
from floorflow.model.cad_layers import CadLayeredOutput


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_vectorize_layers_returns_wall_room_and_opening_objects() -> None:
    taxonomy = _taxonomy()
    layers = CadLayeredOutput(
        room_logits=torch.zeros(1, len(taxonomy.room_ids), 32, 32),
        wall_logits=torch.full((1, 1, 32, 32), -8.0),
        opening_logits=torch.full(
            (1, len(taxonomy.opening_ids), 32, 32),
            -8.0,
        ),
        furniture_logits=torch.full(
            (1, len(taxonomy.furniture_ids), 32, 32),
            -8.0,
        ),
    )
    layers.wall_logits[:, :, 3:5, 3:29] = 8.0
    layers.wall_logits[:, :, 27:29, 3:29] = 8.0
    layers.wall_logits[:, :, 3:29, 3:5] = 8.0
    layers.wall_logits[:, :, 3:29, 27:29] = 8.0
    layers.opening_logits[:, 0, 3:5, 10:17] = 8.0
    layers.furniture_logits[:, 0, 10:14, 9:13] = 8.0
    footprint = torch.ones(1, 32, 32)
    result = vectorize_layers(
        layers,
        footprint,
        taxonomy,
        wall_threshold=0.5,
        opening_threshold=0.3,
    )
    assert result.walls
    assert result.rooms
    assert result.furniture
    assert result.to_plan().walls
    assert result.to_plan().to_json()["furniture"]
    assert result.labels.shape == (32, 32)


def test_vectorize_layers_consumes_furniture_queries() -> None:
    taxonomy = _taxonomy()
    layers = CadLayeredOutput(
        room_logits=torch.zeros(1, len(taxonomy.room_ids), 32, 32),
        wall_logits=torch.full((1, 1, 32, 32), -8.0),
        opening_logits=torch.full(
            (1, len(taxonomy.opening_ids), 32, 32),
            -8.0,
        ),
        furniture_logits=torch.full(
            (1, len(taxonomy.furniture_ids), 32, 32),
            -8.0,
        ),
    )
    query_classes = torch.full(
        (1, 2, len(taxonomy.furniture_ids) + 1),
        -8.0,
    )
    query_classes[0, 0, 0] = 8.0
    query_classes[0, 1, -1] = 8.0
    query_masks = torch.full((1, 2, 16, 16), -8.0)
    query_masks[0, 0, 5:10, 6:11] = 8.0
    result = vectorize_layers(
        layers,
        torch.ones(1, 32, 32),
        taxonomy,
        furniture_queries={
            "class_logits": query_classes,
            "mask_logits": query_masks,
            "query_budget": torch.tensor([2]),
            "room_assignment_logits": torch.tensor(
                [
                    [
                        [8.0, -8.0],
                        [-8.0, 8.0],
                    ]
                ]
            ),
        },
    )
    assert len(result.furniture) == 1
    assert result.furniture[0].label == taxonomy.class_names[
        taxonomy.furniture_ids[0]
    ]
    assert result.to_plan().to_json()["furniture"][0]["room_id"] == "room_1"


def test_projected_opening_width_is_positive_and_host_bounded() -> None:
    taxonomy = _taxonomy()
    wall = ReadoutWall(
        start=(2.0, 2.0),
        end=(10.0, 2.0),
        thickness=3.0,
    )
    detection = OpeningDetection(
        center=(6.0, 2.0),
        width=100.0,
        family="door",
        class_id=taxonomy.opening_ids[0],
        score=1.0,
        wall_index=0,
        wall_distance=0.0,
    )
    rows = project_detections_to_walls([detection], [wall], frame=32.0)
    assert rows[0].width == 8.0


def test_furniture_class_assignment_resolves_specific_room_instance() -> None:
    taxonomy = _taxonomy()
    furniture = (
        ReadoutFurniture(
            label=taxonomy.class_names[taxonomy.furniture_ids[0]],
            polygon=((20, 20), (24, 20), (24, 24), (20, 24)),
            room_index=0,
        ),
    )
    rooms = [
        {
            "label": taxonomy.class_names[taxonomy.room_ids[0]],
            "polygon": [[0, 0], [16, 0], [16, 32], [0, 32]],
        },
        {
            "label": taxonomy.class_names[taxonomy.room_ids[0]],
            "polygon": [[16, 0], [32, 0], [32, 32], [16, 32]],
        },
    ]
    clipped = clip_furniture_to_rooms(
        furniture,
        rooms,
        taxonomy=taxonomy,
    )
    assert len(clipped) == 1
    assert clipped[0].room_index == 1


def test_vector_furniture_is_clipped_to_assigned_room() -> None:
    furniture = (
        ReadoutFurniture(
            label="furniture:Bed",
            polygon=((8, 2), (20, 2), (20, 8), (8, 8)),
            room_index=0,
        ),
        ReadoutFurniture(
            label="furniture:Bed",
            polygon=((1, 1), (2, 1), (2, 2)),
            room_index=3,
        ),
    )
    clipped = clip_furniture_to_rooms(
        furniture,
        [
            {
                "label": "room:Bedroom",
                "polygon": [[0, 0], [16, 0], [16, 16], [0, 16]],
            }
        ],
    )
    assert len(clipped) == 1
    points = clipped[0].polygon
    assert max(point[0] for point in points) == 16.0
