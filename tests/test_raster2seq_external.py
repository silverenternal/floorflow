from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.raster2seq_external import (
    decode_raster2seq_proposals,
    decode_raster2seq_sequence,
    render_raster2seq_input,
    render_raster2seq_predictions,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Swing Beside",
            "window:Regular",
            "room:Kitchen",
            "room:LivingRoom",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def test_decode_raster2seq_sequence_filters_rooms_and_maps_classes() -> None:
    taxonomy = _taxonomy()
    sequence = [
        np.array([0.1, 0.1], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0.9, 0.9], dtype=np.float32),
        1,
        np.array([0.1, 0.5], dtype=np.float32),
        np.array([0.4, 0.5], dtype=np.float32),
        2,
    ]
    logits = np.zeros((7, 11), dtype=np.float32)
    logits[:3, 2] = 1.0
    logits[4:6, 10] = 1.0

    polygons, labels = decode_raster2seq_sequence(sequence, logits, taxonomy)

    assert [len(polygon) for polygon in polygons] == [3, 2]
    assert labels == [taxonomy.class_id("room:LivingRoom"), taxonomy.class_id("door:Swing Beside")]


def test_decode_raster2seq_proposals_retains_class_confidence() -> None:
    taxonomy = _taxonomy()
    sequence = [
        np.array([0.1, 0.1], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0.9, 0.9], dtype=np.float32),
        1,
    ]
    logits = np.zeros((4, 11), dtype=np.float32)
    logits[:, 2] = 2.0
    proposals = decode_raster2seq_proposals(sequence, logits, taxonomy)
    assert len(proposals) == 1
    assert proposals[0].class_id == taxonomy.class_id("room:LivingRoom")
    assert 0.0 < proposals[0].class_probability <= 1.0
    assert proposals[0].area > 0.0


def test_render_input_and_predictions_use_exclusive_labels() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[4:28, 4:28] = taxonomy.class_id("room:Kitchen")
    labels[15:17, 4:28] = taxonomy.class_id("wall")
    image = render_raster2seq_input(labels, taxonomy)
    assert image.shape == (32, 32, 3)
    assert image[0, 0].tolist() == [255, 255, 255]
    assert image[16, 10].tolist() == [0, 0, 0]

    polygon = np.array([[4, 4], [27, 4], [27, 27], [4, 27]], dtype=np.int32)
    output = render_raster2seq_predictions(
        [polygon],
        [taxonomy.class_id("room:LivingRoom")],
        labels.shape,
        taxonomy,
    )
    assert np.all(output[10:20, 10:20] == taxonomy.class_id("room:LivingRoom"))
