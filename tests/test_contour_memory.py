from collections import Counter

import numpy as np
import torch

from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.data.taxonomy import build_taxonomy


def test_contour_memory_retrieves_training_partition(tmp_path) -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
    )
    for index in range(3):
        footprint = np.zeros((32, 32), dtype=np.uint8)
        footprint[4 + index : 20 + index, 4:20] = 1
        target = np.zeros((32, 32), dtype=np.uint8)
        target[4 + index : 20 + index, 4:20] = taxonomy.class_id("room:Bedroom")
        np.savez_compressed(tmp_path / f"{index}.npz", footprint=footprint, target=target)
    memory = ContourPartitionMemory(sorted(tmp_path.glob("*.npz")))
    assert memory.precompute_partitions(taxonomy, workers=1) == 3
    query = np.zeros((1, 32, 32), dtype=np.uint8)
    query[:, 6:22, 4:20] = 1
    partitions = memory.retrieve_partitions(query, taxonomy)
    assert partitions.shape == (1, 32, 32)
    labels = memory.retrieve_labels(query)
    assert labels.shape == (1, 32, 32)
    logits = memory.retrieve_logits(query, taxonomy, "cpu")
    assert logits.shape == (1, len(taxonomy.room_ids), 32, 32)
    assert logits.dtype == torch.float32
