from pathlib import Path

import numpy as np
import torch

from floorflow.data.retrieval_control_dataset import RetrievalControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy


class _Sample:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path


class _Base:
    def __init__(self, paths: list[Path], taxonomy) -> None:
        self.samples = [_Sample(path) for path in paths]
        self.taxonomy = taxonomy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        with np.load(self.samples[index].cache_path) as payload:
            target = torch.from_numpy(payload["target"].copy()).long()
            footprint = torch.from_numpy(payload["footprint"].copy()).float()
        return target, footprint, torch.tensor(1), str(index)


def _write_case(root: Path, index: int, taxonomy) -> None:
    (root / "train").mkdir(parents=True, exist_ok=True)
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8 : 24, 8 : 12] = taxonomy.room_ids[0]
    target[8 : 24, 12 : 24] = taxonomy.room_ids[1]
    footprint = (target > 0).astype(np.uint8)
    np.savez_compressed(
        root / "train" / f"case{index}.npz",
        target=target,
        footprint=footprint,
    )


def test_train_retrieval_excludes_the_query_sample(tmp_path):
    taxonomy = FloorplanTaxonomy(
        class_names=("background", "room:A", "room:B", "wall"),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )
    _write_case(tmp_path, 0, taxonomy)
    _write_case(tmp_path, 1, taxonomy)
    paths = sorted((tmp_path / "train").glob("*.npz"))
    dataset = RetrievalControlDataset(
        _Base(paths, taxonomy),
        tmp_path,
        "train",
        control_size=16,
    )
    target, _footprint, _floor, _name, control = dataset[0]
    assert target.shape == (32, 32)
    assert control["room_logits"].shape == (len(taxonomy.room_ids), 16, 16)
    assert control["program"].shape == (len(taxonomy.room_ids),)
