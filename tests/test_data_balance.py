from pathlib import Path

import numpy as np

from floorflow.data.data_balance import cached_class_presence_weights


class _Dataset:
    def __init__(self, root: Path) -> None:
        self.cache_root = root
        self.augment = True
        self.items = [
            np.asarray([[1, 1], [0, 0]]),
            np.asarray([[2, 2], [0, 0]]),
            np.asarray([[1, 1], [2, 2]]),
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import torch

        return torch.from_numpy(self.items[index]), torch.ones(2, 2), 1, str(index)


def test_cached_rare_class_weights_reuses_file(tmp_path: Path) -> None:
    dataset = _Dataset(tmp_path)
    first = cached_class_presence_weights(dataset, (1, 2), power=0.5)
    cache_files = list(tmp_path.glob("sample_weights_*.npy"))
    assert len(cache_files) == 1
    second = cached_class_presence_weights(dataset, (1, 2), power=0.5)
    assert second.shape == first.shape
    assert np.load(cache_files[0]).shape == first.shape
