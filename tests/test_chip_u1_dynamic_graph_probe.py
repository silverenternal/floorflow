import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_u1_probe_remaps_both_edge_types() -> None:
    path = ROOT / "scripts/audit_chip_u1_dynamic_graph.py"
    spec = importlib.util.spec_from_file_location("u1_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch

    batch = {
        "areas": torch.arange(3.0)[None],
        "boxes": torch.zeros(1, 3, 4),
        "constraints": torch.zeros(1, 3, 5),
        "features": torch.zeros(1, 3, 19),
        "target_latent": torch.zeros(1, 3, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0]]]),
        "p2b": torch.tensor([[[0.0, 2.0, 1.0]]]),
    }
    permutation = torch.tensor([2, 0, 1])
    changed = module.remap_edges(batch, permutation)
    assert changed["b2b"][0, 0, :2].tolist() == [1.0, 2.0]
    assert changed["p2b"][0, 0, 1].item() == 0.0
