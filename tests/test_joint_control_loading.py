from pathlib import Path

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.joint_control_loading import load_joint_control_repair
from floorflow.heads.joint_control_repair import JointControlRepair


def _taxonomy():
    from collections import Counter

    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Door": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_joint_control_loader_resolves_typed_contract(tmp_path: Path) -> None:
    taxonomy = _taxonomy()
    source = JointControlRepair(
        1,
        1,
        room_base_channels=4,
        structure_base_channels=4,
        structure_contract="typed_v2",
    )
    checkpoint = tmp_path / "typed.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "config": {
                "room_base_channels": 4,
                "structure_base_channels": 4,
                "structure_contract": "typed_v2",
            },
        },
        checkpoint,
    )
    loaded, config = load_joint_control_repair(
        checkpoint,
        taxonomy,
        torch.device("cpu"),
    )
    assert config["structure_contract"] == "typed_v2"
    assert loaded.structure_contract == "typed_v2"
    assert loaded.training is False


def test_joint_control_loader_restores_structure_mita(tmp_path: Path) -> None:
    taxonomy = _taxonomy()
    source = JointControlRepair(
        1,
        1,
        room_base_channels=4,
        structure_base_channels=4,
        structure_contract="typed_v2",
        structure_global_route_attention=True,
        structure_route_grid=4,
        structure_route_router_count=4,
        structure_route_top_k=2,
        structure_route_heads=4,
    )
    checkpoint = tmp_path / "structure_mita.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "config": {
                "room_base_channels": 4,
                "structure_base_channels": 4,
                "structure_contract": "typed_v2",
                "structure_global_route_attention": True,
                "structure_route_grid": 4,
                "structure_route_router_count": 4,
                "structure_route_top_k": 2,
                "structure_route_heads": 4,
            },
        },
        checkpoint,
    )
    loaded, _config = load_joint_control_repair(
        checkpoint,
        taxonomy,
        torch.device("cpu"),
    )
    assert loaded.structure_refiner.global_route is not None
    assert loaded.structure_refiner.global_route.router_count == 4
