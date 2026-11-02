from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.raster_control_bank import RasterInstanceControl
from floorflow.heads.raster_control_bank_loading import (
    load_raster_instance_control_model,
)


def test_instance_control_loader_restores_enhanced_head_flags(tmp_path) -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 1}),
            "window": Counter({"Regular Window": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    config = {
        "decoder_channels": 16,
        "hidden_dim": 32,
        "queries": 4,
        "layers": 1,
        "heads": 4,
        "full_resolution_masks": True,
        "full_resolution_channels": 8,
        "adaptive_query_refiner": True,
        "dual_control_raster": True,
        "full_resolution_class_sampler": True,
        "owner_semantic_reassignment": True,
        "room_boundary_owner_refiner": True,
        "joint_room_slot_state": True,
        "joint_slot_owner_fusion": True,
    }
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
        full_resolution_masks=True,
        full_resolution_channels=8,
        adaptive_query_refinement=True,
        dual_control=True,
        full_resolution_class_sampler=True,
        owner_semantic_reassignment=True,
        room_boundary_owner_refiner=True,
        joint_room_slot_state=True,
        joint_slot_owner_fusion=True,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "taxonomy": taxonomy.to_json(),
            "query_count_gate_effective": False,
        },
        tmp_path / "checkpoint.pt",
    )
    loaded = load_raster_instance_control_model(
        tmp_path / "checkpoint.pt",
        taxonomy,
        torch.device("cpu"),
    )
    assert loaded.mask_decoder.adaptive_query_refiner is not None
    assert loaded.control_projection is not None
    assert loaded.full_resolution_class_sampler is not None
    assert loaded.owner_semantic_reassignment is not None
    assert loaded.room_boundary_owner_refiner is not None
    assert loaded.joint_room_slot_state is not None
    assert loaded.joint_slot_owner_fusion
    assert not loaded.query_count_gate
    overridden = load_raster_instance_control_model(
        tmp_path / "checkpoint.pt",
        taxonomy,
        torch.device("cpu"),
        owner_semantic_reassignment=False,
    )
    assert overridden.owner_semantic_reassignment is None
    gated = load_raster_instance_control_model(
        tmp_path / "checkpoint.pt",
        taxonomy,
        torch.device("cpu"),
        query_count_gate=True,
    )
    assert gated.query_count_gate


def test_instance_control_loader_restores_query_set_mita_flags(tmp_path) -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 1}),
            "window": Counter({"Regular Window": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    config = {
        "decoder_channels": 16,
        "hidden_dim": 32,
        "queries": 4,
        "layers": 1,
        "heads": 4,
        "query_set_head": True,
        "query_set_hidden": 32,
        "query_set_pool": 8,
        "query_set_layers": 1,
        "query_set_routing_attention": True,
        "query_set_route_router_count": 4,
        "query_set_route_top_k": 2,
    }
    model_kwargs = dict(config)
    model_kwargs["num_queries"] = model_kwargs.pop("queries")
    model = RasterInstanceControl(taxonomy, **model_kwargs)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "taxonomy": taxonomy.to_json(),
        },
        tmp_path / "query_set_mita.pt",
    )
    loaded = load_raster_instance_control_model(
        tmp_path / "query_set_mita.pt",
        taxonomy,
        torch.device("cpu"),
    )
    assert loaded.query_set_head is not None
    assert loaded.query_set_head.routing_attention
    assert len(loaded.query_set_head.route_attention) == 1
