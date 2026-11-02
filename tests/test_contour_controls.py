from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.contour_controls import predict_contour_controls
from floorflow.model.builder import ModelMacro
from floorflow.model.specs import ArchSpec


def test_predict_contour_controls_shapes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    spec = ArchSpec(
        name="contour-control-test",
        target_channels=taxonomy.num_target_channels,
        base_channels=16,
        levels=2,
        resolution=(32, 32),
        backbone_kwargs={
            "semantic_channel_attention": False,
            "slot_mixing": False,
            "floor_conditioning": False,
            "topology_aux": False,
            "semantic_endpoint_head": False,
            "fourier_conditioning": False,
            "family_streams": True,
            "family_mita_attention": True,
            "room_partition_head": True,
            "num_room_classes": len(taxonomy.room_ids),
            "room_program_head": True,
            "structured_object_head": True,
            "semantic_taxonomy": taxonomy,
            "geometry_heads": True,
            "geometry_hidden": 16,
            "geometry_experts": 2,
            "geometry_top_k": 1,
        },
    )
    model = ModelMacro(spec).build()
    controls = predict_contour_controls(
        model,
        torch.ones(1, 32, 32),
        taxonomy,
        flow_steps=0,
        structure_size=8,
        furniture_size=12,
        support_size=12,
    )
    assert controls.room_logits.shape == (1, len(taxonomy.room_ids), 32, 32)
    assert controls.program.shape == (1, len(taxonomy.room_ids))
    assert controls.structure.shape == (
        1,
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        8,
        8,
    )
    assert controls.furniture.shape == (
        1,
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        12,
        12,
    )
    assert controls.support.shape == (
        1,
        len(taxonomy.room_ids) + 4,
        12,
        12,
    )
