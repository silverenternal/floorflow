"""Parameter-controlled topology-native FloorFlow v2 generator factory."""

from __future__ import annotations

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)

FLOORFLOW_V2_CONFIG = UnifiedGeneratorConfig(
    hidden_dim=96,
    decoder_channels=96,
    identity_queries=12,
    identity_layers=2,
    vector_layers=2,
    polygon_points=24,
    max_walls=128,
    max_openings=8,
    attention_heads=4,
    route_top_k=16,
    max_boundary_points=64,
    pretrained_backbone=False,
    use_mask_anchors=True,
    share_identity_heads=True,
    enable_query_quality=False,
    enable_point_validity=True,
    use_boundary_edge_graph=True,
    encoder_architecture="compact",
    compact_encoder_channels=(64, 128, 192, 288),
    compact_encoder_blocks=(1, 3, 4, 2),
    enable_program_set=True,
)


def build_floorflow_v2_generator(
    taxonomy: FloorplanTaxonomy,
) -> UnifiedFloorplanGenerator:
    """Construct the architecture-first FloorFlow v2 candidate."""

    return UnifiedFloorplanGenerator(taxonomy, FLOORFLOW_V2_CONFIG)


__all__ = ["FLOORFLOW_V2_CONFIG", "build_floorflow_v2_generator"]
