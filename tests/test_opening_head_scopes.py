from __future__ import annotations

from floorflow.heads.opening_detection import OpeningDetectionHead
from floorflow.training.opening_head_scopes import (
    configure_opening_head_scope,
)


def test_opening_head_scopes_isolate_gap_outputs() -> None:
    head = OpeningDetectionHead(8, hidden_dim=16, variant="center_width")
    manifest = configure_opening_head_scope(head, "center")
    assert head.center is not None
    assert all(parameter.requires_grad for parameter in head.center.parameters())
    assert not any(
        parameter.requires_grad for parameter in head.width.parameters()
    )
    assert manifest["trainable_count"] > 0

    configure_opening_head_scope(head, "width_family")
    assert head.width is not None
    assert head.family is not None
    assert all(parameter.requires_grad for parameter in head.width.parameters())
    assert all(parameter.requires_grad for parameter in head.family.parameters())
    assert not any(
        parameter.requires_grad for parameter in head.center.parameters()
    )


def test_opening_output_head_scope_freezes_shared_trunk() -> None:
    head = OpeningDetectionHead(8, hidden_dim=16, variant="center_width")
    configure_opening_head_scope(head, "output_heads")
    assert not any(
        parameter.requires_grad for parameter in head.stem.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in head.foreground.parameters()
    )

    configure_opening_head_scope(head, "gap_fusion")
    assert all(
        parameter.requires_grad for parameter in head.gap_attention.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in head.center.parameters()
    )

