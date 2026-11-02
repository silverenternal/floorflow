from __future__ import annotations

import pytest

from floorflow.chip_flow.def_pin_alignment import align_boxes_to_reference


def _reference(path, *, second_x=303, second_y=120):
    path.write_text(
        "\n".join(
            [
                "UNITS DISTANCE MICRONS 2000 ;",
                "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                "COMPONENTS 2 ;",
                "  - a M + FIXED ( 103 101 ) N ;",
        f"  - b M + FIXED ( {second_x} {second_y} ) N ;",
                "END COMPONENTS",
            ]
        )
    )


def _boxes():
    return {
        "a": {
            "name": "a",
            "x_um": 0.13,
            "y_um": 0.11,
            "width_um": 0.1,
            "height_um": 0.1,
            "orientation": "N",
        },
        "b": {
            "name": "b",
            "x_um": 0.23,
            "y_um": 0.12,
            "width_um": 0.1,
            "height_um": 0.1,
            "orientation": "N",
        },
    }


def test_pin_alignment_preserves_reference_residue(tmp_path) -> None:
    reference = tmp_path / "reference.def"
    _reference(reference)
    aligned, report = align_boxes_to_reference(
        _boxes(),
        reference,
        grid_dbu=100,
    )
    assert round(aligned["a"]["x_um"] * 2000) % 100 == 3
    assert round(aligned["a"]["y_um"] * 2000) % 100 == 1
    assert report["overlap_pair_count"] == 0
    assert report["status"] == "pass"


def test_pin_alignment_rejects_overlap(tmp_path) -> None:
    reference = tmp_path / "reference.def"
    _reference(reference, second_x=300, second_y=125)
    boxes = _boxes()
    boxes["b"]["x_um"] = 0.25
    with pytest.raises(ValueError, match="overlaps"):
        align_boxes_to_reference(
            boxes,
            reference,
            grid_dbu=100,
        )


def test_pin_alignment_can_roll_back_conflicting_macro(tmp_path) -> None:
    reference = tmp_path / "reference.def"
    _reference(reference, second_x=300, second_y=125)
    boxes = _boxes()
    boxes["b"]["x_um"] = 0.25
    aligned, report = align_boxes_to_reference(
        boxes,
        reference,
        grid_dbu=100,
        rollback_conflicts=True,
    )
    assert report["status"] == "pass"
    assert report["overlap_pair_count"] == 0
    assert report["rolled_back_macro_count"] >= 1
    assert aligned["b"]["x_um"] == boxes["b"]["x_um"]
