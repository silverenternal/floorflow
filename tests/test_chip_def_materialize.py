from pathlib import Path

from floorflow.chip_flow.def_materialize import materialize_macro_def


def test_materialize_only_replaces_named_macros(tmp_path: Path) -> None:
    source = tmp_path / "input.def"
    source.write_text(
        "VERSION 5.8 ;\n"
        "COMPONENTS 2 ;\n"
        "  - macro_a RAM + PLACED ( 0 0 ) N ;\n"
        "  - cell_a BUF_X1 + PLACED ( 0 0 ) N ;\n"
        "END COMPONENTS\n"
        "BLOCKAGES 1 ;\n"
        "  - PLACEMENT\n"
        "END BLOCKAGES\n"
        "END DESIGN\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.def"
    boxes = {
        "macro_a": {
            "name": "macro_a",
            "x_um": 2.5,
            "y_um": -1.25,
            "orientation": "S",
        }
    }
    report = materialize_macro_def(
        source,
        output,
        boxes,
        dbu_per_micron=2000,
        remove_blockages=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "- macro_a RAM + FIXED ( 5000 -2500 ) S ;" in text
    assert "- cell_a BUF_X1 + PLACED ( 0 0 ) N ;" in text
    assert "BLOCKAGES" not in text
    assert report["macro_count"] == 1
