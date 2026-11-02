import sys

from floorflow.inference.dense_query_eval_cli import parse_args


def test_typed_readout_options_are_available(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--structure-checkpoint",
            "structure.pt",
            "--query-checkpoint",
            "query.pt",
            "--readout-seam-mode",
            "boundary_gated",
            "--readout-junction-degree-min",
            "3",
            "--readout-exterior-mode",
            "resolution_scaled",
            "--owner-cell-plan",
            "--structure-opening-interval-readout",
            "--graph-source",
            "plan",
            "--canonical-token-readout",
            "--output",
            "report.json",
        ],
    )
    args = parse_args()
    assert args.readout_seam_mode == "boundary_gated"
    assert args.readout_junction_degree_min == 3
    assert args.readout_exterior_mode == "resolution_scaled"
    assert args.owner_cell_plan
    assert args.structure_opening_interval_readout
    assert args.graph_source == "plan"
    assert args.canonical_token_readout


def test_wall_gated_room_seam_is_the_evaluation_default(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--structure-checkpoint",
            "structure.pt",
            "--query-checkpoint",
            "query.pt",
            "--output",
            "report.json",
        ],
    )
    args = parse_args()
    assert args.readout_seam_mode == "wall_gated"
    assert args.readout_room_boundary_threshold == 0.2
