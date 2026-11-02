from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.legalization import translate_into_core
from floorflow.chip_flow.ppa_external import (
    export_macro_tcl,
    floorplan_instance,
    guard_small_graph_pe,
    infer_hierarchy_clusters,
    model_batch,
    parse_orfs_graph,
    physical_boxes,
    placement_connectivity,
    restore_fixed_dimensions,
    shelf_pack_fixed,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_chip_ppa_external",
    ROOT / "scripts/run_chip_ppa_external.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
PRIMARY_KEYS = MODULE.PRIMARY_KEYS

ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "run_chip_def_adapter_suite",
    ROOT / "scripts/run_chip_def_adapter_suite.py",
)
ADAPTER_MODULE = importlib.util.module_from_spec(ADAPTER_SPEC)
assert ADAPTER_SPEC.loader is not None
ADAPTER_SPEC.loader.exec_module(ADAPTER_MODULE)

SAMPLE = (
    "ORFS_DBU_PER_MICRON\t1000\n"
    "ORFS_DIE\t0\t0\t20000\t10000\n"
    "ORFS_CORE\t1000\t1000\t19000\t9000\n"
    "ORFS_MACRO\tm0\tsram_a\t2000\t1000\t0\t0\tUNPLACED\tR0\n"
    "ORFS_MACRO\tm1\tsram_b\t1000\t2000\t0\t0\tUNPLACED\tR0\n"
    "ORFS_PORT\tclk\t0\t5000\n"
    "ORFS_PORT\tq\t20000\t5000\n"
    "ORFS_NET_ITERM\tnet0\tm0\tclk\n"
    "ORFS_NET_ITERM\tnet0\tm1\tclk\n"
    "ORFS_NET_BTERM\tnet0\tclk\n"
    "ORFS_NET_ITERM\tnet1\tm0\tq\n"
    "ORFS_NET_BTERM\tnet1\tq"
)


def test_parse_orfs_graph_and_edges() -> None:
    graph = parse_orfs_graph(SAMPLE)
    assert graph.dbu_per_micron == 1000
    assert graph.core == (1.0, 1.0, 19.0, 9.0)
    assert [macro["name"] for macro in graph.macros] == ["m0", "m1"]

    batch = model_batch(graph, torch.device("cpu"))
    assert batch["n"] == 2
    assert batch["areas"].shape == (1, 2)
    assert batch["constraints"][0, :, 0].tolist() == [1.0, 1.0]
    assert batch["n_b2b"].item() == 1
    assert batch["n_p2b"].item() == 3
    assert batch["n_pins"].item() == 2
    assert torch.all(batch["source_latent"].isfinite())
    assert torch.all(batch["target_latent"].isfinite())


def test_case_record_accepts_design_key_and_platform_alias() -> None:
    manifest = {
        "inventory": [
            {
                "case": "cva6",
                "design_key": "nangate45/cva6",
                "platform": "nangate45",
                "design_nickname": "cva6",
            }
        ]
    }

    assert MODULE.case_record(manifest, "nangate45/cva6")["platform"] == "nangate45"
    assert (
        MODULE.case_record(manifest, "nangate45__cva6")["design_key"]
        == "nangate45/cva6"
    )


def test_floorplan_instance_preserves_external_shapes_and_outline() -> None:
    graph = parse_orfs_graph(SAMPLE)
    instance = floorplan_instance(graph)
    assert instance.block_count == 2
    assert instance.outline == (18.0, 8.0)
    assert np.array_equal(instance.is_fixed, np.ones(2))
    assert instance.target_positions[:, 2:].tolist() == [[2.0, 1.0], [1.0, 2.0]]


def test_physical_boxes_and_tcl_are_deterministic() -> None:
    graph = parse_orfs_graph(SAMPLE)
    latent = torch.tensor(
        [[[0.4, 0.5, 0.0], [0.6, 0.5, 0.0]]],
        dtype=torch.float32,
    )
    boxes = physical_boxes(latent, graph)
    assert boxes.shape == (2, 4)
    assert np.allclose(boxes[:, 2:], [[2.0, 1.0], [1.0, 2.0]])
    connectivity = placement_connectivity(boxes, graph)
    assert connectivity["b2b_hpwl_um"] > 0
    assert connectivity["p2b_hpwl_um"] > 0
    tcl = export_macro_tcl(graph, boxes)
    assert "-macro_name {m0}" in tcl
    assert "-orientation R0" in tcl
    assert tcl.count("place_macro") == 2


def test_reference_orientation_can_be_preserved_explicitly() -> None:
    graph = parse_orfs_graph(
        SAMPLE.replace(
            "ORFS_MACRO\tm0\tsram_a\t2000\t1000\t0\t0\tUNPLACED\tR0",
            "ORFS_MACRO\tm0\tsram_a\t2000\t1000\t0\t0\tPLACED\tMY",
        )
    )
    boxes = np.array([[0.0, 0.0, 2.0, 1.0], [4.0, 0.0, 1.0, 2.0]])
    assert "-orientation R0" in export_macro_tcl(graph, boxes)
    assert "-orientation MY" in export_macro_tcl(
        graph,
        boxes,
        preserve_reference_orientation=True,
    )


def test_small_graph_pe_uses_documented_zero_fallback() -> None:
    class FakeGraphPE(torch.nn.Module):
        k = 2
        dim = 2

        def batched(self, adjacency: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
            del key_mask
            return adjacency.new_zeros((adjacency.shape[0], adjacency.shape[1], 1))

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph_pe = FakeGraphPE()

    model = FakeModel()
    assert guard_small_graph_pe(model, 2)
    adjacency = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert model.graph_pe.batched(adjacency, mask).shape == (1, 2, 2)


def test_instance_hierarchy_produces_model_cluster_ids() -> None:
    graph = parse_orfs_graph(
        SAMPLE.replace("ORFS_MACRO\tm0\t", "ORFS_MACRO\tchip.core.be/m0\t")
        .replace("ORFS_MACRO\tm1\t", "ORFS_MACRO\tchip.core.fe/m1\t")
        .replace("\tm0\t", "\tchip.core.be/m0\t")
        .replace("\tm1\t", "\tchip.core.fe/m1\t")
    )
    clusters = infer_hierarchy_clusters(graph)
    assert set(clusters.values()) == {1, 2}


def test_metrics_audit_retains_predeclared_congestion_endpoint() -> None:
    assert "globalplace__gpl__routability__congestion" in PRIMARY_KEYS


def test_frozen_r0_policy_ignores_stale_reference_orientation() -> None:
    row = {"orientation": "MY"}
    assert ADAPTER_MODULE.expected_macro_orientation(row, "R0") == "R0"
    assert ADAPTER_MODULE.expected_macro_orientation(row, "adapter_report") == "MY"


def test_coordinate_tolerance_can_be_overridden_per_case() -> None:
    policy = {"coordinate_roundtrip_tolerance_um": 0.19}
    assert ADAPTER_MODULE.coordinate_tolerance({}, policy) == 0.19
    assert (
        ADAPTER_MODULE.coordinate_tolerance(
            {"coordinate_roundtrip_tolerance_um": 0.054},
            policy,
        )
        == 0.054
    )


def test_adapter_case_slug_is_unique_across_platforms() -> None:
    assert (
        ADAPTER_MODULE.case_slug(
            {"platform": "asap7", "design_nickname": "cva6"}
        )
        != ADAPTER_MODULE.case_slug(
            {"platform": "nangate45", "design_nickname": "cva6"}
        )
    )


def test_adapter_flow_variant_is_unique_across_platforms() -> None:
    assert (
        ADAPTER_MODULE.adapter_flow_variant(
            {"platform": "asap7", "design_nickname": "cva6"},
            "candidate",
            19,
        )
        != ADAPTER_MODULE.adapter_flow_variant(
            {"platform": "nangate45", "design_nickname": "cva6"},
            "candidate",
            19,
        )
    )


def test_shelf_pack_uses_decreasing_height_to_avoid_row_waste() -> None:
    boxes = np.asarray(
        [
            [0.0, 1.0, 1.0, 8.0],
            [0.0, 8.0, 9.0, 1.0],
            [2.0, 1.0, 1.0, 8.0],
        ],
        dtype=np.float64,
    )

    packed = shelf_pack_fixed(boxes, origin=(0.0, 0.0), outline=(10.0, 10.0))

    assert packed is not None
    assert np.all(packed[:, 0] >= 0.0)
    assert np.all(packed[:, 1] >= 0.0)
    assert np.all(packed[:, 0] + packed[:, 2] <= 10.0)
    assert np.all(packed[:, 1] + packed[:, 3] <= 10.0)
    left = np.maximum(packed[:, None, 0], packed[None, :, 0])
    right = np.minimum(
        packed[:, None, 0] + packed[:, None, 2],
        packed[None, :, 0] + packed[None, :, 2],
    )
    bottom = np.maximum(packed[:, None, 1], packed[None, :, 1])
    top = np.minimum(
        packed[:, None, 1] + packed[:, None, 3],
        packed[None, :, 1] + packed[None, :, 3],
    )
    overlap = (right > left) & (top > bottom)
    assert not np.any(np.triu(overlap, k=1))


def test_shelf_pack_position_ties_preserve_proposal_order() -> None:
    boxes = np.asarray(
        [
            [8.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [4.0, 0.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    size_only = shelf_pack_fixed(boxes, origin=(0.0, 0.0), outline=(6.0, 6.0))
    positioned = shelf_pack_fixed(
        boxes,
        origin=(0.0, 0.0),
        outline=(6.0, 6.0),
        position_aware_ties=True,
    )

    assert size_only is not None
    assert positioned is not None
    assert not np.array_equal(size_only, positioned)
    assert np.all(positioned[:, 2:] == boxes[:, 2:])


def test_restore_fixed_dimensions_matches_graph_dbu() -> None:
    graph = parse_orfs_graph(SAMPLE)
    boxes = np.asarray(
        [[1.0, 2.0, 2.01, 1.02], [4.0, 5.0, 1.01, 2.02]],
        dtype=np.float64,
    )

    restored = restore_fixed_dimensions(boxes, graph)

    np.testing.assert_allclose(restored[:, 2], [2.0, 1.0])
    np.testing.assert_allclose(restored[:, 3], [1.0, 2.0])
    np.testing.assert_allclose(restored[:, :2], boxes[:, :2])


def test_translate_into_core_preserves_relative_geometry() -> None:
    boxes = np.asarray(
        [[-2.0, 1.0, 2.0, 2.0], [4.0, -1.0, 2.0, 2.0]],
        dtype=np.float64,
    )

    translated = translate_into_core(boxes, (0.0, 0.0, 10.0, 10.0))

    assert translated is not None
    assert np.all(translated[:, 0] >= 0.0)
    assert np.all(translated[:, 1] >= 0.0)
    assert np.all(translated[:, 0] + translated[:, 2] <= 10.0)
    assert np.all(translated[:, 1] + translated[:, 3] <= 10.0)
    original_delta = boxes[1, :2] - boxes[0, :2]
    translated_delta = translated[1, :2] - translated[0, :2]
    np.testing.assert_allclose(translated_delta, original_delta)


def test_translate_into_core_rejects_layout_larger_than_core() -> None:
    boxes = np.asarray([[0.0, 0.0, 11.0, 2.0]], dtype=np.float64)

    assert translate_into_core(boxes, (0.0, 0.0, 10.0, 10.0)) is None
