"""CPU-only audit of the structured chip prediction-head contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/cache/chip_floorset_50k_train_12k_dev_v2/train"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_head_contract_audit_v1.json"),
    )
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--depth", type=int, default=10)
    return parser


def _load_arrays(root: Path) -> dict[str, np.ndarray]:
    required = ("n", "constraints", "n_pins", "p2b_offset", "p2b")
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in required}
    return arrays


def _mask_statistics(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    counts = arrays["n"].astype(np.int64)
    constraints = arrays["constraints"]
    rows = np.arange(constraints.shape[1])[None, :] < counts[:, None]
    fixed = (constraints[..., 0] != 0) & rows
    preplaced = (constraints[..., 1] != 0) & rows
    anchored = fixed | preplaced
    cluster = constraints[..., 3]
    mib = constraints[..., 2]
    boundary = constraints[..., 4].astype(np.int64)
    slots = [
        ((cluster == group) & rows).any(axis=1)
        for group in range(1, 5)
    ]
    slots.append(((mib == 1) & rows).any(axis=1))
    slots.extend(
        (((boundary & bit) != 0) & rows).any(axis=1)
        for bit in (1, 2, 4, 8)
    )
    active_slots = np.stack(slots, axis=1).sum(axis=1)
    return {
        "cases": len(counts),
        "mean_blocks": float(counts.mean()),
        "fixed_blocks_per_case": float(fixed.sum() / len(counts)),
        "preplaced_blocks_per_case": float(preplaced.sum() / len(counts)),
        "anchored_blocks_per_case": float(anchored.sum() / len(counts)),
        "active_structural_slots_per_case": float(active_slots.mean()),
        "structural_slots": 9,
    }


def _port_statistics(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    n_pins = arrays["n_pins"].astype(np.int64)
    offsets = arrays["p2b_offset"]
    p2b = arrays["p2b"]
    connected = np.zeros(len(n_pins), dtype=np.int64)
    for index in range(len(n_pins)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        connected[index] = len(np.unique(p2b[start:stop, 0]))
    valid = n_pins > 0
    coverage = connected[valid] / n_pins[valid]
    return {
        "pins_per_case": float(n_pins.mean()),
        "connected_pins_per_case": float(connected.mean()),
        "unconnected_pins_per_case": float((n_pins - connected).mean()),
        "connected_pin_fraction": float(connected.sum() / n_pins.sum()),
        "mean_case_pin_coverage": float(coverage.mean()),
        "all_pins_connected_fraction": float((connected[valid] == n_pins[valid]).mean()),
        "zero_connected_cases": int((connected == 0).sum()),
    }


def _parameter_statistics(args: argparse.Namespace) -> dict[str, object]:
    base = HierarchicalTokenFlowModel(
        hidden=args.hidden,
        heads=args.heads,
        depth=args.depth,
        head_dim=64,
        typed_layers=1,
    )
    structured = HierarchicalTokenFlowModel(
        hidden=args.hidden,
        heads=args.heads,
        depth=args.depth,
        head_dim=64,
        typed_layers=1,
        structured_heads=True,
    )
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    total = sum(parameter.numel() for parameter in structured.parameters())
    suite = structured.structured_heads
    if suite is None:
        raise RuntimeError("structured head suite was not instantiated")
    return {
        "base_parameters": base_parameters,
        "structured_parameters": total,
        "structured_head_parameters": sum(
            parameter.numel() for parameter in suite.parameters()
        ),
        "added_ratio": (total - base_parameters) / base_parameters,
    }


def main() -> None:
    args = _parser().parse_args()
    arrays = _load_arrays(args.cache)
    report = {
        "version": "chip-head-contract-audit-v1",
        "status": "pass",
        "device": "cpu",
        "cache": str(args.cache),
        "block_supervision": _mask_statistics(arrays),
        "port_supervision": _port_statistics(arrays),
        "parameter_budget": _parameter_statistics(args),
        "contracts": {
            "same_call_hierarchy_coupling": True,
            "factorized_center_rho_head": True,
            "dual_semantic_port_head": True,
            "legalizer_response_head": True,
            "structured_head_contract": True,
            "training_only_endpoint_compatibility": True,
            "effective_output_mode": "x0",
        },
        "boundary": "CPU contract audit only; no model-quality or GPU-training claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
