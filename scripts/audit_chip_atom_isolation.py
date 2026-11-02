"""Audit atom ownership and the isolation schedule on CPU."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from floorflow.chip_flow.flow_adapter import build_chip_model, count_parameters
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.training_scopes import (
    parse_scope_schedule,
    scope_owners,
    validate_atom_isolation_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_architecture_mainline_v4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_atom_isolation_audit_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit_model(config: dict[str, Any]):
    architecture = dict(config["architecture"])
    architecture.update(
        {
            "architecture": "typed_hierarchy",
            "hidden": 16,
            "heads": 2,
            "depth": 1,
            "head_dim": 16,
        }
    )
    return build_chip_model({**architecture, **config["branches"]})


def audit_ownership(model: torch.nn.Module) -> dict[str, Any]:
    unowned = []
    duplicated = []
    by_owner: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        if name.startswith("backbone."):
            continue
        owners = scope_owners(name)
        if not owners:
            unowned.append(name)
        if len(owners) > 1:
            duplicated.append({"parameter": name, "owners": list(owners)})
        by_owner.update(owners)
    return {
        "unowned": unowned,
        "duplicated": duplicated,
        "parameter_tensors_by_owner": dict(sorted(by_owner.items())),
    }


def audit_schedule(config: dict[str, Any]) -> dict[str, Any]:
    stages = config["training_stages"]
    schedule = parse_scope_schedule(stages["certification_schedule"])
    validation = validate_atom_isolation_schedule(schedule)
    return {
        "require_atom_isolation": bool(stages["require_atom_isolation"]),
        "stages": [
            {"start": transition.start, "scope": transition.scope.name}
            for transition in schedule
        ],
        **validation,
    }


def main() -> None:
    args = parse_args()
    config = load(args.config.resolve())
    model = build_audit_model(config)
    ownership = audit_ownership(model)
    schedule = audit_schedule(config)
    report = {
        "version": "chip-atom-isolation-audit-v1",
        "status": (
            "pass"
            if not ownership["unowned"]
            and not ownership["duplicated"]
            and schedule["require_atom_isolation"]
            else "fail"
        ),
        "config": str(args.config.resolve()),
        "audit_model_parameters": count_parameters(model),
        "ownership": ownership,
        "schedule": schedule,
        "boundary": "CPU ownership and schedule audit only; no quality claim.",
    }
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
