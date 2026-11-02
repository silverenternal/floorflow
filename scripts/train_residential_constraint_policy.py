"""Train CAGN on real model-origin candidates with hard-masked utility ranking."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.data.constraint_action_dataset import ACTION_NAMES
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.inference.constraint_action_runtime import (
    ActionCandidate,
    ConstraintActionRuntime,
    hard_rule_selection,
)
from floorflow.model.constraint_aware_gate import ConstraintAwareGate
from floorflow.model.constraint_state_encoder import room_family_ids
from floorflow.objectives.constraint_policy import (
    hard_masked_utility_loss,
    hard_masked_utility_pairwise_loss,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_INDEX = ROOT / "outputs/control_contracts/model_candidates_v1/train/index.json"
VAL_INDEX = ROOT / "outputs/control_contracts/model_candidates_v1/val/index.json"
INITIAL_CHECKPOINT = ROOT / "outputs/residential_cagn_gate_v3/seed19.pt"
OUTPUT_ROOT = ROOT / "outputs/residential_cagn_policy_v1"
REPORT = ROOT / "outputs/reports/residential_constraint_policy_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_groups(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["groups"]


def _candidate(row: dict, operation: str) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=row["candidate_id"],
        role=row["role"],
        node_features=torch.tensor(row["node_features"]),
        adjacency=torch.tensor(row["adjacency"]),
        global_features=torch.tensor(row["global_features"]),
        constraint_diagnostics=torch.tensor(row["constraint_diagnostics"]),
        action_features=torch.tensor(row["action_features"]),
        hard_mask=torch.tensor(
            list(row["hard_mask"].values()),
            dtype=torch.bool,
        ),
        declared_legal=bool(row["hard_feasible"]),
        operation_id=ACTION_NAMES.index(operation),
        utility=float(row["quality"]["utility"]),
    )


def _batch_tensors(
    groups: list[dict],
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    candidates = [
        _candidate(row, group["operation"])
        for group in groups
        for row in group["candidates"]
    ]
    count = len(groups[0]["candidates"])
    if any(len(group["candidates"]) != count for group in groups):
        raise ValueError("candidate counts differ across groups")
    node_features = torch.stack(
        [candidate.node_features for candidate in candidates]
    ).to(device)
    adjacency = torch.stack(
        [candidate.adjacency for candidate in candidates]
    ).to(device)
    context = torch.cat(
        [
            torch.stack(
                [candidate.global_features for candidate in candidates]
            ),
            torch.stack(
                [candidate.constraint_diagnostics for candidate in candidates]
            ),
        ],
        dim=-1,
    ).to(device)
    action = torch.stack(
        [candidate.action_features for candidate in candidates]
    ).to(device)
    hard = torch.stack(
        [candidate.hard_mask for candidate in candidates]
    ).to(device)
    declared = torch.tensor(
        [candidate.declared_legal for candidate in candidates],
        device=device,
        dtype=torch.bool,
    )
    hard = hard & declared[:, None]
    operation_ids = torch.tensor(
        [candidate.operation_id for candidate in candidates],
        device=device,
    )
    family_ids = room_family_ids(taxonomy).to(device)
    family_ids = family_ids[None].expand(len(candidates), -1)
    utility = torch.tensor(
        [candidate.utility for candidate in candidates],
        device=device,
    ).reshape(len(groups), count)
    legal = torch.tensor(
        [candidate.role != "near_violation" for candidate in candidates],
        device=device,
    ).reshape(len(groups), count)
    return node_features, adjacency, family_ids, operation_ids, action, hard, context, utility, legal


def _forward(
    model: ConstraintAwareGate,
    groups: list[dict],
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = _batch_tensors(groups, taxonomy, device)
    nodes, adjacency, family, operations, action, hard, context, utility, legal = tensors
    output = model(
        nodes,
        adjacency,
        family,
        operations,
        action,
        hard,
        global_features=context,
    )
    candidates = len(groups[0]["candidates"])
    for key in (
        "gate_logits",
        "gate_probability",
        "legal_logits",
        "uncertainty",
        "hard_feasible",
    ):
        if key in output:
            output[key] = output[key].reshape(len(groups), candidates)
    return (
        output,
        utility,
        hard.reshape(len(groups), candidates, -1),
        legal,
    )


def _summary(
    groups: list[dict],
    decisions: list,
    *,
    label: str,
) -> tuple[dict, list[dict]]:
    by_case = defaultdict(list)
    legal = []
    feasibility = []
    invalid_mass = []
    quality = defaultdict(list)
    for group, decision in zip(groups, decisions, strict=True):
        candidate = group["candidates"][decision.index]
        values = candidate["quality"]
        by_case[group["name"]].append(float(values["utility"]))
        for key, value in values.items():
            quality[key].append(float(value))
        legal.append(candidate["role"] != "near_violation")
        feasibility.append(bool(decision.hard_feasible))
        invalid_mass.append(float(decision.invalid_probability_mass))
    rows = [
        {"name": name, "utility": sum(values) / len(values)}
        for name, values in sorted(by_case.items())
    ]
    return {
        "method": label,
        "groups": len(groups),
        "cases": len(rows),
        "selected_legal_rate": sum(legal) / len(legal),
        "hard_feasible_rate": sum(feasibility) / len(feasibility),
        "invalid_probability_mass": sum(invalid_mass),
        "quality": {
            key: sum(values) / len(values)
            for key, values in quality.items()
        },
    }, rows


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    train = _load_groups(TRAIN_INDEX)
    validation = _load_groups(VAL_INDEX)
    checkpoint = torch.load(
        INITIAL_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    model = ConstraintAwareGate(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            batch = [train[index] for index in order[start : start + args.batch_size]]
            output, utility, hard, legal = _forward(
                model,
                batch,
                taxonomy,
                device,
            )
            ranking = hard_masked_utility_loss(
                output["gate_logits"],
                utility,
                hard.all(dim=-1),
            )
            pairwise = hard_masked_utility_pairwise_loss(
                output["gate_logits"],
                utility,
                hard.all(dim=-1),
            )
            legal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["legal_logits"][hard.all(dim=-1)],
                legal.float()[hard.all(dim=-1)],
            )
            loss = ranking + 0.5 * pairwise + 0.1 * legal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        history.append({"epoch": epoch, "loss": total / max(batches, 1)})
        print(
            f"epoch={epoch} loss={history[-1]['loss']:.5f}",
            flush=True,
        )
    model.eval()
    runtime = ConstraintActionRuntime(model)
    cagn_decisions = []
    hard_decisions = []
    for group in validation:
        candidates = [
            _candidate(row, group["operation"])
            for row in group["candidates"]
        ]
        hard_decisions.append(hard_rule_selection(candidates))
        cagn_decisions.append(
            runtime.select(candidates, taxonomy, device=device)
        )
    hard_summary, hard_rows = _summary(
        validation,
        hard_decisions,
        label="hard_rule_hash",
    )
    cagn_summary, cagn_rows = _summary(
        validation,
        cagn_decisions,
        label="model_origin_cagn",
    )
    paired = paired_delta(
        cagn_rows,
        hard_rows,
        "utility",
        iterations=args.bootstrap,
        seed=args.seed + 7000,
    )
    checks = {
        "hard_feasible_selection": cagn_summary["hard_feasible_rate"] == 1.0,
        "invalid_probability_mass_zero": (
            cagn_summary["invalid_probability_mass"] == 0.0
        ),
        "selected_legal_rate": cagn_summary["selected_legal_rate"] == 1.0,
        "utility_improves": (
            paired["mean_delta"] > 0.0 and paired["ci_low"] > 0.0
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_checkpoint = OUTPUT_ROOT / f"seed{args.seed}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": checkpoint["config"],
            "seed": args.seed,
            "epochs": args.epochs,
        },
        output_checkpoint,
    )
    report = {
        "schema_version": "residential-constraint-policy-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only_validation",
        "candidate_origin": "real model predicted controls",
        "single_trajectory": True,
        "training_cases": len({group["name"] for group in train}),
        "validation_cases": len({group["name"] for group in validation}),
        "history": history,
        "hard_rule": hard_summary,
        "model_origin_cagn": cagn_summary,
        "paired_utility": paired,
        "checks": checks,
        "checkpoint": str(output_checkpoint.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
