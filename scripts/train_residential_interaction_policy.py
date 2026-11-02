"""Train matched one-step and interaction-aware residential action policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.data.interaction_candidate_dataset import index_sequences
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.inference.constraint_action_runtime import (
    ConstraintActionRuntime,
    hard_rule_selection,
)
from floorflow.model.interaction_constraint_gate import (
    InteractionConstraintAwareGate,
)
from floorflow.objectives.constraint_policy import (
    hard_masked_best_action_loss,
    hard_masked_utility_loss,
    hard_masked_utility_pairwise_loss,
)
from floorflow.training.interaction_policy_data import (
    candidate_from_row,
    forward_groups,
    lookahead_utility_map,
    normalized_utility,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_INDEX = (
    ROOT / "outputs/control_contracts/model_interactions_v2/train/index.json"
)
VAL_INDEX = ROOT / "outputs/control_contracts/model_interactions_v2/val/index.json"
INITIAL_CHECKPOINT = ROOT / "outputs/residential_cagn_gate_v3/seed19.pt"
OUTPUT_ROOT = ROOT / "outputs/residential_interaction_policy_v2"
REPORT = ROOT / "outputs/reports/residential_interaction_policy_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--interaction-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-index", type=Path, default=TRAIN_INDEX)
    parser.add_argument("--val-index", type=Path, default=VAL_INDEX)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="freeze the validation-selected one-step base before interaction training",
    )
    parser.add_argument(
        "--interaction-roots-only",
        action="store_true",
        help="train the plan adapter only on states with observable future actions",
    )
    parser.add_argument(
        "--arms",
        default="one_step,state_blind,interaction",
        help="comma-separated arms",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_groups(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") not in {"train", "val"}:
        raise ValueError("interaction training is restricted to train/val")
    return payload["groups"]


def _train_arm(
    *,
    arm: str,
    groups: list[dict],
    taxonomy: FloorplanTaxonomy,
    checkpoint: dict,
    args: argparse.Namespace,
    device: torch.device,
    utility_overrides: dict[str, float] | None,
    validation_sequences: dict[str, dict],
    source_state: dict[str, torch.Tensor] | None = None,
    freeze_base: bool = False,
    epochs: int | None = None,
) -> tuple[InteractionConstraintAwareGate, list[dict], dict]:
    _set_seed(args.seed)
    model = InteractionConstraintAwareGate(**checkpoint["config"]).to(device)
    if source_state is None:
        model.load_base_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(source_state)
    if freeze_base:
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    epochs = args.epochs if epochs is None else int(epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )
    generator = torch.Generator().manual_seed(args.seed)
    state_blind = arm == "state_blind"
    history = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    best_validation = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(groups), generator=generator).tolist()
        total = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            batch = [
                groups[index]
                for index in order[start : start + args.batch_size]
            ]
            output, utility, hard, legal = forward_groups(
                model,
                batch,
                taxonomy,
                device,
                state_blind=state_blind,
                use_sequence_context=arm == "interaction",
                utility_overrides=utility_overrides,
            )
            feasible = hard.all(dim=-1)
            normalized = normalized_utility(utility, feasible)
            ranking = hard_masked_utility_loss(
                output["gate_logits"],
                normalized,
                feasible,
            )
            pairwise = hard_masked_utility_pairwise_loss(
                output["gate_logits"],
                normalized,
                feasible,
            )
            best_action = hard_masked_best_action_loss(
                output["gate_logits"],
                normalized,
                feasible,
            )
            legal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["legal_logits"][feasible],
                legal.float()[feasible],
            )
            value_loss = output["gate_logits"].sum() * 0.0
            if "interaction_value" in output:
                value_loss = (
                    output["interaction_value"] - normalized
                ).pow(2).masked_select(feasible).mean()
            loss = (
                ranking
                + 0.5 * pairwise
                + best_action
                + 0.5 * value_loss
                + 0.1 * legal_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        validation, _ = _evaluate(
            validation_sequences,
            model,
            taxonomy,
            device,
            state_blind=state_blind,
            sequence_context=arm == "interaction",
        )
        score = validation["mean_terminal_utility"]
        history.append(
            {
                "epoch": epoch,
                "loss": total / max(batches, 1),
                "validation_utility": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"arm={arm} epoch={epoch} loss={history[-1]['loss']:.5f} "
            f"val={score:.6f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), history, {
        "epoch": best_epoch,
        "validation": best_validation,
    }


def _run_sequence(
    sequence: dict,
    runtime: ConstraintActionRuntime,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    *,
    hard_rule: bool,
) -> dict:
    group = sequence["root"]
    legal = []
    feasible = []
    invalid_mass = []
    step_utility = []
    roles = []
    selected = None
    while group is not None:
        candidates = [
            candidate_from_row(row, group["operation"])
            for row in group["candidates"]
        ]
        decision = (
            hard_rule_selection(candidates)
            if hard_rule
            else runtime.select(candidates, taxonomy, device=device)
        )
        selected = group["candidates"][decision.index]
        legal.append(selected["role"] != "near_violation")
        feasible.append(bool(decision.hard_feasible))
        invalid_mass.append(float(decision.invalid_probability_mass))
        step_utility.append(float(selected["quality"]["utility"]))
        roles.append(selected["role"])
        group = sequence["children"].get(decision.candidate_id)
    return {
        "name": sequence["root"]["name"],
        "trajectory_utility": (
            float(selected["trajectory_utility"])
            if selected is not None
            else 0.0
        ),
        "terminal_utility": (
            float(selected["quality"]["utility"])
            if selected is not None
            else 0.0
        ),
        "mean_step_utility": sum(step_utility) / max(len(step_utility), 1),
        "steps": len(step_utility),
        "selected_legal_rate": sum(legal) / max(len(legal), 1),
        "hard_feasible_rate": sum(feasible) / max(len(feasible), 1),
        "invalid_probability_mass": sum(invalid_mass),
        "roles": dict(sorted(defaultdict(int, {
            role: roles.count(role) for role in set(roles)
        }).items())),
    }


def _evaluate(
    sequences: dict[str, dict],
    model: InteractionConstraintAwareGate,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    *,
    state_blind: bool,
    sequence_context: bool = True,
) -> tuple[dict, list[dict]]:
    runtime = ConstraintActionRuntime(
        model,
        state_blind=state_blind,
        sequence_blind=not sequence_context,
    )
    rows = [
        _run_sequence(
            sequence,
            runtime,
            taxonomy,
            device,
            hard_rule=False,
        )
        for _, sequence in sorted(sequences.items())
    ]
    return _summary(rows), rows


def _summary(rows: list[dict]) -> dict:
    return {
        "sequences": len(rows),
        "mean_trajectory_utility": (
            sum(row["trajectory_utility"] for row in rows) / len(rows)
        ),
        "mean_terminal_utility": (
            sum(row["terminal_utility"] for row in rows) / len(rows)
        ),
        "mean_step_utility": (
            sum(row["mean_step_utility"] for row in rows) / len(rows)
        ),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "selected_legal_rate": (
            sum(row["selected_legal_rate"] for row in rows) / len(rows)
        ),
        "hard_feasible_rate": (
            sum(row["hard_feasible_rate"] for row in rows) / len(rows)
        ),
        "invalid_probability_mass": sum(
            row["invalid_probability_mass"] for row in rows
        ),
    }


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(
        ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    train_groups = _load_groups(args.train_index)
    val_groups = _load_groups(args.val_index)
    one_step_groups = [group for group in train_groups if group["step"] == 0]
    lookahead_utility = lookahead_utility_map(train_groups)
    sequences = index_sequences(val_groups)
    initial = torch.load(
        INITIAL_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    requested_arms = set(filter(None, args.arms.split(",")))
    allowed_arms = {"one_step", "state_blind", "interaction"}
    if not requested_arms or not requested_arms <= allowed_arms:
        raise ValueError(f"arms must be a subset of {sorted(allowed_arms)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    rows_by_arm = {}
    histories = {}
    checkpoints = {}
    one_step_path = None
    for arm in ("one_step", "state_blind", "interaction"):
        if arm not in requested_arms:
            continue
        training_groups = (
            one_step_groups if arm == "one_step" else train_groups
        )
        if arm == "interaction" and args.interaction_roots_only:
            training_groups = one_step_groups
        source_state = None
        if arm == "interaction" and args.adapter_only:
            if one_step_path is None or not one_step_path.exists():
                raise ValueError("adapter-only mode requires a trained one-step arm")
            source_state = torch.load(
                one_step_path,
                map_location=device,
                weights_only=False,
            )["model"]
        model, history, selection = _train_arm(
            arm=arm,
            groups=training_groups,
            taxonomy=taxonomy,
            checkpoint=initial,
            args=args,
            device=device,
            utility_overrides=(
                lookahead_utility if arm != "one_step" else None
            ),
            validation_sequences=sequences,
            source_state=source_state,
            freeze_base=arm == "interaction" and args.adapter_only,
            epochs=(
                args.interaction_epochs
                if arm == "interaction" and args.interaction_epochs
                else args.epochs
            ),
        )
        summary, rows = _evaluate(
            sequences,
            model,
            taxonomy,
            device,
            state_blind=arm == "state_blind",
            sequence_context=arm == "interaction",
        )
        checkpoint_path = args.output_root / f"{arm}_seed{args.seed}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": initial["config"],
                "arm": arm,
                "seed": args.seed,
                "epochs": args.epochs,
                "selected_epoch": selection["epoch"],
            },
            checkpoint_path,
        )
        if arm == "one_step":
            one_step_path = checkpoint_path
        summaries[arm] = summary
        rows_by_arm[arm] = rows
        histories[arm] = history
        checkpoints[arm] = {
            "path": (
                str(checkpoint_path.relative_to(ROOT))
                if checkpoint_path.is_absolute()
                else str(checkpoint_path)
            ),
            "sha256": _checkpoint_hash(checkpoint_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_groups": len(training_groups),
            "selected_epoch": selection["epoch"],
            "validation_selection": selection["validation"],
        }
    hard_runtime = ConstraintActionRuntime(model)
    hard_rows = [
        _run_sequence(
            sequence,
            hard_runtime,
            taxonomy,
            device,
            hard_rule=True,
        )
        for _, sequence in sorted(sequences.items())
    ]
    summaries["hard_rule"] = _summary(hard_rows)
    comparisons = {}
    if "interaction" in rows_by_arm:
        for baseline in ("one_step", "state_blind", "hard_rule"):
            if baseline == "hard_rule":
                baseline_rows = hard_rows
            elif baseline in rows_by_arm:
                baseline_rows = rows_by_arm[baseline]
            else:
                continue
            comparisons[f"interaction_vs_{baseline}"] = paired_delta(
                rows_by_arm["interaction"],
                baseline_rows,
                "terminal_utility",
                iterations=args.bootstrap,
                seed=args.seed + 9100,
            )
    interaction = summaries.get("interaction")
    one_step = summaries.get("one_step")
    state_blind = summaries.get("state_blind")
    checks = {}
    if interaction is not None:
        checks.update(
            {
                "interaction_selected_legal": (
                    interaction["selected_legal_rate"] == 1.0
                ),
                "interaction_hard_feasible": (
                    interaction["hard_feasible_rate"] == 1.0
                ),
                "interaction_invalid_mass_zero": (
                    interaction["invalid_probability_mass"] == 0.0
                ),
            }
        )
    if "interaction_vs_one_step" in comparisons:
        delta = comparisons["interaction_vs_one_step"]
        checks["interaction_beats_one_step"] = (
            delta["mean_delta"] > 0.0 and delta["ci_low"] > 0.0
        )
    report = {
        "schema_version": "residential-interaction-policy-v2",
        "status": "passed" if checks and all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "candidate_origin": "real predicted controls with two-step legal recovery tree",
        "test_opened": False,
        "single_trajectory": True,
        "test_best_of_n": False,
        "candidate_budget": 5,
        "training_sequences": len(
            {group["sequence_id"] for group in train_groups}
        ),
        "validation_sequences": len(sequences),
        "arms": summaries,
        "comparisons": comparisons,
        "checkpoints": checkpoints,
        "histories": histories,
        "checks": checks,
        "notes": {
            "one_step": "one-step reranker trained on root states",
            "state_blind": "same architecture and interaction data with graph/context zeroed",
            "interaction": "state-action policy trained on counterfactual legal recovery states",
            "utility_scaling": "per-group z-score for optimization; raw utility for evaluation",
            "interaction_target": "best depth-two descendant utility where available",
            "checkpoint_selection": "validation terminal utility only",
            "adapter_only": args.adapter_only,
            "primary_interaction_endpoint": "terminal utility after the selected legal sequence",
        },
        "redundant_arms": {
            "one_step": one_step is not None,
            "state_blind": state_blind is not None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": (
                    str(args.report.relative_to(ROOT))
                    if args.report.is_absolute()
                    else str(args.report)
                ),
                "checks": checks,
            },
            indent=2,
        )
    )
    if not checks or not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
