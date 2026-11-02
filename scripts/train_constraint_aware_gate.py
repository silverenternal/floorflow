"""Train CAGN on hard-masked candidate groups."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from floorflow.evaluation.constraint_calibration import (
    fit_isotonic_calibration,
)
from floorflow.evaluation.constraint_gate_runtime import (
    combined_scores as _combined_scores,
)
from floorflow.evaluation.constraint_gate_runtime import (
    fallback_rows as _fallback_rows,
)
from floorflow.evaluation.constraint_gate_runtime import (
    gate_calibration_metrics as _calibration,
)
from floorflow.evaluation.constraint_gate_runtime import gate_metrics as _metrics
from floorflow.model.constraint_aware_gate import ConstraintAwareGate
from floorflow.objectives.constraint_contrastive import (
    multi_label_violation_loss,
)
from floorflow.objectives.constraint_gate_loss import (
    gate_calibration_loss,
    hard_masked_listwise_loss,
    hard_masked_pairwise_loss,
)
from floorflow.objectives.soft_graph_edit import soft_graph_edit_distance

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_KEYS = (
    "node_features",
    "adjacency",
    "global_features",
    "constraint_diagnostics",
    "node_mask",
    "operation_ids",
    "action_features",
    "hard_mask",
    "legal",
    "violation_labels",
    "candidate_hash",
    "group_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--soft-ged-weight", type=float, default=0.05)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/residential_cagn_features_v1",
    )
    parser.add_argument(
        "--embedding-checkpoint",
        type=Path,
        default=ROOT / "outputs/residential_cagn_embedding_v1/seed19.pt",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "outputs/residential_cagn_gate_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/residential_cagn_validation_v1.json",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in payload.items()
    }


def _groups(features: dict) -> list[torch.Tensor]:
    indices = defaultdict(list)
    for index, group in enumerate(features["group_index"].tolist()):
        indices[int(group)].append(index)
    return [
        torch.tensor(items, device=features["group_index"].device)
        for items in indices.values()
    ]


def _gather_groups(features: dict, groups: list[torch.Tensor]) -> dict:
    counts = {len(group) for group in groups}
    if len(counts) != 1:
        raise ValueError(f"candidate groups have unequal sizes: {sorted(counts)}")
    count = counts.pop()
    indices = torch.stack(groups)
    batch = {
        key: features[key][indices]
        for key in SAMPLE_KEYS
    }
    batch["family_ids"] = features["family_ids"]
    batch["candidates"] = count
    return batch


def _batches(
    features: dict,
    batch_size: int,
    generator: torch.Generator | None = None,
):
    groups = _groups(features)
    order = torch.randperm(len(groups), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        selected = [groups[index] for index in order[start : start + batch_size]]
        yield _gather_groups(features, selected)


def _split_groups(features: dict, keep: torch.Tensor) -> dict:
    row_mask = torch.isin(features["group_index"], keep)
    return {
        key: value[row_mask] if key in SAMPLE_KEYS else value
        for key, value in features.items()
    }


def _flatten(batch: dict) -> tuple[torch.Tensor, ...]:
    groups, candidates = batch["node_features"].shape[:2]
    family = batch["family_ids"][None].expand(groups * candidates, -1)
    return (
        batch["node_features"].flatten(0, 1),
        batch["adjacency"].flatten(0, 1),
        family,
        batch["operation_ids"].flatten(),
        batch["action_features"].flatten(0, 1),
        batch["hard_mask"].flatten(0, 1),
        batch["node_mask"].flatten(0, 1),
        torch.cat(
            [
                batch["global_features"].flatten(0, 1),
                batch["constraint_diagnostics"].flatten(0, 1),
            ],
            dim=-1,
        ),
    )


def _forward(model: ConstraintAwareGate, batch: dict) -> dict:
    output = model(*_flatten(batch))
    groups, candidates = batch["legal"].shape
    for key in ("gate_logits", "gate_probability", "uncertainty", "hard_feasible"):
        output[key] = output[key].reshape(groups, candidates)
    output["legal_logits"] = output["legal_logits"].reshape(
        groups,
        candidates,
    )
    output["violation_logits"] = output["violation_logits"].reshape(
        groups,
        candidates,
        -1,
    )
    nodes = batch["node_features"].shape[2]
    output["node_tokens"] = output["node_tokens"].reshape(
        groups,
        candidates,
        nodes,
        -1,
    )
    return output


def _soft_ged_loss(output: dict, batch: dict) -> torch.Tensor:
    groups, candidates, nodes, _ = output["node_tokens"].shape
    legal = batch["legal"].bool()
    target_index = legal.float().argmax(dim=-1)
    batch_index = torch.arange(groups, device=legal.device)
    target_tokens = output["node_tokens"][batch_index, target_index].detach()
    target_tokens = target_tokens[:, None].expand(
        groups,
        candidates,
        nodes,
        target_tokens.shape[-1],
    )
    candidate_tokens = output["node_tokens"].flatten(0, 1)
    predicted_edges = torch.bmm(
        candidate_tokens,
        candidate_tokens.transpose(1, 2),
    ) / (candidate_tokens.shape[-1] ** 0.5)
    predicted_edges = predicted_edges.sigmoid()
    target_edges = batch["adjacency"][batch_index, target_index]
    target_edges = target_edges[:, None].expand_as(batch["adjacency"]).flatten(0, 1)
    classes = torch.eye(nodes, device=legal.device).expand(
        groups * candidates,
        nodes,
        nodes,
    )
    valid = batch["node_mask"].flatten(0, 1)
    loss, _terms = soft_graph_edit_distance(
        output["node_tokens"].flatten(0, 1)[..., :4],
        target_tokens.flatten(0, 1)[..., :4],
        classes,
        classes,
        predicted_edges,
        target_edges,
        predicted_valid=valid,
        target_valid=valid,
    )
    return loss


def _train_epoch(
    model: ConstraintAwareGate,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    features: dict,
    args: argparse.Namespace,
    epoch: int,
) -> float:
    model.train()
    generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
    total = 0.0
    batches = 0
    use_amp = features["legal"].device.type == "cuda"
    for batch in _batches(features, args.batch_size, generator):
        with torch.autocast(
            device_type=features["legal"].device.type,
            enabled=use_amp,
        ):
            output = _forward(model, batch)
            positive = batch["legal"].bool()
            feasible = batch["hard_mask"].bool().all(dim=-1)
            listwise = hard_masked_listwise_loss(
                output["gate_logits"],
                positive,
                feasible,
            )
            pairwise = hard_masked_pairwise_loss(
                output["gate_logits"],
                positive,
                feasible,
            )
            calibration = gate_calibration_loss(
                output["gate_logits"],
                positive.float(),
                feasible,
            )
            legal_aux = torch.nn.functional.binary_cross_entropy_with_logits(
                output["legal_logits"][feasible],
                positive.float()[feasible],
            )
            violation_aux = multi_label_violation_loss(
                output["violation_logits"][feasible],
                batch["violation_labels"][feasible],
            )
            soft_ged = _soft_ged_loss(output, batch)
            loss = (
                listwise
                + 0.25 * pairwise
                + 0.1 * calibration
                + 0.5 * legal_aux
                + 0.25 * violation_aux
                + args.soft_ged_weight * soft_ged
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def _predict(model: ConstraintAwareGate, features: dict, batch_size: int) -> dict:
    model.eval()
    rows = []
    for batch in _batches(features, batch_size):
        output = _forward(model, batch)
        for index in range(batch["legal"].shape[0]):
            invalid = ~output["hard_feasible"][index]
            rows.append(
                {
                    "scores": output["gate_logits"][index].cpu().numpy(),
                    "legal_logits": output["legal_logits"][index].cpu().numpy(),
                    "hard": batch["hard_mask"][index].bool().all(dim=-1).cpu().numpy(),
                    "legal": batch["legal"][index].cpu().numpy(),
                    "labels": batch["violation_labels"][index].cpu().numpy(),
                    "hashes": batch["candidate_hash"][index].cpu().numpy(),
                    "operation": int(batch["operation_ids"][index, 0]),
                    "group_index": int(batch["group_index"][index, 0]),
                    "invalid_probability_exact_zero": bool(
                        torch.equal(
                            output["gate_probability"][index][invalid],
                            torch.zeros_like(
                                output["gate_probability"][index][invalid]
                            ),
                        )
                    ),
                }
            )
    return {"groups": rows}


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device)
    train = _load(args.feature_root / "train/features.pt", device)
    validation = _load(args.feature_root / "val/features.pt", device)
    model = ConstraintAwareGate().to(device)
    if args.embedding_checkpoint.exists():
        embedding = torch.load(
            args.embedding_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.state.load_state_dict(embedding["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    calibration_mask = validation["group_index"] % 2 == 0
    calibration = _split_groups(
        validation,
        validation["group_index"][calibration_mask],
    )
    evaluation = _split_groups(
        validation,
        validation["group_index"][~calibration_mask],
    )
    checkpoint_path = args.checkpoint_root / f"seed{args.seed}.pt"
    best = {"ndcg": -1.0, "epoch": 0, "state": None}
    history = []
    if args.evaluate_only:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        best = {
            "ndcg": -1.0,
            "epoch": int(checkpoint["epoch"]),
            "state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }
    else:
        for epoch in range(1, args.epochs + 1):
            loss = _train_epoch(
                model,
                optimizer,
                scaler,
                train,
                args,
                epoch,
            )
            scheduler.step()
            calibration_groups = _predict(
                model,
                calibration,
                args.batch_size,
            )["groups"]
            metrics = _metrics(calibration_groups, learned=True)
            history.append({"epoch": epoch, "loss": loss, **metrics})
            print(
                f"epoch={epoch} loss={loss:.4f} "
                f"ndcg={metrics['ndcg_at_5']:.4f}",
                flush=True,
            )
            if metrics["ndcg_at_5"] > best["ndcg"]:
                best = {
                    "ndcg": metrics["ndcg_at_5"],
                    "epoch": epoch,
                    "state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                }
    model.load_state_dict(best["state"])
    calibration_groups = _predict(
        model,
        calibration,
        args.batch_size,
    )["groups"]
    alpha = 0.0
    legal_threshold = 0.0
    best_selection = None
    for candidate_alpha in (0.0, 0.25, 0.5, 1.0, 2.0):
        for candidate_threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
            candidate_rows = _fallback_rows(
                calibration_groups,
                alpha=candidate_alpha,
                legal_threshold=candidate_threshold,
            )
            candidate_metrics = _metrics(candidate_rows, learned=True)
            if candidate_metrics["selected_violation_rate"] > 0.005:
                continue
            selection = candidate_metrics["ndcg_at_5"]
            if best_selection is None or selection > best_selection:
                best_selection = selection
                alpha = candidate_alpha
                legal_threshold = candidate_threshold
    calibration_probability = []
    calibration_target = []
    for group in calibration_groups:
        feasible = group["hard"]
        scores = _combined_scores(group, alpha)[feasible]
        calibration_probability.extend(
            torch.sigmoid(torch.tensor(scores)).tolist()
        )
        calibration_target.extend(
            group["legal"][feasible].astype(float).tolist()
        )
    calibration_parameters = fit_isotonic_calibration(
        np.asarray(calibration_probability),
        np.asarray(calibration_target, dtype=bool),
    )
    evaluation_groups = _predict(
        model,
        evaluation,
        args.batch_size,
    )["groups"]
    adjusted_evaluation = _fallback_rows(
        evaluation_groups,
        alpha=alpha,
        legal_threshold=legal_threshold,
    )
    learned_metrics = _metrics(adjusted_evaluation, learned=True)
    hard_metrics = _metrics(evaluation_groups, learned=False)
    calibration_metrics = _calibration(
        evaluation_groups,
        alpha=alpha,
        calibration_parameters=calibration_parameters,
    )
    checks = {
        "ndcg_gain_at_least_0_03": (
            learned_metrics["ndcg_at_5"] - hard_metrics["ndcg_at_5"] >= 0.03
        ),
        "selected_violation_rate_at_most_0_005": (
            learned_metrics["selected_violation_rate"] <= 0.005
        ),
        "hard_mask_exact": all(
            group["invalid_probability_exact_zero"]
            for group in evaluation_groups
        ),
        "calibration_ece_at_most_0_05": calibration_metrics["ece"] <= 0.05,
    }
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "node_features": 8,
                "action_features": 15,
                "hard_mask_features": 8,
                "hidden_dim": 384,
                "embedding_dim": 192,
                "operations": 5,
                "families": 5,
            },
            "seed": args.seed,
            "epoch": best["epoch"],
        },
        checkpoint_path,
    )
    report = {
        "schema_version": "residential-cagn-validation-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "selection_policy": "train-only fit; even/odd val491 group split",
        "best_epoch": best["epoch"],
        "score_ensemble_weight": alpha,
        "legal_fallback_threshold": legal_threshold,
        "isotonic_calibration_bins": len(calibration_parameters["x"]),
        "hard_gate": hard_metrics,
        "learned_gate": learned_metrics,
        "gain": {
            key: learned_metrics[key] - hard_metrics[key]
            for key in ("ndcg_at_5", "selected_legal_rate", "selected_violation_rate")
        },
        "calibration": calibration_metrics,
        "checks": checks,
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(ROOT))
        if checkpoint_path.is_relative_to(ROOT)
        else str(checkpoint_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.report), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
