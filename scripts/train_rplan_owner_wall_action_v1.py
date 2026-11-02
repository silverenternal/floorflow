"""Train a wall-stage-conditioned scorer over merged owner wall candidates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer
from floorflow.inference.owner_wall_action import (
    build_owner_wall_action_batch,
)
from floorflow.model.cad_action_graph import STAGES
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)
from floorflow.training.cad_wall_candidate_training import candidate_targets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / ".data/external/CE2EPlan/dataset/dataset_json/data_val.json"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-cases", type=int, default=128)
    parser.add_argument("--validation-cases", type=int, default=64)
    parser.add_argument("--validation-offset", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-walls", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--match-tolerance", type=float, default=0.08)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2905)
    return parser.parse_args()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _prepare_batches(
    base: UnifiedFloorplanGenerator,
    loader: DataLoader,
    taxonomy: FloorplanTaxonomy,
    limits: RPlanTensorLimits,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    prepared = []
    with torch.no_grad():
        for raw in loader:
            batch = _device_batch(raw, device)
            base_output = base(
                batch["boundary_mask"],
                batch["boundary_valid"],
                batch["boundary_points"],
                batch["entrance_points"],
                batch["entrance_valid"],
            )
            candidates = build_owner_wall_action_batch(
                base_output,
                batch["boundary_points"],
                taxonomy,
                canvas_size=limits.canvas_size,
                max_walls=int(args.max_walls),
            )
            tokens = torch.stack(
                [
                    base_output["action_tokens"][name]
                    for name in STAGES
                ],
                dim=1,
            )
            prepared.append(
                {
                    "boundary_mask": batch["boundary_mask"].float(),
                    "endpoints": candidates.endpoints,
                    "valid": candidates.valid,
                    "evidence": candidates.evidence,
                    "exterior": candidates.exterior,
                    "action_tokens": tokens,
                    "wall_endpoints": batch["wall_endpoints"],
                    "wall_valid": batch["wall_valid"],
                    "wall_exterior": batch["wall_exterior"],
                }
            )
    return prepared


def _train_epoch(
    scorer: CadWallCandidateScorer,
    graph_projection: torch.nn.Linear,
    batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    scorer.train(optimizer is not None)
    totals = {}
    cases = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            graph_tokens = graph_projection(batch["action_tokens"])
            outputs = scorer(
                batch["boundary_mask"],
                batch["endpoints"],
                batch["valid"],
                batch["evidence"],
                graph_tokens,
                batch["exterior"],
            )
            (
                expected,
                coordinate,
                _budget,
                _exterior_budget,
                _interior_budget,
                exterior_target,
            ) = candidate_targets(
                outputs,
                {
                    "wall_endpoints": batch["wall_endpoints"],
                    "wall_valid": batch["wall_valid"],
                    "wall_exterior": batch["wall_exterior"],
                    "wall_exists": batch["wall_valid"].float(),
                },
                batch["valid"],
                float(args.match_tolerance),
                match_endpoints=batch["endpoints"],
            )
            positive = expected.sum().clamp_min(1.0)
            negative = expected.numel() - positive
            score_elementwise = F.binary_cross_entropy_with_logits(
                outputs["score_logits"],
                expected,
                pos_weight=(negative / positive).sqrt().clamp(1.0, 20.0),
                reduction="none",
            )
            score_loss = (
                score_elementwise[batch["valid"]].mean()
                if bool(batch["valid"].any())
                else score_elementwise.sum() * 0.0
            )
            matched = expected.bool()
            type_loss = (
                F.binary_cross_entropy_with_logits(
                    outputs["exterior_logits"][matched],
                    exterior_target[matched],
                )
                if bool(matched.any())
                else outputs["exterior_logits"].sum() * 0.0
            )
            loss = score_loss + 2.0 * coordinate + 0.5 * type_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in scorer.parameters()
                ):
                    raise RuntimeError("non-finite wall action gradient")
                optimizer.step()
            size = int(batch["boundary_mask"].shape[0])
            cases += size
            values = {
                "loss": loss,
                "score_loss": score_loss,
                "coordinate_loss": coordinate,
                "type_loss": type_loss,
                "valid_candidates": batch["valid"].float().mean(),
            }
            for key, value in values.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach()) * size
                )
    return {
        **{key: value / max(1, cases) for key, value in totals.items()},
        "cases": float(cases),
    }


def main() -> None:
    args = _args()
    _seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("owner wall action training is restricted to cuda:0")
    payload = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    base = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**payload["config"]["model"]),
    ).to(device)
    base.load_state_dict(payload["model"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    limits = RPlanTensorLimits(
        **{
            key: int(value)
            for key, value in payload["config"]["tensors"].items()
        }
    )
    train_rows = load_rplan_rows(
        args.data,
        offset=0,
        limit=int(args.train_cases),
    )
    val_rows = load_rplan_rows(
        args.data,
        offset=int(args.validation_offset),
        limit=int(args.validation_cases),
    )
    train_set = RPlanBoundaryDataset(
        args.data,
        taxonomy,
        rows=train_rows,
        limits=limits,
    )
    val_set = RPlanBoundaryDataset(
        args.data,
        taxonomy,
        rows=val_rows,
        limits=limits,
    )
    common = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.workers),
        "collate_fn": collate_rplan_items,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, **common)
    val_loader = DataLoader(val_set, **common)
    train_batches = _prepare_batches(
        base,
        train_loader,
        taxonomy,
        limits,
        args,
        device,
    )
    val_batches = _prepare_batches(
        base,
        val_loader,
        taxonomy,
        limits,
        args,
        device,
    )
    scorer = CadWallCandidateScorer(
        in_channels=1,
        hidden_dim=int(args.hidden_dim),
        graph_dim=int(args.hidden_dim),
        residual_scale=0.01,
        use_encoded_score=True,
    ).to(device)
    graph_projection = torch.nn.Linear(
        int(args.hidden_dim),
        int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*scorer.parameters(), *graph_projection.parameters()],
        lr=float(args.learning_rate),
        weight_decay=1.0e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    for epoch in range(int(args.epochs)):
        train_metrics = _train_epoch(
            scorer,
            graph_projection,
            train_batches,
            args,
            optimizer,
        )
        val_metrics = _train_epoch(
            scorer,
            graph_projection,
            val_batches,
            args,
            None,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "base_checkpoint": str(args.base_checkpoint),
            "scorer": scorer.state_dict(),
            "graph_projection": graph_projection.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "tensors": payload["config"]["tensors"],
            "config": {
                "hidden_dim": int(args.hidden_dim),
                "max_walls": int(args.max_walls),
                "match_tolerance": float(args.match_tolerance),
                "use_encoded_score": True,
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        score = val_metrics["loss"]
        if best is None or score < best:
            best = score
            torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "test_opened": False,
                "seed": int(args.seed),
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in scorer.parameters()
                ),
                "history": history,
                "best_validation_loss": best,
                "claim_boundary": (
                    "Internal owner-edge action mechanism screen; no test or "
                    "external SOTA claim."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
