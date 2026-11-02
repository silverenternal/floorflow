#!/usr/bin/env python3
"""Train one long-horizon chip-flow arm on the compact FloorSet cache."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.batched import (
    batch_physics_terms,
    flow_loss,
    matched_endpoint_loss,
)
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.model import ChipBlockFlowModel, decode_boxes_batch
from floorflow.chip_flow.proposals import token_ot_reorder
from floorflow.chip_flow.train_cli import json_args
from floorflow.chip_flow.train_evaluation import (
    select_curve_indices,
    validate_disjoint_splits,
)
from floorflow.chip_flow.training_runtime import (
    cuda_rng_state,
    restore_cuda_rng_state,
    save_torch_atomic,
    write_json_atomic,
)
from floorflow.chip_physics import group_penalty
from floorflow.objectives.token_optimal_transport import batch_token_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="dev")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000])
    parser.add_argument("--curve-per-count", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--overlap-weight-scale", type=float, default=1.0)
    parser.add_argument("--ot-endpoint-weight", type=float, default=0.1)
    parser.add_argument("--time-std", type=float, default=1.0)
    parser.add_argument("--source-noise", type=float, default=0.05)
    parser.add_argument("--flow-huber-beta", type=float, default=0.0)
    parser.add_argument("--prior", choices=("shelf", "gaussian"), default="shelf")
    parser.add_argument("--minibatch-ot", action="store_true")
    parser.add_argument("--token-ot", action="store_true")
    parser.add_argument("--sparse-k", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--full-dev-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initialize-model", type=Path, default=None)
    parser.add_argument("--allow-full-data", action="store_true")
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--aux-cache", type=Path, default=None)
    parser.add_argument("--aux-train-split", default="train")
    parser.add_argument("--aux-val-split", default="val")
    parser.add_argument("--aux-batch-size", type=int, default=32)
    parser.add_argument("--aux-curve-per-count", type=int, default=2)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--overlap-threshold", type=float, default=0.05)
    parser.add_argument("--total-threshold", type=float, default=0.030)
    return parser.parse_args()


def model_config(args: argparse.Namespace) -> dict:
    return {
        "feature_dim": 10,
        "hidden": args.hidden,
        "layers": args.layers,
        "heads": args.heads,
        "top_k": args.sparse_k or None,
    }


@torch.no_grad()
def evaluate(
    store: PackedCaseStore,
    indices: np.ndarray,
    model: ChipBlockFlowModel,
    *,
    device: torch.device,
    batch_size: int,
    sampling_steps: int,
    prior: str = "shelf",
    seed: int = 0,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, float] = {}
    cases = 0
    for count in sorted(set(store.arrays["n"][indices].tolist())):
        count_indices = indices[store.arrays["n"][indices] == count]
        for start in range(0, len(count_indices), batch_size):
            batch = store.batch(count_indices[start : start + batch_size], device=device)
            if prior == "gaussian":
                latent = torch.randn(
                    batch["source_latent"].shape,
                    device=device,
                    generator=generator,
                )
            else:
                latent = batch["source_latent"].clone()
            for step in range(sampling_steps):
                time = torch.full(
                    (latent.shape[0],),
                    step / sampling_steps,
                    device=device,
                )
                latent = latent + model(
                    latent,
                    time,
                    batch["features"],
                ) / sampling_steps
            boxes = decode_boxes_batch(latent, batch["areas"])
            terms = batch_physics_terms(
                boxes,
                batch["areas"],
                batch["constraints"],
                batch["b2b"],
                batch["p2b"],
                batch["pins"],
                batch["n_b2b"],
                batch["n_p2b"],
                batch["n_pins"],
            )
            group = torch.stack(
                [
                    group_penalty(
                        boxes[index],
                        batch["constraints"][index, :, 3],
                        batch["areas"][index],
                    )
                    for index in range(latent.shape[0])
                ]
            )
            aligned = []
            for index in range(latent.shape[0]):
                aligned.append(
                    token_ot_reorder(
                        latent[index].detach().cpu(),
                        batch["target_latent"][index].detach().cpu(),
                    )
                )
            aligned_target = torch.stack(aligned).to(device)
            latent_mse = (latent - aligned_target).pow(2).mean(dim=(1, 2))
            values = {
                "latent_mse": latent_mse,
                "overlap": terms["overlap"],
                "group": group,
                "mib": terms["mib"],
                "boundary": terms["boundary"],
                "wl": terms["wl"],
                "area": terms["area"],
            }
            values["total"] = 0.01 * sum(values[name] for name in (
                "overlap",
                "group",
                "mib",
                "boundary",
                "wl",
                "area",
            ))
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.sum())
            cases += latent.shape[0]
    if was_training:
        model.train()
    return {name: value / cases for name, value in totals.items()}


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def sample_domain_batch(
    store: PackedCaseStore,
    rng: np.random.Generator,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    counts = np.asarray(sorted(store.by_block_count), dtype=np.int64)
    count = int(rng.choice(counts))
    candidates = store.indices(count)
    indices = rng.choice(
        candidates,
        size=batch_size,
        replace=len(candidates) < batch_size,
    )
    return store.batch(indices, device=device)


def supervised_flow_loss(
    model: ChipBlockFlowModel,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    size = batch["source_latent"].shape[0]
    time_value = torch.sigmoid(args.time_std * torch.randn(size, device=device))
    target = batch["target_latent"]
    if args.prior == "gaussian":
        source = torch.randn_like(target)
        if args.minibatch_ot:
            order = batch_token_assignment(source.flatten(1), target.flatten(1))
            for name in (
                "areas",
                "boxes",
                "constraints",
                "features",
                "b2b",
                "p2b",
                "pins",
                "n_b2b",
                "n_p2b",
                "n_pins",
                "target_latent",
            ):
                batch[name] = batch[name][order]
            target = batch["target_latent"]
    else:
        source = batch["source_latent"] + args.source_noise * torch.randn_like(
            batch["source_latent"]
        )
    interpolant = (
        (1.0 - time_value[:, None, None]) * source
        + time_value[:, None, None] * target
    )
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        velocity = model(interpolant, time_value, batch["features"])
        flow, estimate = flow_loss(
            velocity,
            interpolant,
            source,
            target,
            time_value,
            huber_beta=args.flow_huber_beta,
        )
    boxes = decode_boxes_batch(estimate.float(), batch["areas"])
    terms = batch_physics_terms(
        boxes,
        batch["areas"],
        batch["constraints"],
        batch["b2b"],
        batch["p2b"],
        batch["pins"],
        batch["n_b2b"],
        batch["n_p2b"],
        batch["n_pins"],
    )
    matched = (
        matched_endpoint_loss(estimate.float(), target)
        if args.token_ot
        else torch.zeros((), device=device)
    )
    physics = terms["total"].mean()
    physics = physics + (
        float(args.overlap_weight_scale) - 1.0
    ) * 0.01 * terms["overlap"].mean()
    loss = flow + args.physics_weight * physics + args.ot_endpoint_weight * matched
    return loss, {"flow": flow, "physics": terms["total"].mean(), "matched": matched}


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: ChipBlockFlowModel,
    ema: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    batch_rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    save_torch_atomic(
        path,
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "batch_rng_state": batch_rng.bit_generator.state,
            "cuda_rng_state": cuda_rng_state(args.device),
            "config": vars(args),
        },
    )


def main() -> None:
    args = parse_args()
    if not args.allow_full_data:
        raise ValueError("train_chip_large is full-data only; pass --allow-full-data")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train = PackedCaseStore(args.cache / args.train_split)
    dev = PackedCaseStore(args.cache / args.val_split)
    validate_disjoint_splits(train, dev)
    aux_train = (
        PackedCaseStore(args.aux_cache / args.aux_train_split)
        if args.aux_cache is not None
        else None
    )
    aux_dev = (
        PackedCaseStore(args.aux_cache / args.aux_val_split)
        if args.aux_cache is not None
        else None
    )
    curve = select_curve_indices(
        dev,
        per_count=args.curve_per_count,
        seed=args.seed,
    )
    model = ChipBlockFlowModel(**model_config(args)).to(device)
    for index, layer in enumerate(model.layers):
        if index >= args.freeze_layers:
            break
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    ema = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    rng = np.random.default_rng(args.seed)
    start_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema = {
            name: value.detach().to(device)
            for name, value in state["ema"].items()
        }
        start_step = int(state["step"]) + 1
        torch.set_rng_state(state["torch_rng"].cpu())
        restore_cuda_rng_state(state.get("cuda_rng_state"), args.device)
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if "batch_rng_state" in state:
            rng.bit_generator.state = state["batch_rng_state"]
    elif args.initialize_model is not None:
        initial = torch.load(
            args.initialize_model,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(initial["model"])
        ema = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{args.arm}.json"
    history = []
    if report_path.is_file():
        history = json.loads(report_path.read_text(encoding="utf-8")).get("history", [])
    eval_steps = set(args.eval_steps)
    started = time.perf_counter()
    for step in range(start_step, args.steps + 1):
        if step in eval_steps:
            metrics = evaluate(
                dev,
                curve,
                model,
                device=device,
                batch_size=args.eval_batch_size,
                sampling_steps=args.sampling_steps,
                prior=args.prior,
                seed=args.seed,
            )
            row = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "curve_cases": len(curve),
                **metrics,
            }
            if aux_dev is not None:
                aux_metrics = evaluate(
                    aux_dev,
                    select_curve_indices(
                        aux_dev,
                        per_count=args.aux_curve_per_count,
                        seed=args.seed,
                    ),
                    model,
                    device=device,
                    batch_size=args.eval_batch_size,
                    sampling_steps=args.sampling_steps,
                    prior=args.prior,
                    seed=args.seed + 1,
                )
                row.update({f"aux_{key}": value for key, value in aux_metrics.items()})
            history.append(row)
            print(json.dumps(row), flush=True)
            write_json_atomic(
                report_path,
                {
                    "status": "running" if step < args.steps else "complete",
                    "arm": args.arm,
                    "config": json_args(args),
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "train_cases": len(train),
                    "dev_cases": len(dev),
                    "history": history,
                },
            )
        if step == args.steps:
            break
        batch = sample_domain_batch(
            train,
            rng,
            batch_size=args.batch_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, parts = supervised_flow_loss(model, batch, args, device)
        aux_parts = None
        if aux_train is not None:
            aux_batch = sample_domain_batch(
                aux_train,
                rng,
                batch_size=args.aux_batch_size,
                device=device,
            )
            aux_loss, aux_parts = supervised_flow_loss(
                model,
                aux_batch,
                args,
                device,
            )
            loss = loss + args.aux_weight * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        with torch.no_grad():
            state = model.state_dict()
            for name, value in ema.items():
                if torch.is_floating_point(value):
                    value.lerp_(state[name].detach(), 1.0 - args.ema_decay)
                else:
                    value.copy_(state[name])
        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                args.out_dir / args.arm / "checkpoints" / "last.pt",
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                batch_rng=rng,
                args=args,
            )
        if step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "flow": float(parts["flow"].detach()),
                        "physics": float(parts["physics"].detach()),
                        "matched_endpoint": float(parts["matched"].detach()),
                        **(
                            {
                                f"aux_{name}": float(value.detach())
                                for name, value in aux_parts.items()
                            }
                            if aux_parts is not None
                            else {}
                        ),
                        "lr": lr,
                    }
                ),
                flush=True,
            )

    final_indices = dev.indices() if args.full_dev_eval else curve
    final = evaluate(
        dev,
        final_indices,
        model,
        device=device,
        batch_size=args.eval_batch_size,
        sampling_steps=args.sampling_steps,
        prior=args.prior,
        seed=args.seed,
    )
    final_aux = None
    if aux_dev is not None:
        final_aux = evaluate(
            aux_dev,
            aux_dev.indices(),
            model,
            device=device,
            batch_size=args.eval_batch_size,
            sampling_steps=args.sampling_steps,
            prior=args.prior,
            seed=args.seed + 1,
        )
    first_threshold = next(
        (
            row["step"]
            for row in history
            if row["overlap"] <= args.overlap_threshold
            and row["total"] <= args.total_threshold
        ),
        None,
    )
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {
            "arm": args.arm,
            "config": json_args(args),
            "history": history,
        }
    )
    report.update(
        {
            "status": "complete",
            "final_dev": final,
            "final_metrics_state": "raw_model",
            **({"final_aux": final_aux} if final_aux is not None else {}),
            "steps_to_threshold": first_threshold,
            "thresholds": {
                "overlap": args.overlap_threshold,
                "total": args.total_threshold,
            },
            "physics_group_in_training": False,
            "physics_group_reason": "reference MST value is non-differentiable",
        }
    )
    write_json_atomic(report_path, report)
    save_checkpoint(
        args.out_dir / args.arm / "checkpoints" / "final.pt",
        step=args.steps,
        model=model,
        ema=ema,
        optimizer=optimizer,
        batch_rng=rng,
        args=args,
    )
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
