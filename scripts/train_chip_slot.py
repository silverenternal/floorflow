#!/usr/bin/env python3
"""Train a global-latent chip flow with a feature-conditioned slot decoder."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.chip_flow.batch_prefetch import BufferedBatchStream
from floorflow.chip_flow.branch_health import branch_health
from floorflow.chip_flow.branch_registry import validate_branch_scope
from floorflow.chip_flow.cache import PackedCaseStore, prepare_training_store
from floorflow.chip_flow.checkpoint_policy import resolve_checkpoint_state
from floorflow.chip_flow.chip_checkpoint_io import save_chip_checkpoint
from floorflow.chip_flow.chip_training_step import accumulate_supervised_gradients
from floorflow.chip_flow.complexity_sampler import make_chip_batch_sampler
from floorflow.chip_flow.data_manifest import (
    load_data_manifest,
    verify_data_manifest,
)
from floorflow.chip_flow.eval_curve_manifest import resolve_eval_curve
from floorflow.chip_flow.flow_adapter import build_chip_model
from floorflow.chip_flow.model_config import model_config
from floorflow.chip_flow.response_sidecar import load_response_sidecar
from floorflow.chip_flow.train_cli import (
    json_args,
    parse_args,
    validate_training_scope,
)
from floorflow.chip_flow.train_evaluation import (
    evaluate,
    subset_indices_by_count,
    validate_disjoint_splits,
)
from floorflow.chip_flow.training_manifest import (
    RESUME_CRITICAL_KEYS,
    RESUME_NEW_PARAM_PREFIXES,
    history_config,
    load_train_subset,
    validate_resume_config,
)
from floorflow.chip_flow.training_runtime import (
    clip_optimizer_gradients_fail_closed,
    ema_decay_at_step,
    ema_gap_by_prefix,
    gradient_norm_by_prefix,
    initial_ema_state,
    make_optimizer,
    nonfinite_parameter_names,
    parameter_delta_by_prefix,
    parameter_probe_by_prefix,
    restore_cuda_rng_state,
    set_optimizer_group_lrs,
    unfreeze_backbone_at_step,
    update_ema_state_,
    write_json_atomic,
)
from floorflow.chip_flow.training_scopes import ChannelStageController, attach_typed_channel_cache
from floorflow.chip_flow.trinity_backbone_init import (
    initialize_trinity_backbone,
)

DIAGNOSTIC_INTERVAL = 500


def main() -> None:
    args = parse_args()
    validate_training_scope(args)
    if args.gpu_data_cache_max_gib <= 0.0: raise ValueError("--gpu-data-cache-max-gib must be positive")
    run_config = json_args(args) | model_config(args)
    validate_branch_scope(run_config, protocol=args.architecture_protocol)
    execution_budget = {
        "micro_batch": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch": int(args.batch_size * args.grad_accum_steps),
        "planned_steps": int(args.steps),
        "planned_sample_exposures": int(
            args.batch_size * args.grad_accum_steps * args.steps
        ),
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train = PackedCaseStore(args.cache / args.train_split)
    dev = PackedCaseStore(args.cache / args.val_split)
    validate_disjoint_splits(train, dev)
    data_manifest = None
    if args.data_manifest is not None:
        data_manifest = load_data_manifest(args.data_manifest)
        verify_data_manifest(
            data_manifest,
            args.cache / args.train_split,
            args.cache / args.val_split,
        )
    response_sidecar = (
        load_response_sidecar(args.response_sidecar)
        if args.response_sidecar is not None
        else None
    )
    data_seed = args.seed if args.data_seed is None else args.data_seed
    allowed_by_count = None
    balanced_sampler = None
    if args.train_indices is not None:
        allowed_by_count = subset_indices_by_count(
            train,
            load_train_subset(args.train_indices),
        )
        print(
            json.dumps(
                {
                    "train_subset": str(args.train_indices),
                    "train_cases": sum(len(values) for values in allowed_by_count.values()),
                    "block_count_groups": len(allowed_by_count),
                }
            ),
            flush=True,
        )
    elif args.sampling_mode != "random":
        allowed_by_count = {
            int(count): np.asarray(indices, dtype=np.int64)
            for count, indices in train.by_block_count.items()
        }
    balanced_sampler = make_chip_batch_sampler(
        args.sampling_mode,
        train,
        allowed_by_count,
        batch_size=args.batch_size,
        seed=data_seed,
        total_steps=args.steps,
    )
    train, train_cache_status = prepare_training_store(
        train,
        device,
        enabled=args.gpu_data_cache,
        max_gib=args.gpu_data_cache_max_gib,
    )
    print(json.dumps({"gpu_data_cache": train_cache_status}), flush=True)
    curve, eval_curve_manifest = resolve_eval_curve(
        dev,
        path=args.eval_curve_manifest,
        per_count=args.curve_per_count,
        eval_seed=args.eval_curve_seed,
    )
    model = build_chip_model(model_config(args)).to(device)
    initialization_provenance = None
    if args.trinity_backbone_init is not None:
        if args.resume is not None:
            raise ValueError(
                "Trinity initialization cannot be combined with --resume"
            )
        initialization_provenance = initialize_trinity_backbone(
            model,
            args.trinity_backbone_init,
        )
    if args.freeze_backbone or args.backbone_unfreeze_step is not None:
        for name, parameter in model.named_parameters():
            if name.startswith("backbone."):
                parameter.requires_grad_(False)
    if args.freeze_existing:
        trainable_prefixes = (
            "order_staging.",
            "coarse_to_fine.",
            "port_consensus.",
            "spatial_occupancy.",
            "complexity_slot.",
            "typed.degree_expert.",
            "backbone.hierarchy_bridge.",
            "global_bbox_consensus.",
            "history_to_latent.",
            "structured_heads.",
            "state_conditioned_edge_bias.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
    stage_controller = ChannelStageController.from_args(args)
    stage_controller.apply_initial(model)
    attach_typed_channel_cache(model, args)
    optimizer = make_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        fused=device.type == "cuda",
        backbone_lr_scale=args.backbone_lr_scale,
        structured_head_lr_scale=args.structured_head_lr_scale,
        legalizer_head_lr_scale=args.legalizer_head_lr_scale,
        structured_head_warmup_steps=args.structured_head_warmup_steps,
        legalizer_head_warmup_steps=args.legalizer_head_warmup_steps,
        include_frozen=(
            args.backbone_unfreeze_step is not None
            or stage_controller.requires_all_optimizer_parameters
        ),
    )
    ema = initial_ema_state(model)
    rng = np.random.default_rng(data_seed)
    data_order_hash = hashlib.sha256()
    start_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_config = state.get("config", {})
        validate_resume_config(
            saved_config,
            {key: getattr(args, key, None) for key in RESUME_CRITICAL_KEYS},
            weights_only=bool(args.resume_weights_only),
        )
        incompatible = model.load_state_dict(
            state["model"],
            strict=not args.resume_allow_new_params,
        )
        if args.resume_allow_new_params:
            allowed_missing = {"typed.cross_attention.type_bias"}
            if args.relation_specific_attention:
                allowed_missing.add("typed.cross_attention.relation_gate")
            disallowed_missing = [
                name
                for name in incompatible.missing_keys
                if name not in allowed_missing
                and not name.startswith(RESUME_NEW_PARAM_PREFIXES)
                and not name.endswith(("lora_a", "lora_b", "symmetric_gate"))
            ]
            if disallowed_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "resume-new-param mismatch: "
                    f"missing={sorted(disallowed_missing)} "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )
        if not args.resume_weights_only:
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
            if state.get("data_order_hash") is not None:
                data_order_hash = hashlib.sha256(
                    bytes.fromhex(state["data_order_hash"])
                )
            if balanced_sampler is not None:
                sampler_state = state.get("balanced_sampler_state")
                if sampler_state is None:
                    raise ValueError(
                        "balanced_stream resume checkpoint lacks sampler state"
                    )
                balanced_sampler.load_state_dict(sampler_state)
        else:
            ema = initial_ema_state(model)
    eval_model = copy.deepcopy(model)

    def evaluation_model(state: dict[str, torch.Tensor] | None = None) -> torch.nn.Module:
        if state is None and args.eval_state == "raw":
            return model
        eval_model.load_state_dict(ema if state is None else state)
        return eval_model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{args.arm}.json"
    checkpoint_dir = args.out_dir / args.arm / "checkpoints"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"
    shortlist_candidates = []
    history = []
    diagnostics = []
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        history = previous.get("history", [])
        diagnostics = previous.get("diagnostics", [])
        previous_eval_state = previous.get("config", {}).get("eval_state", "raw")
        if history and previous_eval_state != args.eval_state:
            raise ValueError(
                "cannot append checkpoint history across different eval states: "
                f"{previous_eval_state} != {args.eval_state}"
            )
        if history and history_config(previous.get("config", {})) != history_config(
            run_config
        ):
            raise ValueError("cannot append history from a different training config")
    best_eval_total = min(
        (float(row["total"]) for row in history),
        default=float("inf"),
    )
    eval_steps = set(args.eval_steps)
    started = time.perf_counter()
    for step in range(start_step, args.steps + 1):
        unfreeze_backbone_at_step(model, args.backbone_unfreeze_step, step)
        stage_controller.update(model, step)
        if step in eval_steps:
            metrics = evaluate(
                dev,
                curve,
                evaluation_model(),
                device=device,
                batch_size=args.eval_batch_size,
                sampling_steps=args.sampling_steps,
                seed=args.seed,
            )
            row = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "curve_cases": len(curve),
                **metrics,
            }
            history.append(row)
            if args.checkpoint_shortlist_size > 0 and (
                len(shortlist_candidates) < args.checkpoint_shortlist_size
                or float(metrics["total"])
                < max(
                    float(candidate["raw_total"])
                    for candidate in shortlist_candidates
                )
            ):
                candidate_path = checkpoint_dir / f"shortlist_step_{step}.pt"
                save_chip_checkpoint(
                    candidate_path,
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    batch_rng=rng,
                    balanced_sampler=balanced_sampler,
                    data_order_hash=data_order_hash.hexdigest(),
                    args=args,
                )
                shortlist_candidates.append(
                    {
                        "step": int(step),
                        "raw_total": float(metrics["total"]),
                        "checkpoint": str(candidate_path.resolve()),
                    }
                )
                shortlist_candidates.sort(key=lambda item: item["raw_total"])
                for removed in shortlist_candidates[
                    args.checkpoint_shortlist_size :
                ]:
                    Path(removed["checkpoint"]).unlink(missing_ok=True)
                del shortlist_candidates[args.checkpoint_shortlist_size :]
            if float(metrics["total"]) < best_eval_total:
                best_eval_total = float(metrics["total"])
                save_chip_checkpoint(
                    best_checkpoint_path,
                    step=step,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    batch_rng=rng,
                    balanced_sampler=balanced_sampler,
                    data_order_hash=data_order_hash.hexdigest(),
                    args=args,
                )
            print(json.dumps(row), flush=True)
            write_json_atomic(
                report_path,
                {
                    "status": "running" if step < args.steps else "complete",
                    "arm": args.arm,
                    "config": run_config,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "train_cases": len(train),
                    "dev_cases": len(dev),
                    "execution_budget": execution_budget,
                    "gpu_data_cache": train_cache_status,
                    "initialization_provenance": initialization_provenance,
                    "data_manifest": data_manifest,
                    "eval_curve_manifest": eval_curve_manifest,
                    "history": history,
                    "diagnostics": diagnostics,
                },
            )
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        if hasattr(balanced_sampler, "set_step"):
            balanced_sampler.set_step(step)
        with BufferedBatchStream(
            train,
            rng,
            batch_size=args.batch_size,
            device=device,
            batches=args.grad_accum_steps,
            geometry_augment=args.geometry_augment,
            geometry_augment_probability=args.geometry_augment_probability,
            allowed_by_count=allowed_by_count,
            sampler=balanced_sampler,
            enabled=args.batch_prefetch,
        ) as batch_stream:
            loss, parts = accumulate_supervised_gradients(
                model,
                batch_stream,
                args,
                device,
                step=step,
                data_order_hash=data_order_hash,
                response_sidecar=response_sidecar,
            )
        diagnostic = step % DIAGNOSTIC_INTERVAL == 0
        probe = parameter_probe_by_prefix(model) if diagnostic else None
        gate_health = branch_health(model) if diagnostic else None
        gradient_by_prefix = gradient_norm_by_prefix(model) if diagnostic else {}
        clip_stats = clip_optimizer_gradients_fail_closed(
            model,
            optimizer,
            args.grad_clip,
            per_group=args.grad_clip_per_group,
            materialize=diagnostic,
        )
        grad_norm = float(clip_stats["global_pre_clip_norm"]) if diagnostic else None
        lr = stage_controller.learning_rate(step, args)
        group_lrs = set_optimizer_group_lrs(optimizer, lr, step)
        optimizer.step()
        invalid_parameters = nonfinite_parameter_names(model)
        if invalid_parameters:
            raise FloatingPointError(
                "non-finite parameters after optimizer step: "
                f"{invalid_parameters[:8]}"
            )
        ema_decay = ema_decay_at_step(args, step)
        update_ema_state_(ema, model.state_dict(), ema_decay)
        if diagnostic:
            diagnostics.append(
                {
                    "step": step,
                    "data_order_hash": data_order_hash.hexdigest(),
                    "total_grad_norm": grad_norm,
                    "gradient_norm_by_prefix": gradient_by_prefix,
                    "branch_health": gate_health,
                    "training_scope": stage_controller.current.name,
                    "gradient_clipping": clip_stats,
                    "group_lrs": group_lrs,
                    "ema_decay": ema_decay,
                    "parameter_update_norm_by_prefix": parameter_delta_by_prefix(
                        model,
                        probe,
                    ),
                    "ema_gap_by_prefix": ema_gap_by_prefix(model, ema),
                }
            )
        if (step + 1) % args.checkpoint_every == 0:
            save_chip_checkpoint(
                args.out_dir / args.arm / "checkpoints" / "last.pt",
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                batch_rng=rng,
                balanced_sampler=balanced_sampler,
                data_order_hash=data_order_hash.hexdigest(),
                args=args,
            )
        if step % 50 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        **{
                            name: float(value.detach())
                            for name, value in parts.items()
                        },
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "gradient_clipping": clip_stats if diagnostic else None,
                        "ema_decay": ema_decay,
                        "data_order_hash_prefix": data_order_hash.hexdigest()[:16],
                    }
                ),
                flush=True,
            )

    final_indices = dev.indices() if args.full_dev_eval else curve
    final_state = ema if args.eval_state == "ema" else model.state_dict()
    (
        selected_state,
        selected_step,
        selected_checkpoint,
        selected_policy,
    ) = resolve_checkpoint_state(
        args.checkpoint_policy,
        final_state=final_state,
        final_step=args.steps,
        final_path=final_checkpoint_path,
        best_path=best_checkpoint_path,
        state_key=args.eval_state,
        selection_report=args.checkpoint_selection_report,
    )
    final = evaluate(
        dev,
        final_indices,
        evaluation_model(selected_state),
        device=device,
        batch_size=args.eval_batch_size,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
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
            "config": run_config,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "train_cases": len(train),
            "dev_cases": len(dev),
            "execution_budget": execution_budget,
                "gpu_data_cache": train_cache_status,
            "initialization_provenance": initialization_provenance,
            "data_manifest": data_manifest,
            "eval_curve_manifest": eval_curve_manifest,
            "history": history,
            "diagnostics": diagnostics,
                    "channel_training": stage_controller.history,
        }
    )
    best_eval = min(
        history,
        key=lambda row: (row["total"], row["step"]),
        default={"step": args.steps, "total": final["total"]},
    )
    report.update(
        {
            "status": "complete",
            "final_dev": final,
            "steps_to_threshold": first_threshold,
            "thresholds": {
                "overlap": args.overlap_threshold,
                "total": args.total_threshold,
            },
            "diagnostics": diagnostics,
            "channel_training": {
                "final_scope": stage_controller.current.name,
                "history": stage_controller.history,
            },
            "checkpoint_selection": {
                "final_policy": selected_policy,
                "selected_step": int(selected_step),
                "selected_checkpoint": str(selected_checkpoint.resolve()),
                "ema_mode": args.ema_mode,
                "ema_base_decay": float(args.ema_decay),
                "final_dev_state": args.eval_state,
                "evaluation_checkpoint_state": (
                    "ema_via_use_ema"
                    if args.eval_state == "ema"
                    else "raw_model"
                ),
                "best_selection_state": args.eval_state,
                "best_checkpoint": str(
                    best_checkpoint_path.resolve()
                ),
                "best_eval_step": int(best_eval["step"]),
                "best_eval_total": float(best_eval["total"]),
                "shortlist_candidates": shortlist_candidates,
            },
        }
    )
    write_json_atomic(report_path, report)
    save_chip_checkpoint(
        final_checkpoint_path,
        step=args.steps,
        model=model,
        ema=ema,
        optimizer=optimizer,
        batch_rng=rng,
        balanced_sampler=balanced_sampler,
        data_order_hash=data_order_hash.hexdigest(),
        args=args,
    )
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
