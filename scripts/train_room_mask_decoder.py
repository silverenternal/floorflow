#!/usr/bin/env python3
"""Train a Mask2Former-inspired room instance mask decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dual_room_mask_decoder import DualPathRoomMaskDecoder
from floorflow.heads.room_mask_decoder import (
    RoomMaskDecoder,
    room_mask_loss,
    room_mask_metrics,
    room_mask_partition_loss,
    room_mask_targets,
)
from floorflow.heads.room_partition import RoomPartitionHead
from floorflow.heads.room_program_set import RoomProgramSetHead
from floorflow.inference.retrieval_prior import ContourRoomRetriever, prior_one_hot
from floorflow.objectives.gradient_balance import gradient_diagnostics
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.room_mask_finetune import (
    configure_room_mask_train_scope,
    load_room_mask_decoder_state,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, default=None)
    parser.add_argument("--program-checkpoint", type=Path, default=None)
    parser.add_argument("--partition-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/room_mask_decoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--decoder-mode",
        choices=("shared", "dual"),
        default="dual",
    )
    parser.add_argument(
        "--train-scope",
        choices=("full", "class_only", "classifier_only"),
        default="full",
    )
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--partition-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--assignment-class-weight", type=float, default=0.25)
    parser.add_argument("--assignment-class-weight-after", type=float, default=1.0)
    parser.add_argument("--assignment-mask-weight", type=float, default=2.0)
    parser.add_argument("--mask-anchor-epochs", type=int, default=1)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-loss-weight", type=float, default=2.0)
    parser.add_argument("--program-loss-weight", type=float, default=0.0)
    parser.add_argument("--gradient-audit-steps", type=int, default=50)
    parser.add_argument(
        "--selection-metric",
        choices=("mask_iou", "rendered_room_macro_iou", "composite"),
        default="mask_iou",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--prototype-k", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def features_for(backbone, target, footprint, floor_ids):
    zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=target.device)
    with torch.no_grad():
        return backbone._encode_features(
            zero,
            torch.zeros(target.shape[0], device=target.device),
            footprint,
            floor_ids=floor_ids,
        )


def run_decoder(
    decoder,
    features,
    prototype_logits=None,
    program_tokens=None,
):
    kwargs = {}
    if program_tokens is not None:
        kwargs["program_tokens"] = program_tokens
    if prototype_logits is None:
        return decoder(features, **kwargs)
    return decoder(features, prototype_logits, **kwargs)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    backbone, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    backbone.requires_grad_(False)
    decoder_payload = None
    if args.decoder_checkpoint is not None:
        decoder_payload = torch.load(
            args.decoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_taxonomy = FloorplanTaxonomy.from_json(decoder_payload["taxonomy"])
        if checkpoint_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("decoder and backbone taxonomies differ")
        decoder_config = decoder_payload.get("config", {})
        for key in ("hidden_dim", "queries", "layers", "heads"):
            setattr(args, key, int(decoder_config.get(key, getattr(args, key))))
        args.decoder_mode = str(decoder_config.get("decoder_mode", args.decoder_mode))
    elif args.train_scope in {"class_only", "classifier_only"}:
        raise ValueError(
            f"--train-scope {args.train_scope} requires --decoder-checkpoint"
        )
    program_model = None
    if args.program_checkpoint is not None:
        program_payload = torch.load(
            args.program_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        program_taxonomy = FloorplanTaxonomy.from_json(program_payload["taxonomy"])
        if program_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("program and backbone taxonomies differ")
        program_config = program_payload.get("config", {})
        program_model = RoomProgramSetHead(
            backbone.output.in_channels,
            len(taxonomy.room_ids),
            hidden_dim=int(program_config.get("hidden_dim", 96)),
            tokens=int(program_config.get("tokens", 8)),
            layers=int(program_config.get("layers", 1)),
            heads=int(program_config.get("heads", 4)),
            num_queries=int(program_config.get("queries", 32)),
        ).to(device)
        program_model.load_state_dict(program_payload["head"])
        program_model.eval().requires_grad_(False)
    partition_head = None
    retriever = None
    train_name_to_index = None
    if args.partition_checkpoint is not None:
        partition_payload = torch.load(args.partition_checkpoint, map_location="cpu", weights_only=False)
        partition_taxonomy = FloorplanTaxonomy.from_json(partition_payload["taxonomy"])
        if partition_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("partition and backbone taxonomy mismatch")
        partition_config = partition_payload.get("config", {})
        retrieval_top_k = int(partition_config.get("retrieval_top_k", 0))
        partition_head = RoomPartitionHead(
            backbone.output.in_channels,
            len(taxonomy.room_ids),
            prior_channels=len(taxonomy.room_ids) if retrieval_top_k > 0 else 0,
        ).to(device)
        partition_head.load_state_dict(partition_payload["head"])
        partition_head.eval()
        for parameter in partition_head.parameters():
            parameter.requires_grad_(False)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=args.partition_checkpoint is None,
        limit=args.limit_train,
    )
    if partition_head is not None:
        train.preload()
        if retrieval_top_k > 0:
            retriever = ContourRoomRetriever(
                train._targets,
                train._footprints,
                taxonomy,
                max(retrieval_top_k, args.prototype_k),
            )
            train_name_to_index = {
                f"{sample.relative_case}__{sample.floor_id}": index
                for index, sample in enumerate(train.samples)
            }
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_val,
    )
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, persistent_workers=args.workers > 0, **loader_args)
    val_loader = DataLoader(val, shuffle=False, persistent_workers=args.workers > 0, **loader_args)
    decoder_type = (
        DualPathRoomMaskDecoder
        if args.decoder_mode == "dual"
        else RoomMaskDecoder
    )
    decoder = decoder_type(
        backbone.output.in_channels + (len(taxonomy.room_ids) if partition_head is not None else 0),
        len(taxonomy.room_ids),
        hidden_dim=args.hidden_dim,
        num_queries=args.queries,
        layers=args.layers,
        heads=args.heads,
        **( 
            {"program_token_dim": int(program_model.class_head.in_features)}
            if program_model is not None and args.decoder_mode == "shared"
            else {}
        ),
    ).to(device)
    if program_model is not None and args.decoder_mode != "shared":
        raise ValueError("program-token fusion requires the shared room decoder")
    if decoder_payload is not None:
        load_room_mask_decoder_state(decoder, decoder_payload["head"])
    scope_manifest = configure_room_mask_train_scope(decoder, args.train_scope)
    optimizer = torch.optim.AdamW(
        trainable_parameters(decoder.parameters()),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    room_weights = None
    if args.class_balance:
        positive = torch.from_numpy(
            load_positive_weights(
                args.cache_root,
                taxonomy.num_target_channels,
                20.0,
            )
        ).to(device)
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        room_weights = positive[room_ids - 1]
        room_weights = room_weights / room_weights.mean().clamp_min(1.0e-6)
    best = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "gradient_audit.jsonl"
    for epoch in range(args.epochs):
        assignment_class_weight = (
            args.assignment_class_weight
            if epoch < args.mask_anchor_epochs
            else args.assignment_class_weight_after
        )
        decoder.train()
        running = 0.0
        seen = 0
        for step, (target, footprint, floor_ids, names) in enumerate(train_loader):
            target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
            features = features_for(backbone, target, footprint, floor_ids)
            decoder_features = features
            partition_logits = None
            if partition_head is not None:
                prior = None
                if retriever is not None:
                    prior = prior_one_hot(
                        retriever.retrieve(
                            footprint.cpu().numpy(),
                            exclude_indices=np.asarray([train_name_to_index[name] for name in names]),
                            top_k=retrieval_top_k,
                        ),
                        len(taxonomy.room_ids),
                        device,
                    )
                with torch.no_grad():
                    partition_logits = partition_head(features, prior)
                decoder_features = torch.cat([features, partition_logits], dim=1)
            if retriever is not None and args.prototype_k > 0:
                prototype_partitions, _prototype_weights = retriever.retrieve_partitions(
                    footprint.cpu().numpy(),
                    exclude_indices=np.asarray([train_name_to_index[name] for name in names]),
                    top_k=args.prototype_k,
                )
                prototype_logits = F.one_hot(
                    torch.from_numpy(prototype_partitions).clamp_min(0).long(),
                    len(taxonomy.room_ids),
                )
                prototype_logits = prototype_logits * torch.from_numpy(
                    prototype_partitions >= 0
                ).long().unsqueeze(-1)
                prototype_logits = prototype_logits.permute(0, 1, 4, 2, 3).float().to(device)
                program_tokens = (
                    program_model(features)["class_tokens"]
                    if program_model is not None
                    else None
                )
                predictions = run_decoder(
                    decoder,
                    decoder_features,
                    prototype_logits,
                    program_tokens,
                )
            else:
                program_tokens = (
                    program_model(features)["class_tokens"]
                    if program_model is not None
                    else None
                )
                predictions = run_decoder(
                    decoder,
                    decoder_features,
                    program_tokens=program_tokens,
                )
            if partition_logits is not None:
                predictions["prior_logits"] = partition_logits
            target_classes, target_masks, target_valid = room_mask_targets(target, taxonomy, args.queries)
            loss, _components = room_mask_loss(
                predictions,
                target_classes,
                target_masks,
                target_valid,
                boundary_weight=args.boundary_weight,
                assignment_class_weight=assignment_class_weight,
                assignment_mask_weight=args.assignment_mask_weight,
                class_weights=room_weights,
                class_loss_weight=args.class_loss_weight,
                mask_loss_weight=args.mask_loss_weight,
                program_loss_weight=args.program_loss_weight,
            )
            partition_loss, _partition_components = room_mask_partition_loss(
                predictions,
                target,
                footprint,
                taxonomy,
                class_weights=room_weights,
            )
            loss = loss + args.partition_weight * partition_loss
            if (
                args.gradient_audit_steps > 0
                and step % args.gradient_audit_steps == 0
            ):
                audit_terms = {
                    "class": _components["_room_mask_class_term"],
                }
                if args.train_scope == "full":
                    audit_terms["mask"] = _components["_room_mask_mask_term"]
                audit = gradient_diagnostics(
                    audit_terms,
                    [
                        parameter
                        for parameter in decoder.parameters()
                        if parameter.requires_grad
                    ],
                )
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": step,
                                **audit,
                            }
                        )
                        + "\n"
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        decoder.eval()
        rows = []
        with torch.no_grad():
            for target, footprint, floor_ids, _names in val_loader:
                target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
                features = features_for(backbone, target, footprint, floor_ids)
                partition_logits = None
                if partition_head is not None:
                    prior = None
                    if retriever is not None:
                        prior = prior_one_hot(
                            retriever.retrieve(footprint.cpu().numpy(), top_k=retrieval_top_k),
                            len(taxonomy.room_ids),
                            device,
                        )
                    partition_logits = partition_head(features, prior)
                decoder_features = (
                    torch.cat([features, partition_logits], dim=1)
                    if partition_logits is not None
                    else features
                )
                if retriever is not None and args.prototype_k > 0:
                    prototype_partitions, _prototype_weights = retriever.retrieve_partitions(
                        footprint.cpu().numpy(),
                        top_k=args.prototype_k,
                    )
                    prototype_logits = F.one_hot(
                        torch.from_numpy(prototype_partitions).clamp_min(0).long(),
                        len(taxonomy.room_ids),
                    )
                    prototype_logits = prototype_logits * torch.from_numpy(
                        prototype_partitions >= 0
                    ).long().unsqueeze(-1)
                    prototype_logits = prototype_logits.permute(0, 1, 4, 2, 3).float().to(device)
                    program_tokens = (
                        program_model(features)["class_tokens"]
                        if program_model is not None
                        else None
                    )
                    predictions = run_decoder(
                        decoder,
                        decoder_features,
                        prototype_logits,
                        program_tokens,
                    )
                else:
                    program_tokens = (
                        program_model(features)["class_tokens"]
                        if program_model is not None
                        else None
                    )
                    predictions = run_decoder(
                        decoder,
                        decoder_features,
                        program_tokens=program_tokens,
                    )
                if partition_logits is not None:
                    predictions["prior_logits"] = partition_logits
                target_classes, target_masks, target_valid = room_mask_targets(target, taxonomy, args.queries)
                rows.append(
                    room_mask_metrics(
                        predictions,
                        target_classes,
                        target_masks,
                        target_valid,
                        taxonomy,
                        target,
                        footprint,
                        assignment_class_weight=assignment_class_weight,
                        assignment_mask_weight=args.assignment_mask_weight,
                    )
                )
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "assignment_class_weight": args.assignment_class_weight,
            "effective_assignment_class_weight": assignment_class_weight,
            "assignment_class_weight_after": args.assignment_class_weight_after,
            "mask_anchor_epochs": args.mask_anchor_epochs,
            "assignment_mask_weight": args.assignment_mask_weight,
            "class_loss_weight": args.class_loss_weight,
            "mask_loss_weight": args.mask_loss_weight,
            "program_loss_weight": args.program_loss_weight,
            "class_balance": args.class_balance,
            "train_scope": args.train_scope,
            "trainable_parameters": scope_manifest["trainable_count"],
            "frozen_parameters": scope_manifest["frozen_count"],
            **{key: sum(row[key] for row in rows) / max(1, len(rows)) for key in rows[0]},
        }
        if args.selection_metric == "composite":
            selection_score = (
                record["rendered_room_macro_iou"]
                + record["matched_class_accuracy"]
                + 0.5 * record["mask_iou"]
            )
        else:
            selection_score = record[args.selection_metric]
        record["selection_metric"] = args.selection_metric
        record["selection_score"] = selection_score
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "head": decoder.state_dict(),
            "backbone_checkpoint": str(args.checkpoint),
            "decoder_checkpoint": (
                str(args.decoder_checkpoint) if args.decoder_checkpoint else None
            ),
            "program_checkpoint": (
                str(args.program_checkpoint) if args.program_checkpoint else None
            ),
            "partition_checkpoint": str(args.partition_checkpoint) if args.partition_checkpoint else None,
            "taxonomy": taxonomy.to_json(),
            "config": {**vars(args), "image_size": int(config.get("image_size", 256))},
            "scope_manifest": scope_manifest,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch:03d}.pt")
        if selection_score > best:
            best = selection_score
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
