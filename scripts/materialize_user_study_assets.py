#!/usr/bin/env python3
"""Materialize blinded user-study assets without collecting participant data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.visualize import colorize
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.representation.core import contour_descriptor


def _descriptor(footprint: torch.Tensor) -> np.ndarray:
    value = contour_descriptor(footprint.numpy().astype(bool))
    return value / max(float(np.linalg.norm(value)), 1.0e-8)


def _crop_resize(image: Image.Image, footprint: torch.Tensor) -> Image.Image:
    mask = footprint.cpu().numpy().astype(bool)
    rows, columns = np.where(mask)
    if rows.size == 0:
        return image.resize((512, 512), Image.Resampling.NEAREST)
    margin = 8
    top = max(0, int(rows.min()) - margin)
    left = max(0, int(columns.min()) - margin)
    bottom = min(mask.shape[0], int(rows.max()) + margin + 1)
    right = min(mask.shape[1], int(columns.max()) + margin + 1)
    return image.crop((left, top, right, bottom)).resize(
        (512, 512),
        Image.Resampling.NEAREST,
    )


def _save_plan(
    path: Path,
    labels: torch.Tensor,
    taxonomy,
    footprint: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = colorize(labels.cpu().numpy(), taxonomy)
    _crop_resize(Image.fromarray(image), footprint).save(path)


def _save_contour(path: Path, footprint: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (footprint.cpu().numpy().astype(bool) * 255).astype(np.uint8)
    _crop_resize(
        Image.fromarray(array, mode="L").convert("RGB"),
        footprint,
    ).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("outputs/reports/user_study_case_pack_v2.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/distribution_style_prompt_r94_v2/best.pt"),
    )
    parser.add_argument(
        "--style-bank",
        type=Path,
        default=Path("outputs/style_banks/cubicasa_train_v1.json"),
    )
    parser.add_argument(
        "--retrieval-weights",
        type=Path,
        default=Path("outputs/reports/a1_retrieval_weights_calib191.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/user_study_pilot_v2"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    bank = json.loads(args.style_bank.read_text(encoding="utf-8"))
    model, taxonomy, _ = restore_distribution_model(args.checkpoint, device)
    test = CubiCasaFloorplanDataset(
        ".data/cubicasa5k",
        pack["split"],
        image_size=256,
        cache_root="outputs/cache/cubicasa5k_full_v2",
        augment=False,
    )
    train = CubiCasaFloorplanDataset(
        ".data/cubicasa5k",
        "train",
        image_size=256,
        cache_root="outputs/cache/cubicasa5k_full_v2",
        augment=False,
    )
    test_index = {str(test[index][3]): index for index in range(len(test))}
    train_names = [str(train[index][3]) for index in range(len(train))]
    train_descriptors = np.stack(
        [
            _descriptor(train[index][1])
            for index in range(len(train))
        ]
    )
    assignments = bank["assignments"]["train"]
    retrieval_weights = torch.tensor(
        json.loads(
            args.retrieval_weights.read_text(encoding="utf-8")
        )["weights"],
        device=device,
    )
    memory = ContourPartitionMemory.from_cache(
        "outputs/cache/cubicasa5k_full_v2",
        top_k=1,
    )
    settings = DistributionOverlaySettings(opening_element_weight=0.0)
    survey_cases = []
    answer_key = []
    for case in pack["case_order"]:
        dataset_index = test_index[case["case_id"]]
        _target, footprint, floor_id, _name = test[dataset_index]
        contour = footprint[None, None].to(device)
        case_slug = case["case_id"].replace("/", "__")
        case_root = args.output_root / "assets" / case_slug
        _save_contour(case_root / "contour.png", footprint)
        style_id = int(case["requested_style_id"])
        style_ids = torch.tensor([style_id], device=device)
        conditioned = decode_overlay(
            model,
            memory,
            retrieval_weights,
            contour,
            taxonomy,
            floor_ids=torch.tensor([floor_id], device=device),
            style_ids=style_ids,
            retrieval_prior_weight=0.5,
            settings=settings,
        )["labels"][0]
        neutral = decode_overlay(
            model,
            memory,
            retrieval_weights,
            contour,
            taxonomy,
            floor_ids=torch.tensor([floor_id], device=device),
            style_ids=None,
            retrieval_prior_weight=0.5,
            settings=settings,
        )["labels"][0]
        style_indices = np.asarray(
            [
                index
                for index, name in enumerate(train_names)
                if int(assignments[name]) == style_id
            ],
            dtype=np.int64,
        )
        scores = train_descriptors[style_indices] @ _descriptor(footprint)
        order = np.argsort(-scores)
        retrieved_index = int(style_indices[order[0]])
        retrieved = train[retrieved_index][0]
        retrieval = torch.as_tensor(
            retrieved.numpy() * footprint.numpy().astype(np.uint8),
            device=device,
        )
        condition_labels = {
            "style_conditioned": conditioned,
            "no_style": neutral,
            "retrieval": retrieval,
        }
        plan_entries = []
        mapping = {}
        for slot, condition in zip(("A", "B", "C"), case["condition_order"]):
            relative = Path(case_slug) / f"{slot}.png"
            _save_plan(
                args.output_root / "assets" / relative,
                condition_labels[condition],
                taxonomy,
                footprint,
            )
            plan_entries.append({"slot": slot, "image": f"assets/{relative}"})
            mapping[slot] = condition
        reference_entries = []
        for reference_rank in range(1, 5):
            reference_index = int(style_indices[order[reference_rank]])
            reference_target, reference_footprint, _, _ = train[reference_index]
            relative = Path(case_slug) / f"style_ref_{reference_rank}.png"
            labels = torch.as_tensor(
                reference_target.numpy() * reference_footprint.numpy().astype(np.uint8),
                device=device,
            )
            _save_plan(
                args.output_root / "assets" / relative,
                labels,
                taxonomy,
                reference_footprint,
            )
            reference_entries.append(f"assets/{relative}")
        survey_cases.append(
            {
                "case_id": case["case_id"],
                "style_prompt": "Requested style: match the reference examples",
                "contour": f"assets/{case_slug}/contour.png",
                "style_references": reference_entries,
                "plans": plan_entries,
            }
        )
        answer_key.append(
            {
                "case_id": case["case_id"],
                "requested_style_id": style_id,
                "slot_conditions": mapping,
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "survey_data.json").write_text(
        json.dumps(
            {
                "version": "user_study_pilot_v2",
                "ethics_status": "PENDING",
                "cases": survey_cases,
                "notice": "Do not deploy until ethics review is READY.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_root / "answer_key.json").write_text(
        json.dumps({"cases": answer_key}, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        "docs/user_study_survey_template.html",
        args.output_root / "survey.html",
    )
    print(
        json.dumps(
            {
                "cases": len(survey_cases),
                "assets": len(list((args.output_root / "assets").rglob("*.png"))),
                "ethics_status": "PENDING",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
