#!/usr/bin/env python3
"""Run one frozen HouseDiffusion sampling pass on packaged CubiCasa programs."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cairosvg
import drawsvg
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

HOUSEDIFFUSION_ROOT = Path("third_party/house-diffusion").resolve()
sys.path.insert(0, str(HOUSEDIFFUSION_ROOT))
sys.path.insert(0, str(HOUSEDIFFUSION_ROOT / "scripts"))
sys.modules.setdefault("drawSvg", drawsvg)

from house_diffusion.rplanhg_datasets import RPlanhgDataset
from house_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    update_arg_parser,
)

ID_COLOR = {
    1: "#EE4D4D",
    2: "#C67C7B",
    3: "#FFD274",
    4: "#BEBEBE",
    5: "#BFE3E8",
    6: "#7BA779",
    7: "#E87A90",
    8: "#FF8C69",
    10: "#1F849B",
    11: "#727171",
    13: "#785A67",
    12: "#D3A2C7",
}


def render_predictions(
    sample: torch.Tensor,
    model_kwargs: dict[str, torch.Tensor],
    start_index: int,
    output_dir: Path,
) -> None:
    sample = sample[-1:]
    for batch_index in range(sample.shape[1]):
        drawing = drawsvg.Drawing(256, 256, displayInline=False)
        drawing.append(drawsvg.Rectangle(0, 0, 256, 256, fill="black"))
        polygons = []
        types = []
        polygon = []
        room_type = 0
        for token_index, point in enumerate(sample[0][batch_index]):
            if model_kwargs["src_key_padding_mask"][batch_index][token_index] == 1:
                continue
            point = point.cpu().numpy()
            if token_index == 0:
                polygon = []
            if (
                token_index > 0
                and (
                    model_kwargs["room_indices"][batch_index, token_index]
                    != model_kwargs["room_indices"][batch_index, token_index - 1]
                ).any()
            ):
                polygons.append(polygon)
                types.append(room_type)
                polygon = []
            point = (point / 2 + 0.5) * 256
            polygon.append((point[0], point[1]))
            room_type = int(
                model_kwargs["room_types"][batch_index][token_index - 1].argmax()
            )
        if polygon:
            polygons.append(polygon)
            types.append(room_type)
        for polygon, room_type in zip(polygons, types):
            if room_type not in ID_COLOR or room_type in {11, 12, 13}:
                continue
            drawing.append(
                drawsvg.Lines(
                    *np.asarray(polygon).flatten().tolist(),
                    close=True,
                    fill=ID_COLOR[room_type],
                    fill_opacity=1.0,
                    stroke="black",
                    stroke_width=1.0,
                )
            )
        png = cairosvg.svg2png(
            bytestring=drawing.as_svg().encode("utf-8"),
            output_width=256,
            output_height=256,
        )
        Image.open(io.BytesIO(png)).save(
            output_dir / f"{start_index + batch_index}c_pred.png"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/external_baselines/house_diffusion/house_diffusion_model.pt"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(
            "outputs/external_baselines/cubicasa_oracle/housediffusion/processed"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/external_baselines/cubicasa_oracle/housediffusion/manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=58)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOUSEDIFF_PROCESSED_DIR"] = str(args.processed_dir.resolve())
    previous_cwd = Path.cwd()
    os.chdir(args.output_dir)
    try:
        for name in ("pred", "gt", "gif", "graphs_gt", "graphs_pred"):
            Path("outputs", name).mkdir(parents=True, exist_ok=True)
        model_args = SimpleNamespace(
            dataset="rplan",
            analog_bit=False,
            clip_denoised=True,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            use_ddim=False,
            model_path=str((previous_cwd / args.model_path).resolve()),
            draw_graph=False,
            save_svg=False,
            timestep_respacing=str(args.timesteps),
        )
        defaults = dict(model_and_diffusion_defaults())
        defaults.update(vars(model_args))
        model_args = SimpleNamespace(**defaults)
        update_arg_parser(model_args)
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(model_args, model_and_diffusion_defaults().keys())
        )
        state = torch.load(
            previous_cwd / args.model_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state)
        device = torch.device(args.device)
        model = model.to(device).eval()
        dataset = RPlanhgDataset(
            set_name="test",
            analog_bit=False,
            target_set=8,
        )
        for index, graph in enumerate(dataset.graphs):
            if np.asarray(graph).size == 0:
                dataset.graphs[index] = np.zeros((1, 3))
        data = iter(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
        )
        generated = 0
        cases = []
        while generated < args.num_samples:
            try:
                data_sample, model_kwargs = next(data)
            except StopIteration:
                break
            model_kwargs = {
                key: value.to(device) for key, value in model_kwargs.items()
            }
            model_kwargs.update(
                {
                    f"syn_{key}": value
                    for key, value in list(model_kwargs.items())
                }
            )
            final = None
            for final in diffusion.p_sample_loop_progressive(
                model,
                data_sample.shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                analog_bit=False,
            ):
                pass
            if final is None:
                raise RuntimeError("HouseDiffusion sampling produced no output")
            sample = final["sample"].unsqueeze(0)
            sample_gt = data_sample.to(device).unsqueeze(0)
            sample = sample.permute([0, 1, 3, 2])
            sample_gt = sample_gt.permute([0, 1, 3, 2])
            render_predictions(
                sample,
                model_kwargs,
                generated,
                Path("outputs/pred"),
            )
            batch_count = int(sample_gt.shape[1])
            cases.extend(range(generated, min(generated + batch_count, args.num_samples)))
            generated += batch_count
        source_manifest = json.loads(
            (previous_cwd / args.manifest).read_text(encoding="utf-8")
        )
        report = {
            "status": "complete",
            "method": "HouseDiffusion",
            "condition_mode": "oracle_room_program",
            "target_derived_control": True,
            "best_of_n_used": False,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "requested_samples": args.num_samples,
            "generated_samples": generated,
            "cases": cases,
            "source_manifest": str((previous_cwd / args.manifest).resolve()),
            "source_records": source_manifest.get("records", [])[:generated],
            "output_dir": str((previous_cwd / args.output_dir).resolve()),
        }
        report_path = Path("sampling_manifest.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
