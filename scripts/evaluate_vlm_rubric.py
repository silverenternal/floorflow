#!/usr/bin/env python3
"""Score generated floorplans with a local Qwen3-VL model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL = (
    "/home/hugo/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
PROMPT = """You are an architectural drawing evaluator.
Inspect this generated floorplan image and return ONLY JSON:
{
  "room_plausibility": 1-5,
  "topology_consistency": 1-5,
  "furniture_room_consistency": 1-5,
  "drawing_readability": 1-5,
  "failure_reasons": ["short reason"]
}
Score 5 only when rooms form a coherent partition, doors/windows plausibly attach to walls,
and furniture is contained in compatible rooms. Do not reward decorative color.
"""


def parse_json(text: str) -> dict:
    text = text.split("assistant")[-1].strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"parse_error": True, "raw": text}
    try:
        payload = json.loads(match.group(0))
        payload["parse_error"] = False
        payload["raw"] = text
        return payload
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="directory or glob-like image directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/vlm_rubric/report.json"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    paths = sorted(args.images.glob("*.png"))[: args.limit]
    if not paths:
        raise SystemExit(f"no PNG files under {args.images}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        response = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]
        payload = parse_json(response)
        payload["image"] = str(path)
        rows.append(payload)
        print(json.dumps({"image": path.name, "parsed": not payload.get("parse_error", False)}), flush=True)
    valid = [row for row in rows if not row.get("parse_error", False)]
    fields = ("room_plausibility", "topology_consistency", "furniture_room_consistency", "drawing_readability")
    summary = {
        "model": args.model,
        "samples": len(rows),
        "parsed": len(valid),
        "best_of_n_used": False,
        "sampling_policy": {
            "purpose": "single_sample",
            "candidates_per_condition": 1,
            "selection_rule": "none",
            "target_used_for_selection": False,
        },
        **{
            f"{field}_mean": sum(float(row.get(field, 0.0)) for row in valid) / max(1, len(valid))
            for field in fields
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
