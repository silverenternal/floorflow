"""Run the public HouseMind language model without claiming full inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / ".data/housemind/weights/housemind-omni"
DATA = ROOT / ".data/housemind/data_test/editing.jsonl"
OUTPUT = ROOT / "outputs/reports/housemind_partial_reproduction_v1.json"


def token_kind(token_id: int, tokenizer: AutoTokenizer) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token.startswith("<room_"):
        return "room"
    if token.startswith("<outline_"):
        return "outline"
    if token in {"<room_start>", "<room_end>"}:
        return "room_control"
    if token in {"<outline_start>", "<outline_end>"}:
        return "outline_control"
    return "text_or_other"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()
    prompts = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()[:16]
    ]
    rows = []
    started = time.perf_counter()
    for record in prompts:
        messages = [{"role": "user", "content": record["input"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        case_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        token_ids = generated[0, encoded.input_ids.shape[1] :].tolist()
        kinds = [token_kind(token_id, tokenizer) for token_id in token_ids]
        rows.append(
            {
                "house_id": record["house_id"],
                "instruction": record["input"],
                "generated_tokens": len(token_ids),
                "room_tokens": kinds.count("room"),
                "outline_tokens": kinds.count("outline"),
                "room_start": kinds.count("room_control"),
                "decoded_prefix": tokenizer.decode(
                    token_ids[:32],
                    skip_special_tokens=False,
                ),
                "elapsed_seconds": time.perf_counter() - case_started,
            }
        )
    report = {
        "status": "partial_reproduction_complete_full_inference_blocked",
        "scope": "HouseMind Omni token-generation probe only",
        "model_path": str(MODEL_ROOT.relative_to(ROOT)),
        "device": str(next(model.parameters()).device),
        "cases": len(rows),
        "max_new_tokens": 128,
        "decoding": "greedy",
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "observed": {
            "room_tokens": sum(row["room_tokens"] for row in rows),
            "outline_tokens": sum(row["outline_tokens"] for row in rows),
            "nonempty_outputs": sum(row["generated_tokens"] > 0 for row in rows),
        },
        "not_recovered": [
            "official image tokenizer forward",
            "official interleaved prompt and edit-sequence template",
            "official constrained decoding",
            "official stopping rule",
            "official VQ-VAE decoder and post-processing",
        ],
        "table_role": "reference ceiling only; not a matched HouseMind-E score",
        "boundary": (
            "These are token-generation traces from public weights. They are "
            "not editable floorplans and must never be assigned Delta-IoU, "
            "Delta-MSE, Node F1, Edge overlap, or GED as HouseMind-E results."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": report["cases"],
                "elapsed_seconds": report["elapsed_seconds"],
                "observed": report["observed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
