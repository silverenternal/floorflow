"""Load public HouseMind weights and verify the custom spatial vocabulary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / ".data/housemind/weights/housemind-omni"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        trust_remote_code=True,
    )
    outline_ids = [
        tokenizer.convert_tokens_to_ids(f"<outline_{index}>")
        for index in range(256)
    ]
    room_ids = [
        tokenizer.convert_tokens_to_ids(f"<room_{index}>")
        for index in range(256)
    ]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()
    messages = [
        {
            "role": "user",
            "content": "Delete the Kitchen from the floor plan.",
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    suffix = generated[0, encoded.input_ids.shape[1] :].tolist()
    report = {
        "status": "loaded_and_token_probe_passed",
        "scope": "HouseMind official public weight and tokenizer probe",
        "model_path": str(MODEL_ROOT.relative_to(ROOT)),
        "model_architecture": model.config.architectures,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(next(model.parameters()).device),
        "tokenizer": {
            "vocab_size": int(tokenizer.vocab_size),
            "outline_tokens": int(sum(value >= 0 for value in outline_ids)),
            "room_tokens": int(sum(value >= 0 for value in room_ids)),
            "outline_id_range": [min(outline_ids), max(outline_ids)],
            "room_id_range": [min(room_ids), max(room_ids)],
        },
        "generation_probe": {
            "prompt": messages[0]["content"],
            "generated_ids": suffix,
            "decoded": tokenizer.decode(suffix, skip_special_tokens=False),
        },
        "checks": {
            "model_loaded": True,
            "tokenizer_loaded": True,
            "outline_vocabulary_256": len(set(outline_ids)) == 256,
            "room_vocabulary_256": len(set(room_ids)) == 256,
            "only_gpu_0_used": torch.cuda.current_device() == 0,
        },
        "boundary": (
            "Loading weights and emitting spatial-token vocabulary does not "
            "recover the official image tokenizer, prompt assembly, generation "
            "stopping rule, or post-processing contract."
        ),
    }
    output = ROOT / "outputs/reports/housemind_weight_probe_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(report["checks"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
