#!/usr/bin/env python3
"""Emit a machine-readable audit of the production target-free generation API."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.target_free import forbidden_parameters

ROOT = Path(__file__).resolve().parents[1]
DECODER_PATH = ROOT / "src" / "floorflow" / "inference" / "overlay_decoder.py"
EVALUATOR_PATH = ROOT / "scripts" / "evaluate_distribution_overlay.py"
OUTPUT_PATH = ROOT / "outputs" / "reports" / "target_free_contract_v1.json"


def _call_arguments(path: Path, callable_name: str) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else None
        if isinstance(function, ast.Attribute):
            name = function.attr
        if name == callable_name:
            calls.append({keyword.arg for keyword in node.keywords if keyword.arg})
    return calls


def main() -> None:
    forbidden = forbidden_parameters(decode_overlay)
    evaluator_calls = _call_arguments(EVALUATOR_PATH, "decode_overlay")
    leaked_keywords = sorted(
        {
            name
            for arguments in evaluator_calls
            for name in arguments
            if "target" in name.lower() or name.lower() in {"gt", "oracle"}
        }
    )
    source = DECODER_PATH.read_bytes()
    report = {
        "status": "pass" if not forbidden and not leaked_keywords else "fail",
        "generation_entrypoint": f"{decode_overlay.__module__}.{decode_overlay.__name__}",
        "generation_signature": str(inspect.signature(decode_overlay)),
        "decorated_target_free": bool(
            getattr(decode_overlay, "target_free", False)
        ),
        "forbidden_parameters": list(forbidden),
        "allowed_information": [
            "footprint",
            "floor/config id",
            "explicit train-bank style id",
            "model outputs derived from the above",
        ],
        "evaluator_uses_entrypoint": bool(evaluator_calls),
        "evaluator_target_keywords_passed_to_generation": leaked_keywords,
        "decoder_sha256": hashlib.sha256(source).hexdigest(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
