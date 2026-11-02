"""Consolidate HouseMind public-asset and inference-feasibility evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    assets = json.loads(
        (
            ROOT / "outputs/reports/housemind_released_asset_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    probe = json.loads(
        (
            ROOT / "outputs/reports/housemind_weight_probe_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = {
        "status": "blocked",
        "scope": "matched HouseMind-E inference feasibility",
        "assets": {
            "benchmark_status": assets["status"],
            "benchmark_sha256": assets["benchmark"]["sha256"],
            "weights_sha256": assets["weights"]["sha256"],
            "cases": assets["released_assets"]["joint_test_cases"],
            "layers": assets["released_assets"]["layers"],
        },
        "weight_probe": probe,
        "missing_contract": [
            "official image-to-outline spatial-token encoding",
            "official room-mask extraction and conditional room encoding",
            "official interleaved prompt/token template",
            "official generation stopping and constrained decoding",
            "official VQ-VAE decoder invocation",
            "official three-stage color and structural post-processing",
        ],
        "checks": {
            "official_code_released": False,
            "benchmark_available": True,
            "weights_available": True,
            "model_weight_load_passed": all(probe["checks"].values()),
            "official_inference_reconstructable": False,
            "test_outputs_scored": False,
        },
        "decision": (
            "HouseMind-E complete-method superiority remains blocked. Use its "
            "published Table 3 values only as non-paired benchmark context."
        ),
        "unblock_condition": (
            "Release or independently verify the image tokenizer forward, exact "
            "prompt/token serialization, constrained decoding, and decoder "
            "post-processing, then reproduce validation behavior before test."
        ),
    }
    output = (
        ROOT / "outputs/reports/housemind_inference_feasibility_v1.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    document = ROOT / "docs/housemind_matched_comparison_boundary_v1.md"
    document.write_text(
        "# HouseMind Matched Comparison Boundary\n\nHouseMind benchmark data and all three public weight packages are available and hashed. The Omni Qwen3-0.6B checkpoint loads on GPU 0 and the tokenizer exposes 256 outline and 256 room codes.\n\nOfficial code remains marked `Code (Soon)`. The missing image tokenizer contract, prompt/token assembly, stopping rules, and post-processing prevent a faithful HouseMind-E rerun.\n\nConsequently, no complete-method defeat or paired benchmark claim is permitted. Published Table 3 remains context only."
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
