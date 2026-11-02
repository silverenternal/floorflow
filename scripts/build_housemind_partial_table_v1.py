"""Build the HouseMind partial-reproduction and reference-ceiling table."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    partial = json.loads(
        (
            ROOT / "outputs/reports/housemind_partial_reproduction_v1.json"
        ).read_text(encoding="utf-8")
    )
    feasibility = json.loads(
        (
            ROOT / "outputs/reports/housemind_inference_feasibility_v1.json"
        ).read_text(encoding="utf-8")
    )
    identity = json.loads(
        (
            ROOT / "outputs/reports/residential_identity_test509_v4_summary.json"
        ).read_text(encoding="utf-8")
    )
    published = {
        "delta_iou": 0.608,
        "delta_mse": 0.0019,
        "node_f1": 0.998,
        "edge_overlap": 0.934,
        "ged": 0.467,
    }
    report = {
        "status": "partial_reproduction",
        "scope": "HouseMind-E released benchmark",
        "rows": {
            "HouseMind-E published": {
                "role": "reported_reference_ceiling",
                **published,
                "matched_inference_reproduced": False,
            },
            "HouseMind public weights": {
                "role": "partial_reproduction_only",
                "model_loaded": feasibility["checks"]["model_weight_load_passed"],
                "cases_generated": partial["cases"],
                "room_tokens_generated": partial["observed"]["room_tokens"],
                "valid_image_outputs": 0,
                "metrics_available": False,
            },
            "FloorFlow strict-locality edit": {
                "role": "different protocol; not a matched rank",
                "room_f1": identity["test_metrics"]["room_f1"],
                "room_iou": identity["test_metrics"]["room_iou"],
                "delta_iou_range": [0.956703, 0.999801],
                "delta_mse_range": [0.000019, 0.000096],
            },
        },
        "checks": {
            "official_inference_chain_recovered": False,
            "image_outputs_produced": False,
            "metrics_assigned_to_partial_model": False,
            "published_values_labeled_ceiling": True,
        },
        "boundary": (
            "The published HouseMind-E row is reference-ceiling context, not a "
            "mathematical upper bound and not a direct comparison. The public "
            "weight probe exercises language-token generation only."
        ),
    }
    output = ROOT / "outputs/reports/housemind_partial_table_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = ROOT / "paper/tables/housemind_partial_reproduction_v1.md"
    table.write_text(
        "\n".join(
            [
                "# HouseMind Partial Reproduction",
                "",
                "| Arm | Delta-IoU | Delta-MSE | Node F1 | Edge overlap | GED | Role |",
                "|---|---:|---:|---:|---:|---:|---|",
                (
                    "| HouseMind-E published | 0.608 | 0.0019 | 0.998 | "
                    "0.934 | 0.467 | reported reference ceiling |"
                ),
                (
                    f"| HouseMind public weights | N/A | N/A | N/A | N/A | N/A | "
                    f"partial: {partial['cases']} token-generation prompts, "
                    f"{partial['observed']['room_tokens']} room tokens |"
                ),
                (
                    "| FloorFlow strict-locality edit | 0.9567-0.9998 | "
                    "0.000019-0.000096 | aligned protocol | aligned protocol | "
                    "aligned protocol | different task/protocol |"
                ),
                "",
                (
                    "Official image tokenization, sequence serialization, "
                    "constrained decoding, VQ-VAE decoding, and post-processing "
                    "remain unavailable. No direct HouseMind rank is claimed."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
