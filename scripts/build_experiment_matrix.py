#!/usr/bin/env python3
"""Build the frozen-test experiment matrix for the current floorplan variants."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "flow_v1_256": ROOT / "outputs/eval/full_v1_flow_baseline/metrics.json",
    "direct_v1_256": ROOT / "outputs/eval/full_v1_direct_claim/metrics.json",
    "advanced_v1_384": ROOT / "outputs/eval/full_v1_384_advanced/metrics.json",
    "advanced_v1_384_pathfix": ROOT / "outputs/eval/full_v1_384_advanced_pathfix/metrics.json",
    "condition_v2_256": ROOT / "outputs/eval/condition_room_v2_pathfix_full/metrics.json",
    "flow_v2_256": ROOT / "outputs/eval/flow_full_v2_direct/metrics.json",
    "semantic_hybrid_v2_v1": ROOT / "outputs/eval/semantic_hybrid_pathfix/test.json",
    "structured_v3": ROOT / "outputs/eval/structured_v3/final_test.json",
    "structured_current_503": ROOT / "outputs/eval/current_model_vlm_100_prob/metrics.json",
}
DEVELOPMENT_VARIANTS = {
    "hierarchical_round1_val191": ROOT / "outputs/eval/hierarchical_round1_val191_v2/metrics.json",
    "room_mask_val191": ROOT / "outputs/train/room_mask_hierarchical_1024_5ep_v3/train.jsonl",
    "structured_cfm_step16_val32": ROOT / "outputs/eval/condition_cfm_step16_val32/metrics.json",
}
METRICS = (
    "room_macro_iou_mean",
    "room_area_coverage_mean",
    "opening_f1_mean",
    "opening_wall_alignment_mean",
    "furniture_iou_mean",
    "furniture_room_containment_mean",
    "room_area_distribution_mae_mean",
    "room_component_gap_mean",
    "wall_closure_mean",
)


def main() -> None:
    rows = []
    for name, path in VARIANTS.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = (
            {f"{key}_mean": value for key, value in payload["metrics"].items()}
            if "metrics" in payload
            else payload
        )
        rows.append(
            {
                "variant": name,
                **{metric: source.get(metric) for metric in METRICS},
                "metrics_path": str(path),
            }
        )
    room_set_path = ROOT / "outputs/eval/room_set/test_metrics.json"
    room_set = json.loads(room_set_path.read_text(encoding="utf-8")) if room_set_path.is_file() else {}
    report = {
        "protocol": {
            "test_cases": 503,
            "best_of_n_used": False,
            "selection": "none",
            "primary_sampling": "single deterministic sample or expected statistics over all samples",
            "best_of_n_role": "fixed-budget ceiling diagnostic only; excluded from publication gate",
        },
        "raster_variants": rows,
        "room_set": room_set,
        "data_audit": json.loads((ROOT / "outputs/reports/data_audit.json").read_text(encoding="utf-8")),
        "distribution": json.loads((ROOT / "outputs/eval/distribution_v2/report.json").read_text(encoding="utf-8")),
        "distribution_current": json.loads(
            (ROOT / "outputs/eval/distribution_structured_cfm_16step/report.json").read_text(encoding="utf-8")
        ),
        "publication_gate": json.loads(
            (ROOT / "outputs/reports/publication_gate_v3.json").read_text(encoding="utf-8")
        ),
        "development_variants": {
            name: str(path)
            for name, path in DEVELOPMENT_VARIANTS.items()
            if path.is_file()
        },
        "missing_experiments": [
            "independent human inter-rater agreement",
            "full-resolution vector-loop supervision",
            "attainment of room IoU and interior-wall IoU research targets",
        ],
    }
    output = ROOT / "outputs/reports/experiment_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
