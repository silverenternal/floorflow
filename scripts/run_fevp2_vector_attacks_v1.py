#!/usr/bin/env python3
"""Run sensitivity attacks against the FEVP-2 vector evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.typed_edit_metrics import normalized_region_mask
from floorflow.evaluation.vector_edit_attacks import (
    perturb_unaffected_opening,
    perturb_unaffected_room_adjacency,
    perturb_unaffected_room_vertex,
    perturb_unaffected_wall_mask,
)
from floorflow.evaluation.vector_edit_protocol import (
    extract_vector_plan,
    vector_plan_preservation_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs/reports/vector_edit_benchmark_v1.json"
TAXONOMY = ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
OUTPUT = ROOT / "outputs/reports/fevp2_vector_attacks_v1.json"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _case_for(rows: list[dict], key: str) -> dict:
    for row in rows:
        if int(row["denominators"].get(key, 0)) > 0:
            return row
    raise ValueError(f"no benchmark case has {key}")


def _metrics(
    row: dict,
    taxonomy: FloorplanTaxonomy,
) -> tuple[dict, dict, dict, np.ndarray]:
    before = _load(ROOT / row["source_paths"]["before"])
    after = _load(ROOT / row["source_paths"]["after"])
    region = normalized_region_mask(
        np.asarray(before["target"]).shape,
        row["edit_region_normalized"],
    )
    changed = np.asarray(before["target"]) != np.asarray(after["target"])
    before_plan = extract_vector_plan(
        before["target"],
        before["room_labels"],
        taxonomy,
    )
    after_plan = extract_vector_plan(
        after["target"],
        after["room_labels"],
        taxonomy,
    )
    baseline = vector_plan_preservation_metrics(
        before_plan,
        after_plan,
        region,
        taxonomy,
        changed_mask=changed,
    )
    return baseline, before_plan, after_plan, changed


def main() -> None:
    taxonomy = FloorplanTaxonomy.read(TAXONOMY)
    rows = json.loads(BENCHMARK.read_text(encoding="utf-8"))["rows"]
    specs = [
        (
            "move_unaffected_room_vertex",
            "outside_room_polygon_exact",
            "before_unaffected_rooms",
            perturb_unaffected_room_vertex,
        ),
        (
            "delete_unaffected_wall_pixel",
            "outside_wall_exact",
            "before_unaffected_walls",
            perturb_unaffected_wall_mask,
        ),
        (
            "move_unaffected_opening",
            "outside_opening_exact",
            "before_unaffected_openings",
            perturb_unaffected_opening,
        ),
        (
            "remove_unaffected_room_adjacency",
            "outside_edge_exact",
            "before_unaffected_rooms",
            perturb_unaffected_room_adjacency,
        ),
    ]
    attacks = []
    for name, metric, denominator_key, transform in specs:
        row = _case_for(rows, denominator_key)
        baseline, before_plan, after_plan, changed = _metrics(row, taxonomy)
        region = normalized_region_mask(
            np.asarray(before_plan.wall_mask).shape,
            row["edit_region_normalized"],
        )
        attacked_plan = transform(after_plan, region)
        attacked = vector_plan_preservation_metrics(
            before_plan,
            attacked_plan,
            region,
            taxonomy,
            changed_mask=changed,
        )
        attacks.append(
            {
                "attack": name,
                "case_id": row["case_id"],
                "metric": metric,
                "baseline": baseline[metric],
                "attacked": attacked[metric],
                "sensitive": (
                    baseline[metric] == 1.0
                    and attacked[metric] < baseline[metric]
                ),
            }
        )
    report = {
        "status": "complete"
        if all(row["sensitive"] for row in attacks)
        else "failed",
        "source": str(BENCHMARK.relative_to(ROOT)),
        "attacks": attacks,
        "checks": {
            "four_vector_attacks": len(attacks) == 4,
            "all_attacks_sensitive": all(
                row["sensitive"] for row in attacks
            ),
            "baseline_not_trivial": all(
                row["baseline"] == 1.0 for row in attacks
            ),
        },
        "boundary": (
            "These deterministic attacks test evaluator sensitivity, not "
            "model quality."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
