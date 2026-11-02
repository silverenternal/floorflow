#!/usr/bin/env python3
"""Audit Raster2Seq's official anchor against FloorFlow's test-case units."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "outputs/reports/fp_raster2seq_official_eval_metrics_v1.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / ".data/external/raster2seq/coco_cubicasa5k_nowalls_v4_test_refined/"
        "annotations_json/test",
    )
    parser.add_argument(
        "--image-id-mapping",
        type=Path,
        default=ROOT
        / ".data/external/raster2seq/coco_cubicasa5k_nowalls_v4_test_refined/"
        "annotations/test_image_id_mapping.json",
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=ROOT / ".data/cubicasa5k/test.txt",
    )
    parser.add_argument(
        "--project-cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/test",
    )
    parser.add_argument(
        "--adapter-diagnostic",
        type=Path,
        default=ROOT
        / "outputs/reports/fp_raster2seq_identity_reconstruction_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_raster2seq_official_anchor_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fp_raster2seq_official_audit_v1.md",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_cases(path: Path) -> list[str]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if (value := line.strip().strip("/"))
    ]


def official_sources(annotation_dir: Path, mapping: dict[str, int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    annotation_ids: set[int] = set()
    for path in annotation_dir.glob("*.json"):
        final_id, source_id = path.stem.split("_")
        if str(int(final_id)) not in mapping:
            raise ValueError(f"annotation id {final_id} is absent from the mapping")
        if int(final_id) in annotation_ids:
            raise ValueError(f"duplicate final image id {final_id}")
        annotation_ids.add(int(final_id))
        counts[int(source_id)] += 1
    if len(annotation_ids) != len(mapping):
        raise ValueError("annotation/source cardinality disagrees with mapping")
    return counts


def project_sources(cases: list[str], cache_root: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for index, case in enumerate(cases):
        floors = sorted((cache_root / case).glob("Floor_*.npz"))
        counts[3500 + index] = len(floors)
    return counts


def main() -> None:
    args = parse_args()
    metrics = read_json(args.metrics)
    mapping = {
        str(key): int(value)
        for key, value in read_json(args.image_id_mapping).items()
    }
    cases = read_cases(args.test_split)
    official = official_sources(args.annotations, mapping)
    project = project_sources(cases, args.project_cache_root)
    all_sources = sorted(set(official) | set(project))
    exact = [
        source
        for source in all_sources
        if official.get(source, 0) == project.get(source, 0)
    ]
    official_more = [
        source
        for source in all_sources
        if official.get(source, 0) > project.get(source, 0)
    ]
    official_less = [
        source
        for source in all_sources
        if official.get(source, 0) < project.get(source, 0)
    ]
    missing_sources = sorted(set(project) - set(official))
    extra_sources = sorted(set(official) - set(project))
    mismatches = [
        {
            "source_id": source,
            "case": cases[source - 3500] if 0 <= source - 3500 < len(cases) else None,
            "official_cases": official.get(source, 0),
            "project_floor_cases": project.get(source, 0),
        }
        for source in (*official_more, *official_less)
    ]
    adapter = read_json(args.adapter_diagnostic)
    report = {
        "status": "official_anchor_complete_matched_identity_blocked",
        "official_anchor": metrics,
        "case_unit_audit": {
            "image_id_mapping_entries": len(mapping),
            "official_annotation_cases": sum(official.values()),
            "official_sources": len(official),
            "project_floor_cases": sum(project.values()),
            "project_sources": len(project),
            "exact_cardinality_sources": len(exact),
            "official_more_sources": len(official_more),
            "official_less_sources": len(official_less),
            "project_sources_absent_from_official": missing_sources,
            "official_sources_absent_from_project": extra_sources,
            "mismatched_sources": mismatches,
            "bijection_possible_for_all_503_project_floors": False,
            "reason": (
                "Raster2Seq floorplan_extraction creates one case per contour "
                "region in a source mask, while FloorFlow creates one case per "
                "SVG floor. Twelve source folders differ in cardinality, "
                "including two absent from official annotations and three "
                "where one official case covers more project floors than can "
                "be paired one-to-one. No authoritative region-to-floor "
                "manifest is present."
            ),
        },
        "matched_identity_audit": {
            "eligible": False,
            "blocked_reason": (
                "A 503-case one-to-one mapping cannot be established from the "
                "released preprocessing artifacts alone. Fuzzy geometry "
                "matching is diagnostic and cannot serve as matched evidence."
            ),
            "required_to_unblock": (
                "Produce an authoritative source-region-to-FloorFlow-floor "
                "manifest, or regenerate both methods with identical case units."
            ),
        },
        "adapter_diagnostic": {
            "eligible_for_main_table": False,
            "status": adapter.get("status"),
            "source": str(args.adapter_diagnostic.relative_to(ROOT)),
            "boundary": (
                "Project-defined official-style renderer sensitivity only; "
                "not an upstream preprocessing result."
            ),
        },
        "claims": {
            "allowed": [
                "Raster2Seq RoomF1 reproduces the public CubiCasa anchor within 0.002.",
                "Raster2Seq's upstream preprocessing uses a different case unit.",
            ],
            "forbidden": [
                "FloorFlow beats Raster2Seq on matched identity reconstruction.",
                "The adapter diagnostic is an official matched score.",
                "The 509-case anchor and the 503-floor protocol form one table.",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Raster2Seq Official Anchor and Case-Unit Audit",
        "",
        "| Item | Raster2Seq official | FloorFlow project |",
        "|---|---:|---:|",
        f"| Cases | {sum(official.values())} | {sum(project.values())} |",
        f"| Source folders | {len(official)} | {len(project)} |",
        "| Eligible for matched identity table | no | no |",
        "",
        (
            f"Official RoomF1 is {metrics['metrics']['room_f1']:.6f}; the "
            f"published anchor is {metrics['published_room_f1_anchor']:.3f}."
        ),
        "",
        (
            f"Case counts agree for {len(exact)} sources and differ for "
            f"{len(mismatches)} sources. A 503-case bijection is therefore "
            "not recoverable without an authoritative region-to-floor manifest."
        ),
        "",
        (
            "Raster2Seq remains a reconstruction method. The official anchor and "
            "the FloorFlow editing table are not one leaderboard."
        ),
        "",
    ]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                **report["case_unit_audit"],
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
