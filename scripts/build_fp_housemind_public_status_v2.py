#!/usr/bin/env python3
"""Audit HouseMind's current public interface, demo, and code status."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "project_page": "https://housemind.github.io/",
    "arxiv_abs": "https://arxiv.org/abs/2603.11640",
    "arxiv_html": "https://arxiv.org/html/2603.11640",
    "cvpr_page": "https://cvpr.thecvf.com/virtual/2026/poster/37878",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_housemind_public_status_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fp_housemind_comparison_v3.md",
    )
    return parser.parse_args()


def fetch(url: str) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FloorFlow-evidence-audit/2.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="replace"), hashlib.sha256(
        payload
    ).hexdigest()


def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else None


def main() -> None:
    args = parse_args()
    pages = {}
    for name, url in SOURCES.items():
        text, digest = fetch(url)
        pages[name] = {
            "url": url,
            "sha256": digest,
            "bytes": len(text.encode("utf-8")),
            "text": text,
        }
    project = pages["project_page"]["text"]
    arxiv_abs = pages["arxiv_abs"]["text"]
    arxiv_html = pages["arxiv_html"]["text"]
    demo_match = re.search(
        r'href="(https://housemind\.ai-structure\.com/)"',
        project,
        flags=re.IGNORECASE,
    )
    code_soon = bool(
        re.search(r"Code\s*\(?\s*Soon\)?", project, flags=re.IGNORECASE)
    )
    title = first_match(
        r'<meta name="citation_title" content="([^"]+)"',
        arxiv_abs,
    )
    comments = first_match(
        r'<td class="tablecell comments mathjax">(.*?)</td>',
        arxiv_abs,
    )
    abstract = first_match(
        r'<blockquote class="abstract mathjax">(.*?)</blockquote>',
        arxiv_abs,
    )
    lower_html = arxiv_html.lower()
    term_counts = {
        term: lower_html.count(term)
        for term in (
            "outside pixel",
            "outside exact",
            "outside room",
            "strict locality",
            "strict-locality",
            "localized modification",
            "keeping others unchanged",
            "conditional watertight",
        )
    }
    quantitative_strict_locality = any(
        term_counts[term] > 0
        for term in (
            "outside pixel",
            "outside exact",
            "outside room",
            "strict locality",
            "strict-locality",
            "conditional watertight",
        )
    )
    sources = {
        name: {
            key: value
            for key, value in page.items()
            if key != "text"
        }
        for name, page in pages.items()
    }
    report = {
        "status": "verified",
        "verified_at": datetime.now(UTC).isoformat(),
        "paper": {
            "title": title,
            "arxiv_id": "2603.11640",
            "comments": comments,
            "abstract": abstract,
        },
        "public_artifacts": {
            "project_page": SOURCES["project_page"],
            "runnable_demo": demo_match.group(1) if demo_match else None,
            "code_status": "code_soon" if code_soon else "not_detected",
            "official_training_or_inference_repository": None,
            "official_code_available": False,
        },
        "metric_coverage": {
            "term_counts_in_paper_html": term_counts,
            "quantitative_strict_locality_audit_found": (
                quantitative_strict_locality
            ),
            "localized_editing_is_qualitatively_claimed": (
                term_counts["localized modification"] > 0
                or term_counts["keeping others unchanged"] > 0
            ),
            "interpretation": (
                "The paper claims localized editing and shows qualitative "
                "visual support, but the public paper text does not expose a "
                "matched outside-exact/validity/watertightness audit."
            ),
        },
        "comparison_decision": {
            "house_mind_role": "complementary_unified_mllm_context",
            "matched_reconstruction_baseline": False,
            "matched_editing_baseline": False,
            "strict_locality_comparison": False,
            "reason": (
                "Official inference, prompt assembly, and token serialization "
                "are not released; strict-locality metrics are not matched."
            ),
        },
        "sources": sources,
        "checks": {
            "official_project_verified": bool(project),
            "arxiv_accepted_cvpr_2026": (
                comments is not None and "CVPR 2026" in comments
            ),
            "runnable_demo_listed": demo_match is not None,
            "code_marked_soon": code_soon,
            "no_matched_strict_locality_claim": (
                not quantitative_strict_locality
            ),
            "qualitative_local_edit_claim_found": (
                term_counts["localized modification"] > 0
                or term_counts["keeping others unchanged"] > 0
            ),
        },
    }
    failures = [name for name, passed in report["checks"].items() if not passed]
    if failures:
        raise ValueError(f"HouseMind public-status checks failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# HouseMind Comparison v3",
        "",
        (
            "HouseMind is a complementary unified MLLM framework for floorplan "
            "understanding, generation, and editing. It is not a matched "
            "FloorFlow baseline because the official inference pipeline is not released."
        ),
        "",
        "| Dimension | HouseMind | FloorFlow | Comparison boundary |",
        "|---|---|---|---|",
        (
            "| Input | Natural-language instruction and multimodal context | "
            "Control-gated local delta with an explicit edit region | "
            "Different interfaces |"
        ),
        (
            "| Unified scope | Understanding, generation, and editing | "
            "Local editing protocol plus completion evidence | "
            "Complementary contributions |"
        ),
        (
            "| Public demo | Runnable online demo listed | "
            "Reproducible local experiment package | Different execution surfaces |"
        ),
        (
            "| Official code | Project page marks `Code (Soon)` | "
            "Source, tests, rows, and hashes packaged | HouseMind cannot be rerun |"
        ),
        (
            "| Localized editing | Qualitatively claimed | "
            "Explicitly constrained and measured | Not a same-protocol comparison |"
        ),
        (
            "| Outside exact | No matched quantitative audit found | "
            "Pixel and room-label exactness reported | FloorFlow-only metric |"
        ),
        (
            "| Validity / watertightness | Geometric validity discussed; "
            "no matched table for strict editing | Explicitly audited | "
            "Different metrics and data |"
        ),
        "",
        "## Allowed wording",
        "",
        (
            "> HouseMind and FloorFlow target complementary aspects of "
            "architectural floorplan generation. FloorFlow contributes a "
            "strict-locality editing protocol and verifiable evidence suite."
        ),
        "",
        "## Forbidden wording",
        "",
        "- FloorFlow beats HouseMind's complete method.",
        "- HouseMind fails strict locality.",
        "- The two models rank on one HouseMind leaderboard.",
        "- Raster2Seq is a native editing-aware method.",
        "",
        "## Sources",
        "",
        f"- Project: <{SOURCES['project_page']}>",
        f"- arXiv: <{SOURCES['arxiv_abs']}>",
        f"- Demo: <{report['public_artifacts']['runnable_demo']}>",
        f"- CVPR page: <{SOURCES['cvpr_page']}>",
        "",
    ]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
