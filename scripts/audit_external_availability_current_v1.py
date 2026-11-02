#!/usr/bin/env python3
"""Audit current public availability of floorplan comparison work."""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOUSEMIND = "https://housemind.github.io/"
HOUSEMIND_SEARCH = (
    "https://api.github.com/search/repositories?"
    "q=HouseMind+floorplan&per_page=10"
)
CE2EPLAN = "https://api.github.com/repos/shidong-wang/CE2EPlan"


def _get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "FloorFlow-evidence-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def main() -> None:
    page = _get(HOUSEMIND).decode("utf-8", errors="replace")
    search = json.loads(_get(HOUSEMIND_SEARCH))
    ce2eplan = json.loads(_get(CE2EPLAN))
    report = {
        "schema_version": "external-availability-current-v1",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "house_mind": {
            "project_page": HOUSEMIND,
            "project_page_reachable": bool(page),
            "code_soon_label": "Code (Soon)" in page,
            "runnable_demo_label": "Runnable online demo" in page,
            "weights_link_present": "Model weights" in page,
            "public_github_repository_count": int(
                search.get("total_count", 0)
            ),
            "complete_inference_available": False,
        },
        "ce2eplan": {
            "repository": ce2eplan["html_url"],
            "default_branch": ce2eplan["default_branch"],
            "pushed_at": ce2eplan["pushed_at"],
            "license_spdx": (
                None
                if ce2eplan.get("license") is None
                else ce2eplan["license"].get("spdx_id")
            ),
            "repository_size_kb": int(ce2eplan["size"]),
        },
        "claim_boundaries": {
            "house_mind": (
                "Weights and demo are public, but the official inference "
                "chain is unavailable; no matched editing score is assigned."
            ),
            "ce2eplan": (
                "Generation-only RPLAN reference with no matched FloorFlow "
                "editing or full-CAD comparison."
            ),
        },
    }
    output = (
        ROOT
        / "outputs/reports/external_availability_current_v1.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
