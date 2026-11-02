"""Record the formally published CE2EPlan anchor and comparability boundary."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    anchor = json.loads(
        (
            ROOT / "outputs/reports/ce2eplan_official_anchor_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = {
        "status": "complete",
        "scope": "CE2EPlan boundary-conditioned RPLAN generation",
        "bibliography": {
            "title": (
                "Directly From Alpha to Omega: Controllable End-to-End "
                "Vector Floor Plan Generation"
            ),
            "authors": ["Shidong Wang", "Renato Pajarola"],
            "venue": (
                "IEEE Transactions on Visualization and Computer Graphics"
            ),
            "published": "2026-07",
            "doi": "10.1109/TVCG.2026.3665422",
            "arxiv": "2602.20377",
            "crossref_verified": True,
            "arxiv_table_verified": True,
        },
        "published_anchor": anchor["published_anchor"],
        "local_reproduction": anchor["reproduced"],
        "source_availability": {
            "code_commit": "774e670941eeda3e6677303ff79e8d517ae0d1c0",
            "license_found": False,
            "source_redistributed": False,
            "rplan_data_redistributed": False,
            "public_sample_count": 200,
        },
        "checks": {
            "formal_venue_verified": True,
            "doi_verified": True,
            "published_table_verified": True,
            "local_reproduction_available": True,
            "editing_comparison": False,
            "matched_rplan_floorflow": False,
        },
        "decision": (
            "Use CE2EPlan as a generation-only RPLAN reference anchor. Its FID "
            "and GED cannot be compared with FloorFlow's CubiCasa editing "
            "Delta-IoU, Delta-MSE, or locality metrics."
        ),
        "boundary": (
            "Formal publication does not make the task matched. No CE2EPlan "
            "editing or full-CAD ranking is claimed."
        ),
    }
    output = ROOT / "outputs/reports/ce2eplan_tvcg_anchor_v2.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = ROOT / "paper/tables/ce2eplan_tvcg_anchor_v2.md"
    table.write_text(
        "\n".join(
            [
                "# CE2EPlan TVCG 2026 Anchor",
                "",
                (
                    "Wang and Pajarola, *Directly From Alpha to Omega*, "
                    "IEEE TVCG, July 2026, DOI "
                    "`10.1109/TVCG.2026.3665422`."
                ),
                "",
                "| Metric | Published | Local reproduction |",
                "|---|---:|---:|",
                (
                    f"| FID | {anchor['published_anchor']['FID']:.2f} | "
                    f"{anchor['reproduced']['FID']:.4f} |"
                ),
                (
                    f"| GED | {anchor['published_anchor']['GED']:.2f} | "
                    f"{anchor['reproduced']['GED_mean']:.4f} |"
                ),
                "",
                (
                    "Generation-only RPLAN anchor; not an editing baseline "
                    "and not a matched FloorFlow leaderboard."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
