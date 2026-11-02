from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "chip_v45_repaired_cohort",
    ROOT / "scripts/build_chip_v45_repaired_cohort_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pair_row_requires_distinct_hashes_and_common_route() -> None:
    row = MODULE.pair_row(
        "p/design",
        "corrected_sensitivity",
        {
            "tcl_sha256": "candidate",
            "legalizer_route": "scale_pack_core_fit",
        },
        {
            "tcl_sha256": "control",
            "legalizer_route": "scale_pack_core_fit",
        },
    )

    assert row["distinct_legalized_placements"]
    assert row["common_route"]
    assert row["routed_endpoint_status"] == "pending_sensitivity"
