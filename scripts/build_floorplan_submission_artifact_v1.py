#!/usr/bin/env python3
"""Build a compact, hash-verified floorplan submission artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/floorplan_submission"
ROOT_FILES = (
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "README.md",
    "floorplansubmissiontodo.json",
    "fptodo.json",
    "pyproject.toml",
    "uv.lock",
)
TREE_PATTERNS = (
    "configs/**/*.json",
    "configs/**/*.txt",
    "docs/**/*.md",
    "examples/**/*",
    "paper/**/*.md",
    "paper/**/*.svg",
    "paper/**/*.pdf",
    "schema/**/*.json",
    "scripts/**/*.py",
    "src/**/*.py",
    "tests/**/*.py",
)
EVIDENCE_FILES = (
    "outputs/reports/residential_edit_test503_v1.json",
    "outputs/reports/residential_sota_main_results_v5.json",
    "outputs/reports/residential_sota_claim_gate_v5.json",
    "outputs/reports/residential_editing_aware_baseline_v1.json",
    "outputs/reports/residential_structural_hardened_v2.json",
    "outputs/reports/predicted_control_plausibility_fiveseed_v12.json",
    "outputs/reports/predicted_control_plausibility_val491_final_v12.json",
    "outputs/reports/predicted_control_fidelity_val491_v3.json",
    "outputs/reports/predicted_operation_success_val191_v1.json",
    "outputs/reports/predicted_operation_success_smoke5_v12.json",
    "outputs/reports/fevp_mechanism_pareto_v2.json",
    "outputs/reports/fevp_protocol_attacks_v2.json",
    "outputs/reports/fp_strict_locality_stress_fiveseed_v1.json",
    "outputs/reports/fp_complexity_strata_fiveseed_v1.json",
    "outputs/reports/matched_local_editing_baselines_v1.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _candidate_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for pattern in TREE_PATTERNS:
        files.extend(ROOT.glob(pattern))
    files.append(ROOT / "third_party/README.md")
    files.extend(ROOT / name for name in EVIDENCE_FILES if (ROOT / name).is_file())
    return sorted(
        {
            path
            for path in files
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        }
    )


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows = []
    for source in _candidate_files():
        relative = (
            Path("PROJECT_README.md")
            if source == ROOT / "README.md"
            else source.relative_to(ROOT)
        )
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    package_readme = OUTPUT / "README.md"
    package_readme.write_text(
        "# FloorFlow Floorplan Submission Artifact\n\n"
        "This is a draft artifact package for the residential floorplan "
        "submission. It contains source code, tests, configuration, documents, "
        "figures, and selected evidence JSON files.\n\n"
        "Datasets, checkpoints, virtual environments, third-party source trees, "
        "and full generated outputs are excluded. See `artifact_manifest.json` "
        "for the source commit, selected files, sizes, and SHA-256 hashes.\n\n"
        "The predicted-control model and held-out test are not frozen. This "
        "package is not a final submission release.\n",
        encoding="utf-8",
    )
    rows.append(
        {
            "path": "README.md",
            "bytes": package_readme.stat().st_size,
            "sha256": _sha256(package_readme),
        }
    )
    manifest = {
        "schema_version": "floorplan-submission-artifact-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": _commit(),
        "status": "draft_model_updating",
        "scope": "residential floorplan code, documents, figures, and selected evidence",
        "excluded": [
            ".data/",
            "outputs/ (except selected evidence reports)",
            "third_party source trees and weights",
            "checkpoints and arrays",
            "virtual environments and caches",
        ],
        "files": rows,
    }
    (OUTPUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "CHECKSUMS.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "files": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
