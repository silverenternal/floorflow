#!/usr/bin/env python3
"""Freeze license-audited ORFS cases before candidate PPA evaluation."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONF = Path("third_party/OpenROAD-flow-scripts")
DEFAULT_LOCK = ROOT / "configs/chip_orfs_external_lock_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"

CANDIDATE_CASES = (
    "nangate45/mempool_group",
    "nangate45/ariane133",
    "nangate45/bp_quad",
    "ariane136",
    "black_parrot",
    "bp_be_top",
    "bp_fe_top",
    "bp_multi_top",
    "asap7/cva6",
    "asap7/swerv_wrapper",
    "asap7/riscv32i",
    "asap7/coralnpu",
    "swerv_wrapper",
    "tinyRocket",
)

SOURCE_LICENSE_ALIASES = {
    "nangate45/ariane133": "flow/designs/src/ariane136/LICENSE",
    "nangate45/bp_quad": "flow/designs/src/black_parrot/LICENSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orfs-root", type=Path, default=ROOT / DEFAULT_CONF)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-home", type=Path, default=ROOT / "outputs/chip_ppa/orfs")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--all-eligible", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def expand_make_value(value: str, variables: dict[str, str], limit: int = 20) -> str:
    current = value
    for _ in range(limit):
        refs = re.findall(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", current)
        for name in refs:
            current = current.replace(f"$({name})", variables.get(name, ""))
        wildcards = re.findall(r"\$\(wildcard\s+([^)]+)\)", current)
        for pattern in wildcards:
            matched = " ".join(
                sorted(
                    path
                    for part in pattern.split()
                    for path in glob.glob(part)
                )
            )
            current = current.replace(f"$(wildcard {pattern})", matched)
        sorts = re.findall(r"\$\(sort\s+([^()]*)\)", current)
        for content in sorts:
            current = current.replace(f"$(sort {content})", " ".join(sorted(content.split())))
        if not refs and not wildcards and not sorts:
            break
    return current


def parse_config(path: Path, orfs_root: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\\n\s*", " ", text)
    variables: dict[str, str] = {}
    for match in re.finditer(
        r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+|\?)?=\s*(.*)$",
        text,
        re.MULTILINE,
    ):
        variables[match.group(1)] = match.group(2).strip()
    flow = orfs_root / "flow"
    variables.setdefault("DESIGN_HOME", str(flow / "designs"))
    variables.setdefault("PLATFORM_HOME", str(flow / "platforms"))
    variables.setdefault("PLATFORM", path.parents[1].name)
    variables.setdefault("PLATFORM_DIR", str(flow / "platforms" / variables["PLATFORM"]))
    variables.setdefault("DESIGN_NICKNAME", variables.get("DESIGN_NAME", path.parent.name))
    variables.setdefault("DESIGN_DIR", str(path.parent))
    for name, value in tuple(variables.items()):
        variables[name] = expand_make_value(value, variables)
    return variables


def make_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for token in value.split():
        if not token or "$(" in token:
            continue
        if "*" in token:
            paths.extend(Path(path) for path in sorted(glob.glob(token)))
        else:
            paths.append(Path(token))
    return paths


def licenses_for_sources(orfs_root: Path, sources: list[Path]) -> list[Path]:
    licenses: set[Path] = set()
    source_root = orfs_root / "flow/designs/src"
    for source in sources:
        try:
            relative = source.resolve().relative_to(source_root.resolve())
        except ValueError:
            continue
        if not relative.parts:
            continue
        design_dir = source_root / relative.parts[0]
        licenses.update(path for path in design_dir.glob("LICENSE*") if path.is_file())
        license_dir = design_dir / "LICENSES"
        if license_dir.is_dir():
            licenses.update(path for path in license_dir.iterdir() if path.is_file())
    return sorted(licenses)


def embedded_license_evidence(
    paths: list[Path],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Extract explicit license identifiers from compact source/data headers."""

    evidence = []
    for path in paths:
        if not path.is_file():
            continue
        header = path.read_text(errors="replace")[:1_500_000]
        matches = re.findall(
            r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)",
            header,
            re.IGNORECASE,
        )
        marker_fallbacks = (
            (r"Licensed under the Apache License, Version 2\.0", "Apache-2.0"),
            (r"BSD 3-Clause License", "BSD-3-Clause"),
            (r"Solderpad Hardware License, Version 0\.51", "SHL-0.51"),
            (r"Licensed under the MIT License", "MIT"),
            (r"GNU Lesser General Public License", "LGPL-2.1-or-later"),
        )
        if not matches:
            for pattern, license_id in marker_fallbacks:
                if re.search(pattern, header, re.IGNORECASE):
                    matches.append(license_id)
                    break
        if matches:
            evidence.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "spdx": matches[0],
                    "evidence": "embedded_header",
                }
            )
            if len(evidence) >= limit:
                break
    return evidence


def macro_masters(lef_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in lef_paths:
        if path.is_file():
            names.update(
                re.findall(
                    r"^\s*MACRO\s+(\S+)",
                    path.read_text(errors="replace"),
                    re.MULTILINE,
                )
            )
    return names


def count_macros_in_def(def_path: Path, masters: set[str]) -> int:
    if not def_path.is_file():
        return 0
    count = 0
    in_components = False
    for line in def_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("COMPONENTS "):
            in_components = True
            continue
        if in_components and stripped.startswith("END COMPONENTS"):
            break
        match = re.match(r"^-\s+\S+\s+(\S+)", stripped)
        if in_components and match and match.group(1) in masters:
            count += 1
    return count


def count_macros_in_odb(odb_path: Path, image_tag: str) -> dict[str, Any]:
    relative = odb_path.resolve().relative_to(ROOT)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{ROOT}:/work",
            "--entrypoint",
            "/bin/bash",
            image_tag,
            "-lc",
            (
                "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad "
                "-exit -no_init -no_splash -threads 1 "
                f"-db /work/{relative} /work/scripts/orfs_count_macros.tcl"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    count_match = re.search(r"^ORFS_MACRO_INSTANCE_COUNT\t(\d+)$", result.stdout, re.MULTILINE)
    master_matches = re.findall(r"^ORFS_MACRO_MASTER\t(\S+)$", result.stdout, re.MULTILINE)
    return {
        "count": int(count_match.group(1)) if count_match else None,
        "source": str(odb_path.relative_to(ROOT)),
        "masters": sorted(master_matches),
    }


def find_macro_count(
    work_home: Path,
    platform: str,
    nickname: str,
    masters: set[str],
    image_tag: str,
) -> dict[str, Any]:
    pattern = f"results/{platform}/{nickname}/*"
    for variant_dir in sorted(path for path in work_home.glob(pattern) if path.is_dir()):
        for name in ("6_final.def", "2_floorplan.def", "5_route.def"):
            def_path = variant_dir / name
            if def_path.is_file():
                return {
                    "count": count_macros_in_def(def_path, masters),
                    "source": str(def_path.relative_to(ROOT)),
                    "masters": sorted(masters),
                    "flow_variant": variant_dir.name,
                }
        for name in ("2_floorplan.odb", "2_4_floorplan_pdn.odb"):
            odb_path = variant_dir / name
            if odb_path.is_file():
                result = count_macros_in_odb(odb_path, image_tag)
                result["flow_variant"] = variant_dir.name
                return result
    return {"count": None, "source": None, "flow_variant": None}


def audit_case(
    orfs_root: Path,
    work_home: Path,
    case: str,
    image_tag: str,
) -> dict[str, Any]:
    design_key = case if "/" in case else f"nangate45/{case}"
    config = orfs_root / "flow/designs" / design_key / "config.mk"
    case_name = design_key.rsplit("/", 1)[-1]
    variables = parse_config(config, orfs_root)
    platform = variables["PLATFORM"]
    nickname = variables["DESIGN_NICKNAME"]
    sources = make_paths(variables.get("VERILOG_FILES", ""))
    lefs = [
        *make_paths(variables.get("ADDITIONAL_LEFS", "")),
        *make_paths(variables.get("WRAP_LEFS", "")),
    ]
    libs = [
        *make_paths(variables.get("ADDITIONAL_LIBS", "")),
        *make_paths(variables.get("WRAP_LIBS", "")),
    ]
    sdc = make_paths(variables.get("SDC_FILE", ""))
    licenses = licenses_for_sources(orfs_root, sources)
    source_headers = embedded_license_evidence(sources)
    alias_path = SOURCE_LICENSE_ALIASES.get(design_key)
    alias_license = orfs_root / alias_path if alias_path else None
    alias_exists = alias_license is not None and alias_license.is_file()
    platform_license = orfs_root / "flow/platforms" / platform / "LICENSE"
    platform_root = orfs_root / "flow/platforms" / platform
    platform_files = sorted(
        [
            *platform_root.glob("lef/*.lef"),
            *platform_root.glob("lib/**/*.lib"),
        ]
    )
    platform_headers = embedded_license_evidence(platform_files, limit=1)
    masters = macro_masters(lefs)
    macro_count = find_macro_count(work_home, platform, nickname, masters, image_tag)
    checks = {
        "config_exists": config.is_file(),
        "verilog_inputs_exist": bool(sources) and all(path.is_file() for path in sources),
        "sdc_exists": len(sdc) == 1 and sdc[0].is_file(),
        "additional_lefs_exist": bool(lefs) and all(path.is_file() for path in lefs),
        "additional_libs_exist": bool(libs) and all(path.is_file() for path in libs),
        "source_license_exists": bool(licenses or source_headers or alias_exists),
        "platform_license_exists": bool(
            platform_license.is_file() or platform_headers
        ),
        "macro_capable": bool(masters),
        "baseline_macro_count_measured": macro_count["count"] is not None,
        "nonzero_macro_count": bool(macro_count["count"]),
    }
    return {
        "case": case_name,
        "design_key": design_key,
        "config": str(config.relative_to(ROOT)),
        "platform": platform,
        "design_name": variables.get("DESIGN_NAME"),
        "design_nickname": nickname,
        "sdc": [str(path.relative_to(ROOT)) for path in sdc if path.is_file()],
        "macro_master_count": len(masters),
        "macro_instance_count": macro_count["count"],
        "macro_count_source": macro_count["source"],
        "source_licenses": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in licenses
        ],
        "source_license_headers": source_headers,
        "source_license_alias": (
            {
                "path": str(alias_license.relative_to(ROOT)),
                "sha256": sha256(alias_license),
                "relationship": "shared_upstream_source_license",
            }
            if alias_exists
            else None
        ),
        "platform_license": {
            "path": (
                str(platform_license.relative_to(ROOT))
                if platform_license.is_file()
                else (
                    platform_headers[0]["path"]
                    if platform_headers
                    else None
                )
            ),
            "sha256": (
                sha256(platform_license)
                if platform_license.is_file()
                else (
                    platform_headers[0]["sha256"]
                    if platform_headers
                    else None
                )
            ),
            "spdx": (
                platform_headers[0]["spdx"]
                if platform_headers and not platform_license.is_file()
                else "Apache-2.0"
                if platform_license.is_file()
                else None
            ),
            "evidence": (
                "license_file"
                if platform_license.is_file()
                else "embedded_header"
                if platform_headers
                else None
            ),
        },
        "checks": checks,
        "eligible": all(checks.values()),
    }


def select_cases(
    inventory: list[dict[str, Any]],
    *,
    max_cases: int = 5,
    all_eligible: bool = False,
) -> dict[str, Any]:
    eligible = [case for case in inventory if case["eligible"]]
    strata = {
        "small_1_40": [case for case in eligible if case["macro_instance_count"] <= 40],
        "medium_41_80": [
            case for case in eligible if 41 <= case["macro_instance_count"] <= 80
        ],
        "large_81_120": [
            case for case in eligible if 81 <= case["macro_instance_count"] <= 120
        ],
        "extra_large_121_plus": [
            case for case in eligible if case["macro_instance_count"] >= 121
        ],
    }
    if all_eligible:
        return {
            "selected_cases": [case["design_key"] for case in eligible],
            "selected_count": len(eligible),
            "stratum_membership": {
                name: [case["design_key"] for case in cases]
                for name, cases in strata.items()
            },
            "empty_strata": [name for name, cases in strata.items() if not cases],
            "macro_complexity_limited_to_small": not strata["medium_41_80"]
            and not strata["large_81_120"]
            and not strata["extra_large_121_plus"],
            "minimum_case_gate_passed": len(eligible) >= 3,
            "target_case_gate_passed": len(eligible) >= max_cases,
            "selection_mode": "all_eligible",
            "independent_design_count": len(eligible),
            "planned_seed_count": 3,
            "planned_case_seed_pairs": len(eligible) * 3,
        }
    selected: list[dict[str, Any]] = []
    for name in (
        "small_1_40",
        "medium_41_80",
        "large_81_120",
        "extra_large_121_plus",
    ):
        if strata[name]:
            selected.append(min(strata[name], key=lambda item: item["design_key"]))
    remainder = sorted(
        (case for case in eligible if case not in selected),
        key=lambda item: (item["macro_instance_count"], item["design_key"]),
    )
    selected.extend(remainder[: max(0, max_cases - len(selected))])
    return {
        "selected_cases": [case["design_key"] for case in selected],
        "selected_count": len(selected),
        "stratum_membership": {
            name: [case["design_key"] for case in cases] for name, cases in strata.items()
        },
        "empty_strata": [name for name, cases in strata.items() if not cases],
        "macro_complexity_limited_to_small": not strata["medium_41_80"]
        and not strata["large_81_120"]
        and not strata["extra_large_121_plus"],
        "minimum_case_gate_passed": len(selected) >= 3,
        "target_case_gate_passed": len(selected) == max_cases,
        "selection_mode": "stratified_first",
        "independent_design_count": len(selected),
        "planned_seed_count": 3,
        "planned_case_seed_pairs": len(selected) * 3,
    }


def docker_image_audit(image_tag: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        check=True,
        capture_output=True,
        text=True,
    )
    image = json.loads(result.stdout)[0]
    return {
        "tag": image_tag,
        "image_id": image["Id"],
        "repo_digests": image.get("RepoDigests", []),
        "created": image.get("Created"),
    }


def main() -> None:
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    inventory = [
        audit_case(args.orfs_root, args.work_home, case, lock["image"]["local_tag"])
        for case in (args.cases or CANDIDATE_CASES)
    ]
    selection = select_cases(
        inventory,
        max_cases=args.max_cases,
        all_eligible=args.all_eligible,
    )
    image = docker_image_audit(lock["image"]["local_tag"])
    repo_commit = git_value(args.orfs_root, "rev-parse", "HEAD")
    checks = {
        "orfs_commit_matches_lock": repo_commit == lock["repository"]["commit"],
        "flow_license_matches": sha256(
            args.orfs_root / lock["repository"]["flow_license_path"]
        )
        == lock["repository"]["flow_license_sha256"],
        "image_loaded": image["image_id"] is not None,
        "minimum_case_gate_passed": selection["minimum_case_gate_passed"],
    }
    report = {
        "version": "2026-10-28-chip-external-benchmark-manifest-v1",
        "status": "pass" if all(checks.values()) else "pending",
        "scope": "Static and license audit plus deterministic case freeze before candidate PPA",
        "source": {
            "url": lock["repository"]["url"],
            "commit": repo_commit,
            "flow_license": lock["repository"]["flow_license"],
            "flow_license_sha256": sha256(
                args.orfs_root / lock["repository"]["flow_license_path"]
            ),
            "platform": lock["platform"],
            "image": image,
            "oci_manifest": lock["image"],
        },
        "inventory": inventory,
        "selection": selection,
        "checks": checks,
        "policy": {
            "candidate_outcomes_inspected": False,
            "flowplace_assets_used": False,
            "selection_rule": lock["flow_settings"],
            "unmeasured_cases_are_not_selected": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selection": selection, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
