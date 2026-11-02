#!/usr/bin/env python3
"""Build a checkpoint-level evidence ledger for the chip CVPR route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


ARMS = {
    "flat_dense_seed19": {
        "family": "flat_token",
        "seed": 19,
        "role": "control",
        "report": "outputs/chip_large/v2_20k/dense_no_ot_huber.json",
        "checkpoint": "outputs/chip_large/v2_20k/dense_no_ot_huber/checkpoints/final.pt",
        "official": "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized_cases.json",
    },
    "flat_dense_seed20": {
        "family": "flat_token",
        "seed": 20,
        "role": "control_replication",
        "report": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s20.json",
        "checkpoint": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s20/checkpoints/final.pt",
        "official": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s20/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s20/official_test100_nfe8_legalized_cases.json",
    },
    "flat_dense_seed21": {
        "family": "flat_token",
        "seed": 21,
        "role": "control_replication",
        "report": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s21.json",
        "checkpoint": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s21/checkpoints/final.pt",
        "official": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s21/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber_s21/official_test100_nfe8_legalized_cases.json",
    },
    "global_slot_seed19": {
        "family": "global_slot",
        "seed": 19,
        "role": "primary_candidate",
        "report": "outputs/chip_slot/global_v1/global_slot_overlap10_20k.json",
        "checkpoint": "outputs/chip_slot/global_v1/global_slot_overlap10_20k/checkpoints/final.pt",
        "official": "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    },
    "global_slot_seed20": {
        "family": "global_slot",
        "seed": 20,
        "role": "candidate_replication",
        "report": "outputs/chip_slot/global_v2/global_slot_seed20_overlap10_20k.json",
        "checkpoint": "outputs/chip_slot/global_v2/global_slot_seed20_overlap10_20k/checkpoints/final.pt",
        "official": "outputs/chip_slot/global_v2/global_slot_seed20_overlap10_20k/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_slot/global_v2/global_slot_seed20_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    },
    "global_slot_seed21": {
        "family": "global_slot",
        "seed": 21,
        "role": "candidate_replication",
        "report": "outputs/chip_slot/global_v2/global_slot_seed21_overlap10_20k.json",
        "checkpoint": "outputs/chip_slot/global_v2/global_slot_seed21_overlap10_20k/checkpoints/final.pt",
        "official": "outputs/chip_slot/global_v2/global_slot_seed21_overlap10_20k/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_slot/global_v2/global_slot_seed21_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    },
    "fixed_hierarchy_seed19": {
        "family": "fixed_slot_hierarchy",
        "seed": 19,
        "role": "negative_control",
        "report": "outputs/chip_hierarchical/v2_matched/hierarchical_slot_overlap10_20k.json",
        "checkpoint": "outputs/chip_hierarchical/v2_matched/hierarchical_slot_overlap10_20k/checkpoints/final.pt",
        "official": "outputs/chip_hierarchical/v2_matched/hierarchical_slot_overlap10_20k/official_test100_nfe8_legalized.json",
        "cases": "outputs/chip_hierarchical/v2_matched/hierarchical_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
    },
}


RETIRED_ARTIFACTS = {
    "sparse_v1_seed19": {
        "report": "outputs/chip_sparse/v1/sparse_prod_seed19_20k.json",
        "checkpoint": "outputs/chip_sparse/v1/sparse_prod_seed19_20k/checkpoints/last.pt",
        "reason": "Incomplete pre-v2 run; checkpoint shape is incompatible with sparse hierarchy v2.",
    },
    "flat_sparse16_compute_claim": {
        "report": "outputs/reports/chip_large_v2_ablation_summary.json",
        "checkpoint": "outputs/chip_large/v2_20k/sparse16_ot/checkpoints/final.pt",
        "reason": "Top-k masked dense score computation; cannot support a sparse-compute claim.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_cvpr_evidence_ledger_v1.json",
    )
    parser.add_argument("--skip-checkpoint-load", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_parameters(state: dict[str, torch.Tensor]) -> int:
    return sum(
        int(value.numel())
        for value in state.values()
        if torch.is_tensor(value)
    )


def architecture(config: dict) -> str:
    return str(config.get("architecture", "flat_token"))


def arm_row(
    name: str,
    spec: dict,
    *,
    load_checkpoint: bool,
) -> dict:
    paths = {
        key: ROOT / str(spec[key])
        for key in ("report", "checkpoint", "official", "cases")
    }
    checks = {f"{key}_exists": path.is_file() for key, path in paths.items()}
    row = {
        "arm": name,
        "family": spec["family"],
        "seed": int(spec["seed"]),
        "claim_role": spec["role"],
        "paths": {key: str(path) for key, path in paths.items()},
        "checks": checks,
    }
    if not paths["report"].is_file() or not paths["official"].is_file():
        row["eligible"] = False
        return row
    report = load_json(paths["report"])
    official = load_json(paths["official"])
    config = report.get("config", {})
    row.update(
        {
            "report_status": report.get("status"),
            "report_parameters": report.get("model_parameters"),
            "config_architecture": architecture(config),
            "steps": config.get("steps"),
            "batch_size": config.get("batch_size"),
            "sampling_steps": official.get("sampling_steps"),
            "use_ema": official.get("use_ema"),
            "legalize": official.get("evaluation", {}).get("legalize"),
            "metric_schema": official.get("metric_schema"),
            "official": {
                key: official.get(key)
                for key in (
                    "cases",
                    "raw_hard_cost",
                    "raw_feasible_fraction",
                    "legalized_hard_cost",
                    "legalized_v_rel",
                    "legalized_hpwl_gap",
                    "legalized_area_gap",
                    "legalized_feasible_fraction",
                    "latent_mse",
                    "rho_mae",
                    "active_slot_count_min",
                    "active_slot_count_max",
                    "router_budget_min",
                    "router_budget_max",
                )
                if key in official
            },
        }
    )
    has_legalized_cases = False
    if paths["cases"].is_file():
        cases = load_json(paths["cases"])
        row["case_indices"] = sorted(int(item["index"]) for item in cases)
        has_legalized_cases = bool(cases) and all(
            "legalized_hard_cost" in item for item in cases
        )
    if paths["checkpoint"].is_file():
        row["checkpoint_sha256"] = sha256(paths["checkpoint"])
        row["checkpoint_bytes"] = paths["checkpoint"].stat().st_size
        if load_checkpoint:
            payload = torch.load(
                paths["checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
            state = payload.get("model", {})
            row["checkpoint_step"] = payload.get("step")
            row["checkpoint_parameters"] = state_parameters(state)
            row["checkpoint_architecture"] = architecture(
                payload.get("config", {})
            )
    expected_indices = list(range(100))
    row["checks"].update(
        {
            "report_complete": report.get("status") == "complete",
            "official_cases_100": official.get("cases") == 100,
            "case_indices_complete": row.get("case_indices") == expected_indices,
            "nfe8": official.get("sampling_steps") == 8,
            "legalized": (
                bool(official.get("evaluation", {}).get("legalize"))
                or has_legalized_cases
            ),
            "architecture_consistent": (
                row.get("checkpoint_architecture", row["config_architecture"])
                == row["config_architecture"]
            ),
            "parameters_consistent": (
                row.get("checkpoint_parameters", report.get("model_parameters"))
                == report.get("model_parameters")
            ),
        }
    )
    row["eligible"] = all(
        value is True
        for key, value in row["checks"].items()
        if key != "parameters_consistent" or row["checks"]["report_complete"]
    )
    return row


def main() -> None:
    args = parse_args()
    rows = {
        name: arm_row(
            name,
            spec,
            load_checkpoint=not args.skip_checkpoint_load,
        )
        for name, spec in ARMS.items()
    }
    core_families = {"flat_token", "global_slot"}
    eligible_core = [
        row
        for row in rows.values()
        if row["eligible"] and row["family"] in core_families
    ]
    retired = {}
    for name, spec in RETIRED_ARTIFACTS.items():
        paths = {
            key: ROOT / value
            for key, value in spec.items()
            if key != "reason"
        }
        retired[name] = {
            **spec,
            "exists": {key: path.is_file() for key, path in paths.items()},
            "eligible": False,
        }
    seed_report = load_json(
        ROOT / "outputs/reports/plan_c_global_slot_seeds_v1.json"
    )
    hierarchy_report = load_json(
        ROOT / "outputs/reports/two_domain_dynamic_hierarchy_v1.json"
    )
    checks = {
        "six_core_seed_rows_eligible": len(eligible_core) == 6,
        "three_flat_seeds_eligible": sum(
            row["family"] == "flat_token" for row in eligible_core
        )
        == 3,
        "three_global_seeds_eligible": sum(
            row["family"] == "global_slot" for row in eligible_core
        )
        == 3,
        "global_seed_gate_pass": seed_report.get("gate", {}).get("status")
        == "pass",
        "hierarchy_negative_retained": hierarchy_report.get(
            "route_b_promotion", {}
        ).get("status")
        == "fail",
        "sparse_v2_has_no_checkpoint": not (
            ROOT / "outputs/chip_sparse/v2"
        ).exists(),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "checkpoint and protocol ledger only; no accuracy claim",
        "core_arm_count": len(ARMS),
        "eligible_core_arm_count": len(eligible_core),
        "checks": checks,
        "arms": rows,
        "retired_artifacts": retired,
        "claim_status": {
            "primary_fallback": "global_slot_seed_replication",
            "global_slot_gate": seed_report.get("gate"),
            "sparse_hierarchy_v2": "no trained checkpoint",
            "fixed_slot_hierarchy": "negative control",
        },
        "protocol_warning": (
            "Eligibility means internal traceability at NFE8 with scale-pack "
            "legalization. External Trinity comparison still requires P1."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
