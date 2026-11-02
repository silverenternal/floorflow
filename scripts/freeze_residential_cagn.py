"""Freeze the CAGN seed and selector using validation only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import train_constraint_aware_gate as runtime

from floorflow.model.constraint_aware_gate import ConstraintAwareGate

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/residential_cagn_validation_freeze_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    device = torch.device("cuda:0")
    features = runtime._load(
        ROOT / "outputs/control_contracts/residential_cagn_features_v1/val/features.pt",
        device,
    )
    calibration = runtime._split_groups(
        features,
        features["group_index"][features["group_index"] % 2 == 0],
    )
    candidates = []
    for seed in (19, 20, 21, 22, 23):
        checkpoint_path = (
            ROOT / f"outputs/residential_cagn_gate_v3/seed{seed}.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model = ConstraintAwareGate(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        groups = runtime._predict(
            model,
            calibration,
            128,
        )["groups"]
        best = None
        for alpha in (0.0, 0.25, 0.5, 1.0, 2.0):
            for threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
                rows = runtime._fallback_rows(
                    groups,
                    alpha=alpha,
                    legal_threshold=threshold,
                )
                metrics = runtime._metrics(rows, learned=True)
                if metrics["selected_violation_rate"] > 0.005:
                    continue
                if best is None or metrics["ndcg_at_5"] > best["ndcg_at_5"]:
                    best = {
                        **metrics,
                        "alpha": alpha,
                        "legal_fallback_threshold": threshold,
                    }
        if best is None:
            raise RuntimeError(f"seed{seed} has no calibration-safe selector")
        candidates.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": _sha256(checkpoint_path),
                **best,
            }
        )
    selected = max(candidates, key=lambda row: row["ndcg_at_5"])
    report = {
        "schema_version": "residential-cagn-validation-freeze-v1",
        "status": "frozen",
        "selection_split": "even validation group index",
        "selection_metric": "NDCG@5 subject to selected-violation <= 0.005",
        "test503_opened": False,
        "candidates": candidates,
        "selected": selected,
        "decode_policy": {
            "hard_mask": "raw structural hard feasibility",
            "score": "gate_logits + alpha * legal_logits",
            "fallback": "largest legal logit when selected legal probability is below threshold",
            "candidate_limit": 4,
            "best_of_n": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
