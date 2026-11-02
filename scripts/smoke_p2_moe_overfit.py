"""Synthetic 16-case overfit check for the wall/opening spatial MoE."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from floorflow.heads.topology_moe import SpatialExpertMoE


def main() -> int:
    torch.manual_seed(7)
    cases = 16
    hidden = torch.randn(cases, 16, 8, 8)
    target = torch.zeros(cases, 2, 8, 8)
    target[:, 0, :, 3:5] = 1.0
    target[:, 1, 2:4, 6:8] = 1.0
    model = SpatialExpertMoE(16, 2, experts=4, top_k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = None
    for _ in range(80):
        optimizer.zero_grad()
        prediction, diagnostics = model(hidden)
        loss = torch.nn.functional.mse_loss(prediction.sigmoid(), target)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction, diagnostics = model(hidden)
        final = float(torch.nn.functional.mse_loss(prediction.sigmoid(), target))
    usage = diagnostics["expert_usage"].mean(dim=0)
    report = {
        "status": "pass" if final < initial * 0.8 else "fail",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "initial_loss": initial,
        "final_loss": final,
        "relative_reduction": (initial - final) / max(initial, 1.0e-9),
        "expert_usage": usage.tolist(),
        "max_expert_fraction": float(usage.max()),
        "gate_entropy": float(diagnostics["gate_entropy"].detach()),
        "gate_load_balance": float(diagnostics["gate_load_balance"].detach()),
    }
    path = Path("outputs/reports/p2_moe_overfit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" and report["max_expert_fraction"] < 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
