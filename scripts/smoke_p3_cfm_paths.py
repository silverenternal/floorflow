"""Fixed-budget toy comparison of uniform and logit-normal CFM times."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from floorflow.flow.cfm_backend import sample_time


def run(sampling: str, seed: int, steps: int = 200) -> float:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 16),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    source = torch.randn(128, 1) * 2.0
    target = 2.0 * source + 1.0
    velocity = target - source
    for _ in range(steps):
        time = sample_time(
            128,
            device=source.device,
            dtype=source.dtype,
            sampling=sampling,
        )
        location = (1.0 - time[:, None]) * source + time[:, None] * target
        prediction = model(torch.cat([location, time[:, None]], dim=1))
        loss = (prediction - velocity).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    eval_time = torch.linspace(0.0, 1.0, 101)[:, None]
    eval_time = eval_time.repeat_interleave(4, dim=1).reshape(-1, 1)
    eval_source = torch.randn_like(eval_time) * 2.0
    eval_location = (1.0 - eval_time) * eval_source + eval_time * (
        2.0 * eval_source + 1.0
    )
    with torch.no_grad():
        error = model(torch.cat([eval_location, eval_time], dim=1))
        return float((error - eval_source - 1.0).pow(2).mean())


def main() -> int:
    torch.set_num_threads(1)
    results = {
        sampling: [run(sampling, seed) for seed in range(3)]
        for sampling in ("uniform", "logit_normal")
    }
    means = {
        name: sum(values) / len(values)
        for name, values in results.items()
    }
    report = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "steps": 200,
        "results": results,
        "means": means,
        "promotion": (
            "logit_normal"
            if means["logit_normal"] < means["uniform"]
            else "uniform"
        ),
        "note": "Toy path comparison only; full promotion remains gated by P5.",
    }
    path = Path("outputs/reports/p3_cfm_path_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
