"""Optional Torch profiler capture for one training epoch."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def profile_training_epoch(
    device: torch.device,
    output_dir: str | Path | None,
    *,
    active_steps: int = 3,
):
    if output_dir is None:
        yield None
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=max(1, int(active_steps)),
            repeat=1,
        ),
    ) as profiler:
        if hasattr(torch, "profiler") and hasattr(
            torch.profiler,
            "record_function",
        ):
            with torch.profiler.record_function("floorflow_training_epoch"):
                yield profiler
        else:  # pragma: no cover
            yield profiler
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(target / "trace.json"))
    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "key": event.key,
                "calls": int(event.count),
                "self_cpu_us": float(event.self_cpu_time_total),
                "self_cuda_us": float(
                    getattr(event, "self_device_time_total", 0.0)
                ),
            }
        )
    rows.sort(
        key=lambda value: (
            value["self_cuda_us"],
            value["self_cpu_us"],
        ),
        reverse=True,
    )
    (target / "summary.json").write_text(
        json.dumps(
            {
                "capture": {
                    "wait_steps": 1,
                    "warmup_steps": 1,
                    "active_steps": max(1, int(active_steps)),
                    "record_shapes": False,
                    "profile_memory": False,
                },
                "operators": rows[:100],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = ["profile_training_epoch"]
