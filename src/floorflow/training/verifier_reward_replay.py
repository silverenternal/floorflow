"""Asynchronous, target-keyed verifier reward replay for GPU training."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.objectives.token_reward import token_candidate_reward
from floorflow.training.rplan_generator_contract import contract_from_output

REWARD_REPLAY_VERSION = "floorflow-verifier-reward-replay-v1"


@dataclass(frozen=True)
class CachedVerifierReward:
    reward: float
    refreshed_at: int
    source: str
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierRewardBatch:
    rewards: torch.Tensor
    valid: torch.Tensor


def _to_cpu(value):
    if torch.is_tensor(value):
        dtype = torch.float32 if value.is_floating_point() else None
        return value.detach().to(
            device="cpu",
            dtype=dtype,
            copy=True,
        )
    if isinstance(value, list):
        return [str(item) for item in value]
    return value


def _row(batch: dict, output: dict, row: int) -> tuple[dict, dict]:
    batch_size = int(batch["boundary_mask"].shape[0])

    def select(values: dict) -> dict:
        selected = {}
        for key, value in values.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                selected[key] = value[row : row + 1]
            elif isinstance(value, list) and len(value) == batch_size:
                selected[key] = [value[row]]
            else:
                selected[key] = value
        return selected

    return select(batch), select(output)


def _compute_reward(
    batch: dict,
    output: dict,
    taxonomy,
    tolerances: FEVPTolerances,
) -> CachedVerifierReward:
    contract = contract_from_output(output, batch, taxonomy, tolerances)
    expected_rooms = int(batch["room_exists"][0].sum())
    reward, components = token_candidate_reward(
        contract,
        {},
        expected_room_count=expected_rooms,
    )
    return CachedVerifierReward(
        reward=float(reward),
        refreshed_at=-1,
        source="asynchronous",
        components={key: float(value) for key, value in components.items()},
    )


class VerifierRewardReplay:
    """Replay verified returns without running CPU verifiers in the optimizer step."""

    def __init__(
        self,
        taxonomy,
        tolerances: FEVPTolerances,
        *,
        namespace: str,
        refresh_interval: int = 20,
        max_age_steps: int = 200,
        max_workers: int = 1,
        replay_path: str | Path | None = None,
        synchronous: bool = False,
    ) -> None:
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if max_age_steps <= 0:
            raise ValueError("max_age_steps must be positive")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.taxonomy = taxonomy
        self.tolerances = tolerances
        self.namespace = str(namespace)
        self.refresh_interval = int(refresh_interval)
        self.max_age_steps = int(max_age_steps)
        self.synchronous = bool(synchronous)
        self._executor = None if self.synchronous else ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="floorflow-verifier",
        )
        self._lock = threading.Lock()
        self._entries: dict[str, CachedVerifierReward] = {}
        self._pending: dict[str, Future[CachedVerifierReward]] = {}
        self._errors: dict[str, str] = {}
        self._submitted = 0
        self._completed = 0
        if replay_path is not None:
            self.load_replay(replay_path)

    def _key(
        self,
        batch: dict,
        row: int,
    ) -> str:
        keys = batch.get("verifier_target_keys")
        if not isinstance(keys, list) or row >= len(keys):
            raise ValueError("batch is missing verifier_target_keys")
        return f"{self.namespace}:{keys[row]}"

    def _finish(
        self,
        key: str,
        reward: CachedVerifierReward,
    ) -> None:
        with self._lock:
            self._entries[key] = reward
            self._pending.pop(key, None)
            self._completed += 1

    def _fail(self, key: str, error: BaseException) -> None:
        with self._lock:
            self._pending.pop(key, None)
            self._errors[key] = f"{type(error).__name__}: {error}"

    def maybe_refresh(
        self,
        output: dict,
        batch: dict,
        step: int,
    ) -> None:
        """Schedule stale rows; never wait for CPU verification in the train loop."""

        if int(step) % self.refresh_interval != 0:
            return
        with self._lock:
            if self._errors:
                key, error = next(iter(self._errors.items()))
                raise RuntimeError(f"verifier reward failed for {key}: {error}")
        batch_size = int(batch["boundary_mask"].shape[0])
        for row in range(batch_size):
            key = self._key(batch, row)
            with self._lock:
                entry = self._entries.get(key)
                if key in self._pending:
                    continue
                if entry is not None and entry.refreshed_at >= 0:
                    age = int(step) - entry.refreshed_at
                    if age < self.refresh_interval:
                        continue
            row_batch, row_output = _row(batch, output, row)
            cpu_batch = {key: _to_cpu(value) for key, value in row_batch.items()}
            cpu_output = {
                key: _to_cpu(value) for key, value in row_output.items()
            }

            def task(
                task_batch=cpu_batch,
                task_output=cpu_output,
                refresh_step=int(step),
            ) -> CachedVerifierReward:
                value = _compute_reward(
                    task_batch,
                    task_output,
                    self.taxonomy,
                    self.tolerances,
                )
                return CachedVerifierReward(
                    reward=value.reward,
                    refreshed_at=refresh_step,
                    source=value.source,
                    components=value.components,
                )

            with self._lock:
                self._submitted += 1
                if self.synchronous:
                    try:
                        result = task()
                    except (
                        ArithmeticError,
                        LookupError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        self._pending.pop(key, None)
                        self._errors[key] = f"{type(error).__name__}: {error}"
                    else:
                        self._entries[key] = result
                        self._completed += 1
                    continue
                future = self._executor.submit(task)
                self._pending[key] = future
            future.add_done_callback(
                lambda value, cache_key=key: self._resolve(cache_key, value)
            )

    def _resolve(
        self,
        key: str,
        future: Future[CachedVerifierReward],
    ) -> None:
        try:
            reward = future.result()
        except (
            ArithmeticError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            self._fail(key, error)
            return
        self._finish(key, reward)

    def rewards(
        self,
        batch: dict,
        step: int,
        *,
        device: torch.device,
    ) -> VerifierRewardBatch:
        batch_size = int(batch["boundary_mask"].shape[0])
        values = [0.0] * batch_size
        valid = [False] * batch_size
        for row in range(batch_size):
            key = self._key(batch, row)
            with self._lock:
                entry = self._entries.get(key)
            if entry is None:
                continue
            age = int(step) - entry.refreshed_at
            if entry.refreshed_at >= 0 and age > self.max_age_steps:
                continue
            values[row] = entry.reward
            valid[row] = True
        return VerifierRewardBatch(
            rewards=torch.tensor(values, dtype=torch.float32, device=device),
            valid=torch.tensor(valid, dtype=torch.bool, device=device),
        )

    def load_replay(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != REWARD_REPLAY_VERSION:
            raise ValueError("unsupported verifier reward replay version")
        if str(payload.get("namespace")) != self.namespace:
            raise ValueError("verifier replay namespace mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise TypeError("verifier replay entries must be a mapping")
        with self._lock:
            for target_key, value in entries.items():
                reward = float(value["reward"])
                components = {
                    str(name): float(score)
                    for name, score in value.get("components", {}).items()
                }
                self._entries[str(target_key)] = CachedVerifierReward(
                    reward=reward,
                    refreshed_at=-1,
                    source="offline_replay",
                    components=components,
                )

    def write_replay(self, path: str | Path) -> None:
        with self._lock:
            entries = {
                key: {
                    "reward": value.reward,
                    "components": value.components,
                    "source": value.source,
                }
                for key, value in self._entries.items()
            }
        payload = {
            "version": REWARD_REPLAY_VERSION,
            "namespace": self.namespace,
            "entries": entries,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def statistics(self) -> dict[str, object]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "cached": len(self._entries),
                "pending": len(self._pending),
                "errors": len(self._errors),
                "last_error": next(iter(self._errors.values()), None),
            }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None


__all__ = [
    "REWARD_REPLAY_VERSION",
    "CachedVerifierReward",
    "VerifierRewardBatch",
    "VerifierRewardReplay",
]
