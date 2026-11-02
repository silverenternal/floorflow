"""Validation readouts for the optional token generation branch."""

from __future__ import annotations

from floorflow.evaluation.token_readout import (
    constrained_generation_metrics,
    token_teacher_forced_metrics,
)
from floorflow.training.floorplan_token_training_runtime import (
    FloorplanTokenTrainingRuntime,
)


def token_validation_readout(
    model,
    output: dict,
    batch: dict,
    runtime: FloorplanTokenTrainingRuntime,
    *,
    collect_generation: bool,
) -> tuple[dict[str, float], dict[str, float] | None]:
    if not runtime.enabled or "token_logits" not in output:
        return {}, None
    metrics = token_teacher_forced_metrics(
        output["token_logits"],
        batch["token_ids"],
        batch["token_loss_mask"],
        batch["token_stage_ids"],
        runtime.vocabulary,
        grammar_mask=batch.get("token_grammar_mask"),
    )
    if not collect_generation or runtime.config.generation_eval_samples <= 0:
        return metrics, None
    sample_count = min(
        runtime.config.generation_eval_samples,
        int(batch["token_ids"].shape[0]),
    )
    generated_values: dict[str, list[float]] = {}
    for row in range(sample_count):
        target_length = int(batch["token_length"][row])
        generated = constrained_generation_metrics(
            model.token_head,
            output["token_context_tokens"][row : row + 1],
            output["token_context_valid"][row : row + 1],
            runtime.vocabulary,
            batch["token_ids"][row],
            max_new_tokens=runtime.generation_budget(target_length),
            efficiency_budget=runtime.config.generation_efficiency_budget,
        )
        for key, value in generated.items():
            generated_values.setdefault(key, []).append(float(value))
    return metrics, {
        key: sum(values) / len(values)
        for key, values in generated_values.items()
    }


__all__ = ["token_validation_readout"]
