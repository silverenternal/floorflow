"""Paired vector-only, teacher-token, and generated-token validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.evaluation.space_syntax import (
    adjacency_from_logits,
    space_syntax_metrics,
)
from floorflow.inference.floorplan_token_generation import (
    TokenGenerationBudgetError,
    generate_floorplan_tokens,
)
from floorflow.inference.junction_face_readout import junction_face_candidate
from floorflow.inference.token_beam_search import beam_search_floorplan_tokens
from floorflow.inference.token_graph_projection import (
    project_wall_endpoints_to_token_junctions,
)
from floorflow.inference.typed_state_machine import select_scored_state
from floorflow.objectives.token_grammar import stage_ids_for_tokens
from floorflow.objectives.token_reward import token_candidate_reward
from floorflow.representation.unified_floorplan_tokens import FloorplanToken
from floorflow.training.floorplan_token_training_runtime import (
    FloorplanTokenTrainingRuntime,
)
from floorflow.training.rplan_generator_contract import contract_from_output


def _slice_batch(batch: dict, row: int, batch_size: int) -> dict:
    output = {}
    for key, value in batch.items():
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            output[key] = value[row : row + 1]
        else:
            output[key] = value
    return output


def _slice_output(output: dict, row: int, batch_size: int) -> dict:
    sliced = {}
    for key, value in output.items():
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            sliced[key] = value[row : row + 1]
        else:
            sliced[key] = value
    return sliced


def _candidate_score(
    output: dict,
    contract: dict,
    taxonomy,
    *,
    expected_rooms: int,
    sequence_score: float = 0.0,
    generated_count: int = 1,
) -> tuple[float, dict, dict[str, float]]:
    predicted_rooms = (output["room_exists"].sigmoid() >= 0.5)[0].cpu().numpy()
    adjacency = adjacency_from_logits(output["room_adjacency_logits"][0])
    adjacency[~predicted_rooms, :] = False
    adjacency[:, ~predicted_rooms] = False
    labels = [
        taxonomy.class_names[taxonomy.room_ids[int(local_class)]]
        if bool(predicted_rooms[index])
        and 0 <= int(local_class) < len(taxonomy.room_ids)
        else "room:Unknown"
        for index, local_class in enumerate(
            output["room_class"][0].argmax(dim=-1)
        )
    ]
    syntax = space_syntax_metrics(
        adjacency[np.ix_(predicted_rooms, predicted_rooms)],
        [
            label
            for label, valid in zip(labels, predicted_rooms, strict=True)
            if valid
        ],
    )
    reward, components = token_candidate_reward(
        contract,
        syntax,
        expected_room_count=expected_rooms,
    )
    score = reward + 0.01 * sequence_score / max(1, generated_count)
    return score, {**syntax, **components, "reward_total": reward}, components


@torch.no_grad()
def token_consumer_validation(
    model,
    batch: dict,
    runtime: FloorplanTokenTrainingRuntime,
    taxonomy,
    tolerances: FEVPTolerances,
    *,
    max_samples: int,
    immutable_reference: Sequence[FloorplanToken] | None = None,
    immutable_mask: Sequence[bool] | None = None,
) -> tuple[dict[str, float], dict[str, list[dict[str, object]]]]:
    if not runtime.enabled or max_samples <= 0:
        return {}, {}
    base = model(
        batch["boundary_mask"],
        batch["boundary_valid"],
        batch["boundary_points"],
        batch["entrance_points"],
        batch["entrance_valid"],
    )
    teacher = model.refine_with_token_ids(
        base,
        batch["token_ids"],
        batch["token_mask"],
        batch["token_stage_ids"],
    )
    batch_size = int(batch["boundary_mask"].shape[0])
    sample_count = min(int(max_samples), batch_size)
    totals: dict[str, list[float]] = {}
    details: dict[str, list[dict[str, object]]] = {
        "vector": [],
        "teacher": [],
        "generated": [],
    }
    for row in range(sample_count):
        row_batch = _slice_batch(batch, row, batch_size)
        vector_output = _slice_output(base, row, batch_size)
        teacher_output = project_wall_endpoints_to_token_junctions(
            _slice_output(teacher, row, batch_size)
        )
        vector_contract = contract_from_output(
            vector_output,
            row_batch,
            taxonomy,
            tolerances,
        )
        teacher_contract = contract_from_output(
            teacher_output,
            row_batch,
            taxonomy,
            tolerances,
        )
        details["vector"].append(vector_contract)
        details["teacher"].append(teacher_contract)
        prompt_tokens = tuple(
            FloorplanToken.parse(runtime.vocabulary.decode(int(token_id)))
            for token_id in batch["token_ids"][row, :5].cpu().tolist()
        )
        target_length = int(batch["token_length"][row])
        generated_contract = {"contract_valid": False}
        completed = False
        generated_count = 0
        candidates = []
        try:
            if runtime.config.beam_width > 1:
                candidates = beam_search_floorplan_tokens(
                    model.token_head,
                    vector_output["token_context_tokens"],
                    vector_output["token_context_valid"],
                    runtime.vocabulary,
                    prompt_tokens,
                    max_new_tokens=runtime.generation_budget(target_length),
                    beam_width=runtime.config.beam_width,
                    immutable_reference=immutable_reference,
                    immutable_mask=immutable_mask,
                )
                completed = bool(candidates)
            else:
                generated, score = generate_floorplan_tokens(
                    model.token_head,
                    vector_output["token_context_tokens"],
                    vector_output["token_context_valid"],
                    runtime.vocabulary,
                    prompt_tokens,
                    max_new_tokens=runtime.generation_budget(target_length),
                    immutable_reference=immutable_reference,
                    immutable_mask=immutable_mask,
                )
                candidates = [(generated, float(sum(score)))]
                completed = generated[-1].kind == "END"
        except TokenGenerationBudgetError as error:
            generated = error.tokens
            generated_count = max(0, len(generated) - len(prompt_tokens))
        if candidates:
            candidate_outputs = {"vector": vector_output}
            candidate_scores = {}
            candidate_meta = {}
            expected_rooms = int(batch["room_exists"][row].sum())
            vector_score, vector_detail, _vector_reward = _candidate_score(
                vector_output,
                vector_contract,
                taxonomy,
                expected_rooms=expected_rooms,
            )
            candidate_scores["vector"] = vector_score
            candidate_meta["vector"] = {
                "contract": vector_contract,
                "syntax": vector_detail,
                "variant": "vector",
                "reward": vector_detail["reward_total"],
                "reward_components": {
                    key: value
                    for key, value in vector_detail.items()
                    if key.startswith("reward_")
                },
            }
            for generated, sequence_score in candidates:
                if generated[-1].kind != "END":
                    continue
                generated_count = max(0, len(generated) - len(prompt_tokens))
                ids = torch.tensor(
                    [[runtime.vocabulary.index(token) for token in generated]],
                    dtype=torch.long,
                    device=batch["token_ids"].device,
                )
                stages = torch.tensor(
                    [stage_ids_for_tokens(generated)],
                    dtype=torch.long,
                    device=batch["token_ids"].device,
                )
                generated_output = model.refine_with_token_ids(
                    vector_output,
                    ids,
                    torch.ones_like(ids, dtype=torch.bool),
                    stages,
                )
                generated_output = project_wall_endpoints_to_token_junctions(
                    generated_output
                )
                variants = [("refined", generated_output)]
                if "codetree_polygons" in generated_output:
                    codetree_output = dict(generated_output)
                    codetree_output["room_polygons"] = generated_output.get(
                        "codetree_polygons_aligned",
                        generated_output["codetree_polygons"],
                    )
                    variants.append(("codetree", codetree_output))
                face_output = junction_face_candidate(generated_output)
                if face_output is not None:
                    variants.append(("junction_faces", face_output))
                for variant_name, variant_output in variants:
                    contract = contract_from_output(
                        variant_output,
                        row_batch,
                        taxonomy,
                        tolerances,
                    )
                    score, detail, reward_components = _candidate_score(
                        variant_output,
                        contract,
                        taxonomy,
                        expected_rooms=expected_rooms,
                        sequence_score=sequence_score,
                        generated_count=generated_count,
                    )
                    key = f"{variant_name}_{len(candidate_outputs)}"
                    candidate_outputs[key] = variant_output
                    candidate_scores[key] = score
                    candidate_meta[key] = {
                        "contract": contract,
                        "syntax": detail,
                        "variant": variant_name,
                        "reward": detail["reward_total"],
                        "reward_components": reward_components,
                    }
            if len(candidate_outputs) > 1:
                generated_output, selection = select_scored_state(
                    candidate_outputs,
                    candidate_scores,
                    fallback="vector",
                )
                selected_name = str(selection["selected"])
                selected = candidate_meta[selected_name]
                generated_contract = selected["contract"]
                syntax = selected["syntax"]
                selected_variant = selected["variant"]
                selected_reward = selected["reward"]
                full_reward_components = selected["reward_components"]
                totals.setdefault("token_consumer_beam_candidates", []).append(
                    float(len(candidate_outputs) - 1)
                )
                totals.setdefault(
                    "token_consumer_space_syntax_advantage",
                    [],
                ).append(float(syntax["public_integration_advantage"]))
                totals.setdefault(
                    "token_consumer_codetree_selected",
                    [],
                ).append(float(selected_variant == "codetree"))
                totals.setdefault(
                    "token_consumer_junction_faces_selected",
                    [],
                ).append(float(selected_variant == "junction_faces"))
                totals.setdefault(
                    "token_consumer_selected_reward",
                    [],
                ).append(float(selected_reward))
                for key, value in full_reward_components.items():
                    totals.setdefault(
                        f"token_consumer_{key}",
                        [],
                    ).append(float(value))
                rollback = selected_name == "vector"
                totals.setdefault(
                    "token_consumer_generated_rollback",
                    [],
                ).append(float(rollback))
            else:
                completed = False
        details["generated"].append(generated_contract)
        for branch, contract in (
            ("vector", vector_contract),
            ("teacher", teacher_contract),
            ("generated", generated_contract),
        ):
            totals.setdefault(
                f"token_consumer_{branch}_contract_valid",
                [],
            ).append(float(bool(contract.get("contract_valid", False))))
            totals.setdefault(
                f"token_consumer_{branch}_token_roundtrip",
                [],
            ).append(float(bool(contract.get("token_roundtrip_exact", False))))
            for metric_name in (
                "functional_watertight",
                "hole_count",
                "dangling_endpoint_rate",
                "boundary_collision_pixels",
                "outside_footprint_pixels",
            ):
                totals.setdefault(
                    f"token_consumer_{branch}_{metric_name}",
                    [],
                ).append(float(contract.get(metric_name, 0.0)))
            if contract.get("contract_valid"):
                expected_rooms = int(batch["room_exists"][row].sum())
                totals.setdefault(
                    f"token_consumer_{branch}_room_count_abs_error",
                    [],
                ).append(float(abs(int(contract["room_count"]) - expected_rooms)))
        totals.setdefault("token_consumer_generation_completed", []).append(
            float(completed)
        )
        totals.setdefault("token_consumer_generation_count", []).append(
            float(generated_count)
        )
        totals.setdefault(
            "token_consumer_generation_within_efficiency_budget",
            [],
        ).append(
            float(
                completed
                and generated_count <= runtime.config.generation_efficiency_budget
            )
        )
        totals.setdefault("token_consumer_target_length", []).append(
            float(target_length)
        )
    metrics = {
        key: sum(values) / len(values)
        for key, values in totals.items()
    }
    metrics["token_consumer_generated_minus_vector_valid"] = (
        metrics["token_consumer_generated_contract_valid"]
        - metrics["token_consumer_vector_contract_valid"]
    )
    metrics["token_consumer_teacher_minus_vector_valid"] = (
        metrics["token_consumer_teacher_contract_valid"]
        - metrics["token_consumer_vector_contract_valid"]
    )
    metrics["token_consumer_generated_validity_regression"] = max(
        0.0,
        metrics["token_consumer_vector_contract_valid"]
        - metrics["token_consumer_generated_contract_valid"],
    )
    metrics["token_consumer_teacher_validity_regression"] = max(
        0.0,
        metrics["token_consumer_vector_contract_valid"]
        - metrics["token_consumer_teacher_contract_valid"],
    )
    metrics["token_consumer_generated_watertight_regression"] = max(
        0.0,
        metrics["token_consumer_vector_functional_watertight"]
        - metrics["token_consumer_generated_functional_watertight"],
    )
    metrics["token_consumer_teacher_watertight_regression"] = max(
        0.0,
        metrics["token_consumer_vector_functional_watertight"]
        - metrics["token_consumer_teacher_functional_watertight"],
    )
    return metrics, details


def aggregate_token_consumer_reports(
    reports: list[tuple[dict[str, float], dict[str, list[dict[str, object]]]]],
    *,
    bootstrap_iterations: int = 1000,
) -> dict[str, float]:
    if not reports:
        return {}
    totals: dict[str, float] = {}
    sample_count = 0
    combined_details: dict[str, list[dict[str, object]]] = {
        "vector": [],
        "teacher": [],
        "generated": [],
    }
    for metrics, details in reports:
        count = len(details["vector"])
        sample_count += count
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * count
        for branch, values in combined_details.items():
            values.extend(details[branch])
    result = {key: value / max(1, sample_count) for key, value in totals.items()}
    result["token_consumer_cases"] = float(sample_count)
    valid_deltas = [
        float(bool(generated.get("contract_valid", False)))
        - float(bool(vector.get("contract_valid", False)))
        for generated, vector in zip(
            combined_details["generated"],
            combined_details["vector"],
            strict=True,
        )
    ]
    watertight_deltas = [
        float(generated.get("functional_watertight", 0.0))
        - float(vector.get("functional_watertight", 0.0))
        for generated, vector in zip(
            combined_details["generated"],
            combined_details["vector"],
            strict=True,
        )
    ]
    result["token_consumer_generated_minus_vector_valid"] = (
        float(np.mean(valid_deltas)) if valid_deltas else 0.0
    )
    result["token_consumer_generated_minus_vector_watertight"] = (
        float(np.mean(watertight_deltas)) if watertight_deltas else 0.0
    )
    for name, values in (
        ("valid", valid_deltas),
        ("watertight", watertight_deltas),
    ):
        low, high = _bootstrap_ci(values, bootstrap_iterations)
        result[f"token_consumer_generated_minus_vector_{name}_ci_low"] = low
        result[f"token_consumer_generated_minus_vector_{name}_ci_high"] = high
    return result


def _bootstrap_ci(values: list[float], iterations: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(1701)
    means = np.asarray(
        [
            generator.choice(array, size=len(array), replace=True).mean()
            for _ in range(max(1, int(iterations)))
        ]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


__all__ = [
    "aggregate_token_consumer_reports",
    "token_consumer_validation",
]
