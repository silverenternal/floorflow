"""Encode observable remaining-operation context for interaction policies."""

from __future__ import annotations

from collections import Counter

from floorflow.data.constraint_action_dataset import ACTION_NAMES

SEQUENCE_FEATURES = 2 + 2 * len(ACTION_NAMES)


def _mode_operation(groups: list[dict]) -> str | None:
    operations = [
        group["operation"]
        for group in groups
        if group is not None
    ]
    if not operations:
        return None
    return Counter(operations).most_common(1)[0][0]


def _operation_features(operation: str | None) -> list[float]:
    if operation is None:
        return [0.0] * len(ACTION_NAMES)
    features = [0.0] * len(ACTION_NAMES)
    features[ACTION_NAMES.index(operation)] = 1.0
    return features


def enrich_sequence_context(groups: list[dict]) -> dict:
    """Attach current depth and future typed-action plan to every candidate."""

    sequences = {}
    for group in groups:
        sequences.setdefault(group["sequence_id"], []).append(group)
    augmented = 0
    for sequence_groups in sequences.values():
        by_step = {}
        children = {}
        for group in sequence_groups:
            by_step[group["step"]] = max(
                by_step.get(group["step"], 0),
                group["step"],
            )
            if group["parent_candidate_id"] is not None:
                children[group["parent_candidate_id"]] = group
        depth = max(by_step) + 1
        for group in sequence_groups:
            step = group["step"]
            candidate_ids = {
                candidate["candidate_id"] for candidate in group["candidates"]
            }
            direct_children = [
                children[candidate_id]
                for candidate_id in candidate_ids
                if candidate_id in children
            ]
            next_operation = _mode_operation(direct_children)
            grandchildren = []
            for child in direct_children:
                for candidate in child["candidates"]:
                    grandchild = children.get(candidate["candidate_id"])
                    if grandchild is not None:
                        grandchildren.append(grandchild)
            second_operation = _mode_operation(grandchildren)
            features = [
                step / max(depth - 1, 1),
                (depth - 1 - step) / max(depth - 1, 1),
                *_operation_features(next_operation),
                *_operation_features(second_operation),
            ]
            for candidate in group["candidates"]:
                candidate["sequence_features"] = features
                augmented += 1
    return {
        "sequences": len(sequences),
        "groups": len(groups),
        "candidates_augmented": augmented,
        "feature_dim": SEQUENCE_FEATURES,
    }


__all__ = ["SEQUENCE_FEATURES", "enrich_sequence_context"]
