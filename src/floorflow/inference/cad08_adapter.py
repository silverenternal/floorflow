"""Faithful CPU adapters for existing decoded O outputs and independent evidence."""

from __future__ import annotations

import os

import numpy as np


def forbid_neural_execution() -> None:
    """Fail before any Module invocation or checkpoint deserialization."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    def blocked(*args, **kwargs):
        raise RuntimeError("CAD08 saved-cache CPU only: neural execution and weight loading are forbidden")

    torch.nn.Module.__call__ = blocked
    torch.nn.Module._call_impl = blocked
    torch.nn.Module.load_state_dict = blocked
    torch.load = blocked
    torch.jit.load = blocked
    torch.set_num_threads(1)


def compose_visible(layers: dict) -> np.ndarray:
    labels = np.where(layers["furniture"] != 0, layers["furniture"], layers["room_labels"])
    labels = np.where(layers["wall"] != 0, 1, labels)
    return (np.where(layers["opening"] != 0, layers["opening"], labels) * layers["footprint"]).astype(np.uint8)


def joint_presence_type(presence: np.ndarray, type_logits: np.ndarray) -> dict:
    """Preserve foreground mass separately from conditional type probabilities."""
    logits = type_logits.astype(np.float64)
    exponents = np.exp(logits - logits.max(axis=0, keepdims=True))
    conditional = exponents / exponents.sum(axis=0, keepdims=True)
    return {
        "presence": presence,
        "conditional_type": conditional,
        "joint_type": conditional * presence[None],
        "background": 1 - presence,
    }


def decode_saved_native(native: dict, footprint: np.ndarray, taxonomy) -> dict:
    """Replay scalar decoding only; consume saved decoder-grid probabilities."""
    inside = footprint.astype(bool)
    room_ids = np.asarray(taxonomy.room_ids)
    indices = native["room_logits"].argmax(axis=0)
    present = native["program"].astype(bool)
    rooms = (room_ids[indices] * present[indices] * inside).astype(np.uint8)
    opening_ids = np.asarray(taxonomy.opening_ids)
    opening = np.where(
        (native["opening_probability_decoder"] >= 0.5) & native["opening_candidate_support"],
        opening_ids[native["opening_type_logits_decoder"].argmax(axis=0)],
        0,
    ).astype(np.uint8)
    wall = ((native["wall_probability_decoder"] >= 0.6) & inside & (opening == 0)).astype(np.uint8)
    furniture_ids = np.asarray(taxonomy.furniture_ids)
    furniture = (
        np.where(
            native["furniture_probability_decoder"] >= 0.5,
            furniture_ids[native["furniture_type_logits_decoder"].argmax(axis=0)],
            0,
        )
        * inside
    ).astype(np.uint8)
    value = {
        "room_labels": rooms,
        "opening": opening,
        "wall": wall,
        "furniture": furniture,
        "footprint": footprint.astype(np.uint8),
        "room_support": (
            np.reciprocal(np.float32(1) + np.exp(-native["support_logits_auxiliary"][0].astype(np.float32)))
            >= np.float32(0.5)
        ).astype(np.uint8),
    }
    value["prediction"] = compose_visible(value)
    return value


def control_initialization(controls: dict, footprint: np.ndarray, taxonomy) -> dict:
    """Existing direct-control baseline decoder, not a neural prediction."""
    import torch

    from floorflow.evaluation.control_baseline import control_layers

    tensors = {k: torch.from_numpy(np.asarray(v).copy()) for k, v in controls.items()}
    rooms, wall, opening, furniture = control_layers(tensors, torch.from_numpy(footprint.copy()).bool(), taxonomy)
    value = {
        k: v.numpy().astype(np.uint8)
        for k, v in zip(("room_labels", "wall", "opening", "furniture"), (rooms, wall, opening, furniture), strict=True)
    }
    value["footprint"] = footprint.astype(np.uint8)
    value["prediction"] = compose_visible(value)
    value["room_support"] = np.isin(value["room_labels"], taxonomy.room_ids).astype(np.uint8)
    return value


def compare_decoded(saved: dict, reconstructed: dict) -> dict:
    fields = {}
    for key, value in reconstructed.items():
        if key not in saved:
            fields[key] = {"status": "unknown", "missing_field": key}
        else:
            changed = int(np.count_nonzero(saved[key] != value))
            fields[key] = {"status": "pass" if changed == 0 else "fail", "changed_pixels": changed}
    return {
        "status": "fail"
        if any(v["status"] == "fail" for v in fields.values())
        else "unknown"
        if any(v["status"] == "unknown" for v in fields.values())
        else "pass",
        "fields": fields,
    }
