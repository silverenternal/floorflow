import torch

from floorflow.evaluation.style_provenance import audit_style_provenance


def _bank() -> dict:
    return {
        "num_styles": 2,
        "inference_rule": "style id is supplied by user or deployment",
        "assignments": {"train": {"a": 0, "b": 1}},
    }


def test_style_provenance_accepts_train_only_explicit_bank() -> None:
    checkpoint = {
        "config": {"num_styles": 2},
        "pixel_head": {
            "style_pixel_residual.class_bias.weight": torch.ones(2, 3)
        },
    }
    report = audit_style_provenance(
        _bank(),
        train_names=["a", "b"],
        val_names=["c"],
        test_names=["d"],
        checkpoint=checkpoint,
        run_manifest={"condition_mode": "contour_only", "num_styles": 2},
    )
    assert report["status"] == "pass"
    assert all(report["checks"].values())


def test_style_provenance_rejects_split_leak_and_nonfinite_weights() -> None:
    bank = _bank()
    bank["assignments"]["val"] = {"c": 0}
    checkpoint = {
        "config": {"num_styles": 3},
        "pixel_head": {
            "style_pixel_residual.class_bias.weight": torch.tensor(
                [[float("nan"), 0.0]]
            )
        },
    }
    report = audit_style_provenance(
        bank,
        train_names=["a", "b"],
        val_names=["c"],
        test_names=["d"],
        checkpoint=checkpoint,
        run_manifest={"condition_mode": "target_control", "num_styles": 2},
    )
    assert report["status"] == "fail"
    assert not report["checks"]["only_train_split_has_assignments"]
    assert not report["checks"]["style_weights_finite"]
