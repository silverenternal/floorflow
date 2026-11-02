from types import SimpleNamespace

from floorflow.training.distribution_manifest import build_joint_run_manifest


def test_joint_manifest_preserves_reproducibility_metadata() -> None:
    args = SimpleNamespace(
        dataset="CubiCasa",
        skv4_root="data/skv4",
        flow_matcher="torchcfm",
        time_sampling="uniform",
        time_sampling_mean=0.5,
        time_sampling_std=0.1,
        minibatch_ot=True,
        ot_token_grid=8,
        flow_sigma=0.0,
        flow_endpoint_weight=1.0,
        flow_change_weight=0.0,
        batch_size=4,
        gradient_accumulation_steps=2,
        element_warmup_updates=10,
        element_warmup_lr=0.001,
        selection="ema",
        gradient_balance_interval=5,
        gradient_balance_min_scale=0.1,
        gradient_balance_max_scale=10.0,
        loss_metric_patience=3,
        loss_metric_drop=0.1,
        validation_metric_drop=0.2,
        family_iou_weight=1.0,
        room_iou_weight=1.0,
        tversky_weight=1.0,
        boundary_weight=0.1,
        positive_weight_clip=20.0,
        dynamic_route_weight=1.0,
        style_manifest=None,
        style_contrast_weight=0.0,
        style_contrast_margin=0.1,
        style_semantic_weight=0.0,
        style_orthogonality_weight=0.0,
        style_dropout=0.0,
        style_lr=0.001,
        parent_reliability_init=0.0,
    )
    model = SimpleNamespace(
        element_mask_version="v2",
        num_styles=8,
        style_rank=4,
        hierarchy_support_strength=1.0,
        element_blend_init=0.0,
        element_feedback_init=0.0,
        dynamic_head_variant="default",
    )
    runtime = SimpleNamespace(manifest=lambda: {"device": "cuda"})
    convergence = SimpleNamespace(manifest=lambda: {"base_lr": 0.001})

    manifest = build_joint_run_manifest(
        args=args,
        model=model,
        runtime=runtime,
        convergence=convergence,
        flow_preset="default",
        flow_backend="torchcfm",
        flow_backend_version="2.0",
        batches_per_epoch=10,
        updates_per_epoch=5,
        total_updates=60,
        joint_updates=50,
    )

    assert manifest["effective_batch_size"] == 8
    assert manifest["runtime"] == {"device": "cuda"}
    assert manifest["condition_mode"] == "contour_only"
