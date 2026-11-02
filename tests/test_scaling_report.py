from floorflow.evaluation.scaling_report import build_scaling_report


def test_scaling_report_finds_knee_and_val_sweet_spot() -> None:
    report = build_scaling_report(
        [
            {
                "train_size": 512,
                "val_size": 64,
                "room_macro_iou": 0.20,
                "fpeval_wall_f1": 0.40,
                "fpeval_opening_f1": 0.10,
            },
            {
                "train_size": 1024,
                "val_size": 64,
                "room_macro_iou": 0.201,
                "fpeval_wall_f1": 0.401,
                "fpeval_opening_f1": 0.101,
            },
            {
                "train_size": 2048,
                "val_size": 64,
                "room_macro_iou": 0.2015,
                "fpeval_wall_f1": 0.4015,
                "fpeval_opening_f1": 0.1015,
            },
        ],
        [
            {
                "val_size": 32,
                "composite": 0.25,
                "metric_std": 0.02,
                "rank_stability": 0.98,
            },
            {
                "val_size": 64,
                "composite": 0.235,
                "metric_std": 0.005,
                "rank_stability": 0.995,
            },
        ],
    )
    assert report["train_knee"] == 512
    assert report["still_data_limited"] is False
    assert report["sweet_spot"]["train_size"] == 512
    assert report["sweet_spot"]["val_size"] == 64


def test_scaling_report_flags_data_limited_curve() -> None:
    report = build_scaling_report(
        [
            {"train_size": 512, "room_macro_iou": 0.1},
            {"train_size": 1024, "room_macro_iou": 0.2},
            {"train_size": 2048, "room_macro_iou": 0.3},
        ]
    )
    assert report["still_data_limited"] is True
    assert report["train_knee"] == 2048
