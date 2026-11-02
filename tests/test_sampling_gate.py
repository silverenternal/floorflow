from floorflow.inference.sampling_policy import SamplingPolicy, validate_primary_report


def test_valid_primary_sampling_report() -> None:
    report = {
        "best_of_n_used": False,
        "sampling_policy": SamplingPolicy(purpose="distribution", candidates_per_condition=8).manifest(),
    }
    valid, errors = validate_primary_report(report)
    assert valid
    assert not errors


def test_ceiling_report_is_rejected_as_primary() -> None:
    report = {
        "best_of_n_used": True,
        "sampling_policy": SamplingPolicy(
            purpose="diagnostic_ceiling",
            candidates_per_condition=4,
        ).manifest(),
    }
    valid, errors = validate_primary_report(report)
    assert not valid
    assert errors
