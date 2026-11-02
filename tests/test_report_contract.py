from floorflow.evaluation.report_contract import split_report_fields


def test_validation_report_cannot_mark_test_opened() -> None:
    assert split_report_fields("val") == {
        "split": "val",
        "test_opened": False,
    }


def test_test_report_marks_test_opened() -> None:
    assert split_report_fields("test") == {
        "split": "test",
        "test_opened": True,
    }
