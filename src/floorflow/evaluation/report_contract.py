"""Shared report metadata contracts."""

from __future__ import annotations


def split_report_fields(split: str) -> dict[str, str | bool]:
    """Return protocol fields that must follow the evaluated split."""

    return {
        "split": str(split),
        "test_opened": str(split) == "test",
    }


__all__ = ["split_report_fields"]
