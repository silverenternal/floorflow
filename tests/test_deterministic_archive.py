"""Tests for deterministic evidence archives."""

from __future__ import annotations

from pathlib import Path

from floorflow.evaluation.deterministic_archive import build_deterministic_tar_gz


def test_deterministic_archive_rebuilds_to_same_hash(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_text("alpha\n", encoding="utf-8")
    (root / "nested" / "b.py").write_text("print('beta')\n", encoding="utf-8")

    first = build_deterministic_tar_gz(root, tmp_path / "first.tar.gz")
    second = build_deterministic_tar_gz(root, tmp_path / "second.tar.gz")

    assert first == second
    assert (tmp_path / "first.tar.gz").read_bytes() == (
        tmp_path / "second.tar.gz"
    ).read_bytes()


def test_archive_ignores_source_mtime(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    source = root / "a.txt"
    source.write_text("alpha\n", encoding="utf-8")
    first = build_deterministic_tar_gz(root, tmp_path / "first.tar.gz")
    source.touch()
    second = build_deterministic_tar_gz(root, tmp_path / "second.tar.gz")

    assert first == second
