"""Deterministic archive helpers for reproducible evidence packaging."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
from pathlib import Path

FIXED_MTIME = 0
FIXED_UID = 0
FIXED_GID = 0


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_mode(path: Path) -> int:
    if path.suffix in {".py", ".sh"}:
        return 0o755
    return 0o644


def _tar_info(root: Path, path: Path) -> tarfile.TarInfo:
    relative = path.relative_to(root)
    info = tarfile.TarInfo(name=f"{root.name}/{relative.as_posix()}")
    info.size = path.stat().st_size
    info.mtime = FIXED_MTIME
    info.mode = _normalized_mode(path)
    info.uid = FIXED_UID
    info.gid = FIXED_GID
    info.uname = ""
    info.gname = ""
    return info


def build_deterministic_tar_gz(root: Path, archive: Path) -> str:
    """Build a sorted, metadata-normalized tar.gz and return its digest."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=FIXED_MTIME,
        ) as compressed,
        tarfile.open(
            mode="w",
            fileobj=compressed,
            format=tarfile.PAX_FORMAT,
        ) as handle,
    ):
        for path in files:
            with path.open("rb") as source:
                handle.addfile(_tar_info(root, path), source)
    return sha256_file(archive)


__all__ = ["build_deterministic_tar_gz", "sha256_file"]
