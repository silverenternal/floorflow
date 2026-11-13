"""Versioned nested-content digest for CAD08 history records.

The schema tag is retained for byte compatibility with the original history
prototype. It is not a file hash and does not establish authorship or safety.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

SCHEMA = b"floorflow-oracle-bound-content-v1\0"


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "big") + payload


def _encode(value: Any) -> bytes:
    if value is None:
        return _frame(b"n", b"")
    if isinstance(value, np.generic):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.subdtype is not None:
            raise TypeError("Object and structured array dtypes are not supported")
        array = np.ascontiguousarray(value)
        shape = b"".join(int(dim).to_bytes(8, "big") for dim in value.shape)
        payload = _frame(b"d", value.dtype.str.encode("ascii")) + _frame(b"s", shape) + _frame(b"v", array.tobytes())
        return _frame(b"a", payload)
    if isinstance(value, bool):
        return _frame(b"b", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame(b"i", str(value).encode("ascii"))
    if isinstance(value, float):
        return _frame(b"f", struct.pack(">d", value))
    if isinstance(value, str):
        return _frame(b"s", value.encode("utf-8"))
    if isinstance(value, (list, tuple)):
        tag = b"l" if isinstance(value, list) else b"t"
        return _frame(tag, b"".join(_encode(item) for item in value))
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Content mapping keys must be strings")
        return _frame(b"m", b"".join(_encode(key) + _encode(value[key]) for key in sorted(value)))
    raise TypeError(f"Unsupported content type: {type(value).__name__}")


def content_digest(value: Any) -> str:
    """Hash a supported nested value, including array dtype, shape and bytes."""
    return hashlib.sha256(SCHEMA + _encode(value)).hexdigest()
