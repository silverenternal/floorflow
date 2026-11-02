"""Shared structural membership masks for chip hierarchy states."""

from __future__ import annotations

import torch


def structural_membership(constraints: torch.Tensor) -> torch.Tensor:
    """Return cluster(4), MIB(1), boundary(4) membership masks."""

    cluster = constraints[..., 3].long()
    mib = constraints[..., 2].long()
    boundary = constraints[..., 4].long()
    masks = [cluster == group for group in range(1, 5)] + [mib == 1]
    masks += [(boundary & bit).ne(0) for bit in (1, 2, 4, 8)]
    return torch.stack(masks, dim=-1)


__all__ = ["structural_membership"]
