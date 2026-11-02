"""Extend a frozen token vocabulary without reindexing existing tokens."""

from __future__ import annotations

from collections.abc import Iterable

from floorflow.objectives.token_grammar import FloorplanTokenVocabulary


def close_token_vocabulary(
    base: FloorplanTokenVocabulary,
    extra_tokens: Iterable[str],
) -> tuple[FloorplanTokenVocabulary, tuple[str, ...]]:
    """Append unseen tokens after the base vocabulary in stable order."""

    existing = set(base.tokens)
    added = tuple(
        sorted(
            {
                str(token)
                for token in extra_tokens
                if str(token) not in existing
            }
        )
    )
    return (
        FloorplanTokenVocabulary(
            tokens=(*base.tokens, *added),
            pad_token=base.pad_token,
            unk_token=base.unk_token,
            version=base.version,
        ),
        added,
    )


__all__ = ["close_token_vocabulary"]
