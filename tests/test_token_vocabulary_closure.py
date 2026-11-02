from __future__ import annotations

from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.objectives.token_vocabulary_closure import (
    close_token_vocabulary,
)


def test_vocabulary_closure_preserves_existing_ids() -> None:
    base = FloorplanTokenVocabulary(("A|1", "B|2"))
    closed, added = close_token_vocabulary(
        base,
        ("C|3", "A|1", "B|4"),
    )
    assert closed.tokens[: len(base.tokens)] == base.tokens
    assert added == ("B|4", "C|3")
    assert closed.index("A|1") == base.index("A|1")
    assert closed.index("B|2") == base.index("B|2")
