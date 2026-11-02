#!/usr/bin/env python3
"""Close an RPLAN vocabulary with train-only audit OOV tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.objectives.token_vocabulary_closure import (
    close_token_vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-vocabulary", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = FloorplanTokenVocabulary.read(args.base_vocabulary)
    audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    if audit.get("decision") != "pass":
        raise ValueError("train token audit must pass before vocabulary closure")
    extra = audit.get("unknown_token_counts", {})
    closed, added = close_token_vocabulary(base, extra)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = args.output_dir / "vocabulary.json"
    closed.write(vocabulary_path)
    report = {
        "schema_version": "rplan-token-vocabulary-closure-v1",
        "base_vocabulary": str(args.base_vocabulary),
        "base_vocabulary_sha256": _sha256(args.base_vocabulary),
        "train_audit": str(args.train_audit),
        "train_audit_sha256": _sha256(args.train_audit),
        "base_tokens": len(base.tokens),
        "added_tokens": len(added),
        "final_tokens": len(closed.tokens),
        "model_vocab_size": closed.model_vocab_size,
        "added_values": list(added),
        "vocabulary_sha256": closed.sha256(),
        "vocabulary_path": str(vocabulary_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
