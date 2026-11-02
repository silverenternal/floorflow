#!/usr/bin/env python3
"""Generate the authoritative Table 2 caption with its v3 correction record."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/tables/cvpr_table2_caption_v3.md"


def main() -> None:
    text = """**Table 2. Main editing, strict-locality, topology, and graph results
(v3).** Five-seed CubiCasa test503 results use the hardened FEVP definitions.
Correction record (v3): use five-check validity rather than the legacy
single-flag value; report Student-t seed intervals; and treat the watertight
guard as a diagnostic without a net-benefit claim.
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {"status": "complete", "output": str(OUTPUT.relative_to(ROOT))}
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
