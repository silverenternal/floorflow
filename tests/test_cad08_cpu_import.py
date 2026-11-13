"""The CPU CAD08 API must not import the optional Torch runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cad08_transaction_import_does_not_load_torch():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import floorflow.inference.cad08_transaction; "
        "import floorflow.geometry.cell_polygons"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True)
