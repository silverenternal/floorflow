import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_chip_todo_active_reuse_paths_exist() -> None:
    todo = json.loads((ROOT / "chipstodo.json").read_text())
    missing = []
    for task in todo["tasks"]:
        for reference in task.get("reuse", []):
            if reference and not (ROOT / reference).exists():
                missing.append((task["id"], reference))
    assert missing == []
