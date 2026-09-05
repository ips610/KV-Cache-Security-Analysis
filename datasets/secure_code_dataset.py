"""Loader for the secure-code task set (90 diverse app-building tasks).

The dataset (``secure_code/coding_tasks.jsonl``) holds 90 tasks sampled from
MiniMaxAI/VIBE, stratified over domain x difficulty; it is regenerated
byte-identically with ``datasets/secure_code/build_tasks.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_TASKS_PATH = str(Path(__file__).resolve().parent / "secure_code" / "coding_tasks.jsonl")


def load_tasks(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Read the coding tasks JSONL into a list of {'task': ...} dicts."""
    tasks_path = Path(path or DEFAULT_TASKS_PATH)
    records: List[Dict[str, str]] = []
    with open(tasks_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "task" not in obj or not str(obj["task"]).strip():
                raise ValueError(f"Each record must contain a non-empty 'task': {obj!r}")
            records.append({"task": str(obj["task"])})
    return records


# Backwards-compatible alias (previous name for the loader).
load_auth_tasks = load_tasks
