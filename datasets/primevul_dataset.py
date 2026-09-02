"""Loader for the PrimeVul ground-truth task sets.

Unlike ``secure_code_dataset.load_tasks``, this keeps every field: the code
under review is supplied by the dataset rather than generated, and the runner
also needs ``sample_id`` / ``cwe`` / ``label`` to write provenance. The graph
passes the whole input dict to every node, so the extra keys reach the agents
untouched.

``task`` remains the join key everywhere downstream (``Latency.json.message``,
the on-hit dense-baseline map, the artifact folder slug), so it is checked for
uniqueness here as well as at build time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "primevul"
DEFAULT_TASKS_PATH = str(DATA_DIR / "tasks_vulnerable.jsonl")
PAIRS_PATH = str(DATA_DIR / "primevul_pairs.jsonl")

REQUIRED = ("task", "code", "sample_id", "pair_id", "cwe", "arm", "label")


def load_tasks(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read a PrimeVul arm task file into a list of full task dicts."""
    tasks_path = Path(path or DEFAULT_TASKS_PATH)
    records: List[Dict[str, Any]] = []
    with open(tasks_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = [field for field in REQUIRED if field not in obj]
            if missing:
                raise ValueError(f"{tasks_path}: record is missing {missing}: {obj!r}")
            if not str(obj["task"]).strip() or not str(obj["code"]).strip():
                raise ValueError(f"{tasks_path}: empty task or code in {obj['sample_id']!r}")
            records.append(obj)
    if len({r["task"] for r in records}) != len(records):
        raise ValueError(f"{tasks_path}: task strings are not unique; they are the KV join key")
    return records


def load_pairs(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return the pair table keyed by ``sample_id`` (ground truth + provenance)."""
    pairs_path = Path(path or PAIRS_PATH)
    table: Dict[str, Dict[str, Any]] = {}
    with open(pairs_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            table[row["sample_id"]] = row
    return table
