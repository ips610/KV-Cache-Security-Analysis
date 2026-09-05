"""Build the 90-task secure-code corpus (coding_tasks.jsonl).

Samples 90 diverse app-building tasks deterministically from the MiniMaxAI/VIBE
dataset (200 curated full-stack prompts, MIT license). See docs/TASK_SOURCES.md
for the full source survey.

The sample is stratified over the 5x3 domain x difficulty grid (6 tasks per
stratum) using a fixed seed, so reruns produce byte-identical output. The
generated coding_tasks.jsonl is committed to the repo; runs never need
network access.

The ten authentication tasks from early development (auth_tasks.jsonl) are not
part of the corpus the paper reports; ``--include-auth`` prepends them, which
reproduces the 100-task file used before the reported runs.

Usage:
    python datasets/secure_code/build_tasks.py
    python datasets/secure_code/build_tasks.py --vibe-path <local vibe_queries.jsonl>
    python datasets/secure_code/build_tasks.py --include-auth
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTH_TASKS_PATH = HERE / "auth_tasks.jsonl"
OUTPUT_PATH = HERE / "coding_tasks.jsonl"

VIBE_URL = "https://huggingface.co/datasets/MiniMaxAI/VIBE/resolve/main/vibe_queries.jsonl"
VIBE_SAMPLE_SIZE = 90
SEED = 42


def read_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_vibe(local_path: str | None) -> list[dict]:
    if local_path:
        text = Path(local_path).read_text(encoding="utf-8")
    else:
        print(f"Downloading {VIBE_URL} ...")
        with urllib.request.urlopen(VIBE_URL) as resp:
            text = resp.read().decode("utf-8")
    rows = read_jsonl(text)
    for row in rows:
        for field in ("idx", "query", "domain", "difficulty"):
            if field not in row:
                raise ValueError(f"VIBE record missing '{field}': {row!r}")
    return rows


def sample_vibe(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample over domain x difficulty, deterministic under `seed`."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["idx"]):
        strata[(row["domain"], row["difficulty"])].append(row)

    per_stratum, remainder = divmod(n, len(strata))
    rng = random.Random(seed)
    picked: list[dict] = []
    for i, key in enumerate(sorted(strata)):
        take = per_stratum + (1 if i < remainder else 0)
        picked.extend(rng.sample(strata[key], take))
    return sorted(picked, key=lambda r: r["idx"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vibe-path",
        default=None,
        help="Local vibe_queries.jsonl to use instead of downloading",
    )
    parser.add_argument(
        "--include-auth",
        action="store_true",
        help="Prepend the 10 authentication tasks (auth_tasks.jsonl); not used by the paper",
    )
    args = parser.parse_args()

    vibe_rows = fetch_vibe(args.vibe_path)
    sampled = sample_vibe(vibe_rows, VIBE_SAMPLE_SIZE, SEED)

    records: list[dict] = []
    if args.include_auth:
        auth_tasks = read_jsonl(AUTH_TASKS_PATH.read_text(encoding="utf-8"))
        records += [{"task": t["task"], "source": "auth"} for t in auth_tasks]
    records += [
        {
            "task": row["query"],
            "source": "MiniMaxAI/VIBE",
            "vibe_idx": row["idx"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
        }
        for row in sampled
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} tasks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
