"""Copy the small, high-value artifacts out of a results tree so they can be committed.

``results/`` is gitignored deliberately (.gitignore:5-8): a sweep tree carries
hundreds of MB of per-task reports and is meant to move between machines by
rsync. But the *summaries* -- the metric tables, the provenance, the per-task
blend diagnostics -- are a few MB and are exactly what a paper needs to be
reproducible from the repo alone.

This copies the summaries into a committed directory and leaves the bulk behind.
What is deliberately NOT copied: ``validation_report*.md``, ``task.txt`` and
``generated_code.md``, which together are >99% of the tree's size. Those stay in
``results/`` and should be archived separately if the raw verdicts are needed.

Per-task ``blend_debug.json`` files are small individually but numerous, so they
are concatenated into one JSONL per cell rather than copied one by one.

Usage::

    python experiments/collect_reports.py results/CacheBlendRun results/CacheBlendRun__view_*

Output defaults to ``docs/experiment_reports/``. Note it must NOT be
``docs/results/``: the ``results/`` pattern in .gitignore carries no leading
slash, so git matches it at any depth and would ignore everything written there
without saying so.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Copied verbatim, relative to a results root.
_REPORT_GLOBS = (
    "report/*.md",
    "report/*.json",
    "report/*.csv",
    "report/tables/*",
    "report/figures/*",
    "*.md",
    "*.yaml",
)

# Copied verbatim, relative to each cell directory (rep_*).
_CELL_FILES = ("env.json", "metrics.json", "Latency.json")

# Aggregated rather than copied file-by-file.
_PER_TASK_JSON = "blend_debug.json"

# Never copied: this is the bulk of the tree.
_EXCLUDED = ("validation_report", "task.txt", "generated_code.md")

_DEFAULT_MAX_MB = 50.0
_REP_DIR_RE = re.compile(r"rep_\d+")


def _copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def collect_root(root: Path, out_dir: Path) -> Dict[str, Any]:
    """Copy one results tree's summaries into ``out_dir/<root name>``."""
    target = out_dir / root.name
    copied = 0
    total = 0

    for pattern in _REPORT_GLOBS:
        for src in sorted(root.glob(pattern)):
            if src.is_file():
                total += _copy(src, target / src.relative_to(root))
                copied += 1

    # manifest.json is gitignored by pattern (.gitignore:16), so it is renamed
    # on the way out; it is the record of which cells actually ran.
    manifest = root / "manifest.json"
    if manifest.is_file():
        total += _copy(manifest, target / "manifest.copy.json")
        copied += 1

    cells = 0
    debug_records = 0
    for rep_dir in sorted(root.glob("*/*/*/rep_*")):
        if not rep_dir.is_dir() or not _REP_DIR_RE.fullmatch(rep_dir.name):
            continue
        cells += 1
        rel = rep_dir.relative_to(root)
        for name in _CELL_FILES:
            src = rep_dir / name
            if src.is_file():
                total += _copy(src, target / rel / name)
                copied += 1

        debug: List[Dict[str, Any]] = []
        for src in sorted(rep_dir.rglob(_PER_TASK_JSON)):
            try:
                record = json.loads(src.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            record["task_dir"] = src.parent.name
            debug.append(record)
        if debug:
            dst = target / rel / "blend_debug.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in debug
            )
            dst.write_text(payload, encoding="utf-8")
            total += len(payload.encode("utf-8"))
            copied += 1
            debug_records += len(debug)

    return {
        "root": str(root),
        "target": str(target),
        "files_copied": copied,
        "cells": cells,
        "blend_debug_records": debug_records,
        "bytes": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="results trees to collect from")
    # NOT docs/results: the `results/` pattern in .gitignore has no leading
    # slash, so it matches a directory of that name at any depth and would
    # silently swallow everything copied here.
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "docs" / "experiment_reports")
    parser.add_argument("--max-mb", type=float, default=_DEFAULT_MAX_MB,
                        help="refuse to write more than this; guards against "
                             "sweeping the raw reports in by accident")
    args = parser.parse_args()

    summaries = []
    for root in args.roots:
        if not root.is_dir():
            print(f"skip (not a directory): {root}")
            continue
        summary = collect_root(root, args.out)
        summaries.append(summary)
        print(
            f"{root.name}: {summary['files_copied']} files, "
            f"{summary['cells']} cells, "
            f"{summary['blend_debug_records']} debug records, "
            f"{summary['bytes'] / 1e6:.1f} MB -> {summary['target']}"
        )

    total_mb = sum(s["bytes"] for s in summaries) / 1e6
    print(f"\ntotal: {total_mb:.1f} MB in {args.out}")
    if total_mb > args.max_mb:
        print(
            f"\nWARNING: {total_mb:.1f} MB exceeds --max-mb {args.max_mb}. "
            f"Check that no raw per-task reports were swept in before committing."
        )
        return 1
    print(
        "\nRaw per-task validation reports were NOT copied -- they are the bulk "
        "of the tree and stay in results/. Archive them separately if the "
        "individual verdicts are needed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
