"""Pure-Python helpers for saving secure-code run artifacts in an organized tree.

No KV-cache or model logic lives here — these helpers only derive folder names
and write the generator's full response + the validator's report to disk under a
per-run directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def task_slug(index: int, task_text: str) -> str:
    """Return an ordered, filesystem-safe folder name for a task, e.g. ``00_login``.

    The function name is pulled from a ``function <name>(`` phrase when present;
    otherwise a sanitized prefix of the task text is used.
    """
    match = re.search(r"function\s+(\w+)\s*\(", task_text)
    if match:
        name = match.group(1)
    else:
        name = re.sub(r"[^\w]+", "_", task_text.strip())[:30].strip("_") or "task"
    return f"{index:02d}_{name}"


def write_task_artifacts(
    *,
    run_dir: PathLike,
    pass_name: str,
    index: int,
    task_text: str,
    generator_text: str,
    report_text: str,
    dense_baseline_report_text: Optional[str] = None,
    dense_baseline_report_b_text: Optional[str] = None,
) -> Path:
    """Write the task prompt, full generator response, and validator report.

    ``dense_baseline_report_text`` is the on-hit measurement-only dense
    baseline's report (same task, dense prefill instead of kv_reuse); when
    present it is saved as ``validation_report_dense_baseline.md`` so the two
    reports can be compared side by side. It is None on anchor misses (the
    natural report itself is already dense).

    ``dense_baseline_report_b_text`` is an optional *second* dense baseline of
    the same validation, saved as ``validation_report_dense_baseline_b.md``.
    Diffing baseline A against baseline B measures the dense-vs-dense noise
    floor on the same cases fidelity is reported over.

    Creates ``<run_dir>/<pass_name>/<slug>/`` and returns that folder path.
    """
    folder = Path(run_dir) / pass_name / task_slug(index, task_text)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task.txt").write_text(task_text, encoding="utf-8")
    (folder / "generated_code.md").write_text(generator_text or "", encoding="utf-8")
    (folder / "validation_report.md").write_text(report_text or "", encoding="utf-8")
    if dense_baseline_report_text is not None:
        (folder / "validation_report_dense_baseline.md").write_text(
            dense_baseline_report_text, encoding="utf-8"
        )
    if dense_baseline_report_b_text is not None:
        (folder / "validation_report_dense_baseline_b.md").write_text(
            dense_baseline_report_b_text, encoding="utf-8"
        )
    return folder


def write_run_meta(run_dir: PathLike, meta: Dict[str, Any]) -> Path:
    """Dump ``meta.json`` describing the run in the run root. Returns its path."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    meta_path = run_path / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return meta_path


def write_primevul_task_artifacts(
    *,
    run_dir: PathLike,
    pass_name: str,
    index: int,
    task_text: str,
    source_code: str,
    report_text: str,
    audit_metadata: Dict[str, Any],
    dense_baseline_report_text: Optional[str] = None,
    dense_baseline_report_b_text: Optional[str] = None,
) -> Path:
    """Persist PrimeVul provenance without presenting code as Agent 1 output."""
    folder = Path(run_dir) / pass_name / task_slug(index, task_text)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task.txt").write_text(task_text, encoding="utf-8")
    (folder / "dataset_source_code.md").write_text(source_code, encoding="utf-8")
    (folder / "handoff_audit.json").write_text(
        json.dumps(audit_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "validation_report.md").write_text(report_text or "", encoding="utf-8")
    if dense_baseline_report_text is not None:
        (folder / "validation_report_dense_baseline.md").write_text(
            dense_baseline_report_text, encoding="utf-8"
        )
    if dense_baseline_report_b_text is not None:
        (folder / "validation_report_dense_baseline_b.md").write_text(
            dense_baseline_report_b_text, encoding="utf-8"
        )
    return folder
