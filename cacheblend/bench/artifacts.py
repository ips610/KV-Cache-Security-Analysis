"""Write CacheBlend results in the exact layout KVCOMM's analysis expects.

The point of the whole exercise is that the CacheBlend arm is scored by the
*same* code as the KVCOMM arm, so no metric difference can be blamed on a
different analysis path. That means matching three contracts exactly:

* folder layout ``<model>/<mode>/<kv_point>/rep_<n>/<pass>/<task_slug>/`` -
  ``paper_metrics._CASE_RE`` requires exactly these six components.
* report filenames - ``validation_report.md`` paired with
  ``validation_report_dense_baseline.md`` (and optionally ``..._b.md``).
* ``Latency.json`` records whose ``mode`` is literally ``kv_reuse`` or
  ``dense_prefill`` (``secure_code_report.derive_flags`` *raises* on anything
  else), whose reference runs carry ``measure_only: true``, and whose
  ``agent_role`` is exactly ``Security Validator``.

Blend maps to ``kv_reuse`` and full prefill to ``dense_prefill`` because that is
what those labels mean to the analysis: the approximate arm and its paired exact
reference.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

VALIDATOR_ROLE = "Security Validator"
MODE_BLEND = "kv_reuse"
MODE_DENSE = "dense_prefill"


def cell_slug(recomp_ratio: float, suffix_config: str) -> str:
    """Name the kv-point directory for one (r, suffix config) cell.

    The suffix config is part of the slug so Config A and Config B never land in
    the same cell -- they are different treatments, and pooling them would repeat
    the collapse that experiments/make_kv_view.py exists to prevent.
    """
    return f"cb_r{recomp_ratio:.2f}_suf{suffix_config}"


def cell_dir(
    out_root: Path,
    model_slug: str,
    recomp_ratio: float,
    suffix_config: str,
    rep_index: int,
) -> Path:
    return (
        Path(out_root)
        / model_slug
        / "cacheblend"
        / cell_slug(recomp_ratio, suffix_config)
        / f"rep_{rep_index}"
    )


def write_task_artifacts(
    *,
    run_dir: Path,
    pass_name: str,
    task_slug: str,
    task_text: str,
    generated_code: str,
    blend_report: str,
    dense_report: str,
    dense_report_b: Optional[str] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Path:
    """Mirror experiments/secure_code_artifacts.write_task_artifacts.

    ``task_slug`` is passed in rather than recomputed: it comes from the frozen
    inputs, which derived it with KVCOMM's own ``task_slug()``, so the folder
    names are identical across the two arms by construction.
    """
    folder = Path(run_dir) / pass_name / task_slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task.txt").write_text(task_text, encoding="utf-8")
    (folder / "generated_code.md").write_text(generated_code or "", encoding="utf-8")
    (folder / "validation_report.md").write_text(blend_report or "", encoding="utf-8")
    (folder / "validation_report_dense_baseline.md").write_text(
        dense_report or "", encoding="utf-8"
    )
    if dense_report_b is not None:
        (folder / "validation_report_dense_baseline_b.md").write_text(
            dense_report_b, encoding="utf-8"
        )
    if debug is not None:
        (folder / "blend_debug.json").write_text(
            json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return folder


def latency_record(
    *,
    mode: str,
    message: str,
    input_tokens: int,
    output_tokens: int,
    generation_ttft: float,
    generation_total_latency: float,
    preprocess_latency: float,
    measure_only: bool,
    request_uid: str,
    agent_id: str = "1",
    agent_name: str = "SecurityValidator",
) -> Dict[str, Any]:
    """Build one Latency.json record in KVCOMM's schema.

    ``ttft`` deliberately includes ``preprocess_latency``, matching KVCOMM
    (gpt_chat.py:1359-1360) which counts anchor-blend preprocessing inside TTFT.
    For CacheBlend the analogous cost is chunk KV collection, and in this
    pipeline each code chunk is used exactly once, so that cost is not amortised
    and excluding it would flatter the blend arm. The kernel-only number is kept
    separately in ``generation_ttft``, which is what CacheBlend's own examples
    report.
    """
    if mode not in (MODE_BLEND, MODE_DENSE):
        raise ValueError(
            f"mode must be {MODE_BLEND!r} or {MODE_DENSE!r}, got {mode!r}; "
            "secure_code_report.derive_flags raises on anything else."
        )
    return {
        "timestamp": time.time(),
        "mode": mode,
        "ttft": float(preprocess_latency + generation_ttft),
        "generation_ttft": float(generation_ttft),
        "generation_total_latency": float(generation_total_latency),
        "preprocess_latency": float(preprocess_latency),
        "anchor_making_latency": 0.0,
        # CacheBlend always blends -- there is no hit/miss gate, so every blend
        # run is a "hit" and there is no fallback path.
        "anchor_hit": mode == MODE_BLEND,
        "offset_fallback": False,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "request_uid": request_uid,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_role": VALIDATOR_ROLE,
        "message": message,
        "placeholder_ids": ["user_question", "agent_0_current"],
        "measure_only": measure_only,
    }


def append_latency(run_dir: Path, records: List[Dict[str, Any]]) -> Path:
    """Append records to the cell's Latency.json, rewriting the whole file.

    KVCOMM appends per generation; rewriting keeps the file valid JSON at every
    point so a killed run still leaves something the analysis can read.
    """
    path = Path(run_dir) / "Latency.json"
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            existing = []
    existing.extend(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def write_env(run_dir: Path, env: Dict[str, Any]) -> Path:
    path = Path(run_dir) / "env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    path = Path(run_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(out_root: Path, cells: Dict[str, Dict[str, Any]]) -> Path:
    """Write a sweep-root manifest in experiments/harness/manifest.py's schema.

    ``mode`` is recorded honestly as ``cacheblend``. report.py's cross-model
    matrix filters on ``mode == "kvcomm"`` and will therefore skip these cells;
    that table stays empty rather than the manifest lying about what ran.
    """
    path = Path(out_root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "cells": cells}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
