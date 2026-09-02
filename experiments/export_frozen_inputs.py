"""Freeze the Security Validator's inputs from a completed KVCOMM sweep.

The CacheBlend arm must validate *exactly* the code KVCOMM validated, otherwise
the two arms differ by generator output as well as by KV-reuse mechanism. This
script walks a finished sweep tree, picks one canonical kv point, and writes the
task text + generated code per (model, task) to a frozen JSONL that the
CacheBlend adapter consumes instead of running a generator of its own.

It also records two things the comparison depends on:

* ``code_identical_across_kv`` - the generator is configured dense+greedy
  (``generator_dense: true``), so its output *should* be identical across kv
  points. For Llama-3.2-3B it is not: 59/90 tasks differ between gamma=0.1 and
  gamma in {0.3, 0.5}. Tasks that drift are flagged as off-corpus rather than
  silently averaged over.
* ``kvcomm_natural_mode_by_kv`` - which tasks KVCOMM actually reused KV on. Its
  fidelity numbers are conditioned on anchor hits; CacheBlend covers every task.
  Exporting the hit set is what lets the analysis report a coverage-matched
  comparison alongside the full-coverage one.

Stdlib only - no torch, no transformers, no GPU.

Usage::

    python experiments/export_frozen_inputs.py \
        --sweep-root results/MarginThresholdRun \
        --canonical-kv kv_t0.3_a20_w5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# Folder layout produced by experiments/sweep.py + secure_code_artifacts.py:
#   <root>/<model_slug>/<mode>/<kv_point>/rep_<n>/<pass>/<task_slug>/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TASKS = _REPO_ROOT / "datasets" / "secure_code" / "coding_tasks.jsonl"
_DEFAULT_OUT = _REPO_ROOT / "datasets" / "secure_code" / "frozen_validator_inputs"

_VALIDATOR_ROLE = "Security Validator"
_REP_RE = re.compile(r"^rep_(\d+)$")
_SLUG_RE = re.compile(r"^(\d+)_")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    """Read a run artifact. Artifacts are always written as UTF-8."""
    return path.read_text(encoding="utf-8")


def _load_tasks(tasks_path: Path) -> List[str]:
    tasks: List[str] = []
    with open(tasks_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line)["task"])
    return tasks


def _slug_index(slug: str) -> Optional[int]:
    match = _SLUG_RE.match(slug)
    return int(match.group(1)) if match else None


def _discover_cells(sweep_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Return {model_slug: {kv_point: [rep_dir, ...]}} for every finished cell."""
    cells: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for model_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        if model_dir.name in {"report", "logs"}:
            continue
        for mode_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for kv_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
                reps = sorted(
                    p for p in kv_dir.iterdir() if p.is_dir() and _REP_RE.match(p.name)
                )
                if reps:
                    cells[model_dir.name][kv_dir.name].extend(reps)
    return {model: dict(kvs) for model, kvs in cells.items()}


def _task_dirs(rep_dir: Path) -> Dict[int, Path]:
    """Map task_index -> task folder for one repetition, across all pass names."""
    found: Dict[int, Path] = {}
    for pass_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir()):
        if pass_dir.name == "logs":
            continue
        for task_dir in sorted(p for p in pass_dir.iterdir() if p.is_dir()):
            index = _slug_index(task_dir.name)
            if index is None or not (task_dir / "task.txt").exists():
                continue
            found.setdefault(index, task_dir)
    return found


def _validator_records(rep_dir: Path) -> List[Dict[str, Any]]:
    latency_path = rep_dir / "Latency.json"
    if not latency_path.exists():
        return []
    try:
        records = json.loads(_read_text(latency_path))
    except json.JSONDecodeError:
        return []
    if not isinstance(records, list):
        return []
    return [r for r in records if r.get("agent_role") == _VALIDATOR_ROLE]


def _natural_validator_index(rep_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Map task text -> the NATURAL validator record (measure_only records excluded).

    The natural pass is the one whose mode decides the arm: ``kv_reuse`` on an
    anchor hit, ``dense_prefill`` on a miss. Reference runs carry
    ``measure_only: true`` and must not be counted as coverage.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for record in _validator_records(rep_dir):
        if record.get("measure_only"):
            continue
        message = record.get("message")
        if isinstance(message, str):
            index.setdefault(message, record)
    return index


def _reference_input_tokens(rep_dir: Path) -> Dict[str, int]:
    """Map task text -> input_tokens of the DENSE validator pass for that task.

    This is ground truth for the prompt-spec exporter: a reconstructed
    ``full_ids`` must have exactly this length. Prefers the measure_only dense
    reference (anchor hits); falls back to the natural dense pass (misses).
    """
    tokens: Dict[str, int] = {}
    fallback: Dict[str, int] = {}
    for record in _validator_records(rep_dir):
        message = record.get("message")
        count = record.get("input_tokens")
        if not isinstance(message, str) or not isinstance(count, int):
            continue
        if record.get("mode") != "dense_prefill":
            continue
        if record.get("measure_only"):
            tokens.setdefault(message, count)
        else:
            fallback.setdefault(message, count)
    for message, count in fallback.items():
        tokens.setdefault(message, count)
    return tokens


def _generator_index(rep_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Map task text -> the generator's natural record for that task."""
    latency_path = rep_dir / "Latency.json"
    if not latency_path.exists():
        return {}
    try:
        records = json.loads(_read_text(latency_path))
    except json.JSONDecodeError:
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for record in records if isinstance(records, list) else []:
        if record.get("agent_role") == _VALIDATOR_ROLE or record.get("measure_only"):
            continue
        message = record.get("message")
        if isinstance(message, str):
            index.setdefault(message, record)
    return index


def _max_tokens(rep_dir: Path) -> Optional[int]:
    """Read the generation cap this cell ran under, for truncation detection."""
    env_path = rep_dir / "env.json"
    if not env_path.exists():
        return None
    try:
        env = json.loads(_read_text(env_path))
    except json.JSONDecodeError:
        return None
    value = (env.get("cli_args") or {}).get("max_tokens")
    return value if isinstance(value, int) else None


def _generator_agent_id(rep_dir: Path) -> Optional[str]:
    """Recover the generator's agent id from the validator's placeholder ids.

    The validator prompt embeds ``{agent_<id>_current}``; the id is assigned by
    the graph at construction time, so it must be read from the run rather than
    assumed.
    """
    for record in _validator_records(rep_dir):
        for placeholder in record.get("placeholder_ids") or []:
            match = re.fullmatch(r"agent_(\w+)_current", str(placeholder))
            if match:
                return match.group(1)
    return None


def _collect_model(
    model_slug: str,
    kv_points: Dict[str, List[Path]],
    canonical_kv: str,
    tasks: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the frozen records for one model plus its stability audit."""
    if canonical_kv not in kv_points:
        raise SystemExit(
            f"{model_slug}: canonical kv point '{canonical_kv}' not found. "
            f"Available: {sorted(kv_points)}"
        )

    # code_sha256[task_index][kv_point] -> hash, for the cross-kv drift audit.
    code_hashes: Dict[int, Dict[str, str]] = defaultdict(dict)
    natural_modes: Dict[int, Dict[str, str]] = defaultdict(dict)
    has_dense_pair: Dict[int, Dict[str, bool]] = defaultdict(dict)

    for kv_point, rep_dirs in sorted(kv_points.items()):
        rep_dir = rep_dirs[0]
        dirs = _task_dirs(rep_dir)
        natural = _natural_validator_index(rep_dir)
        for index, task_dir in sorted(dirs.items()):
            code_path = task_dir / "generated_code.md"
            if code_path.exists():
                code_hashes[index][kv_point] = _sha256(_read_text(code_path))
            task_text = _read_text(task_dir / "task.txt")
            record = natural.get(task_text)
            if record is not None:
                natural_modes[index][kv_point] = str(record.get("mode"))
            has_dense_pair[index][kv_point] = (
                task_dir / "validation_report_dense_baseline.md"
            ).exists()

    canonical_rep = kv_points[canonical_kv][0]
    canonical_dirs = _task_dirs(canonical_rep)
    reference_tokens = _reference_input_tokens(canonical_rep)
    generator_agent_id = _generator_agent_id(canonical_rep)
    gen_index = _generator_index(canonical_rep)
    max_tokens = _max_tokens(canonical_rep)

    records: List[Dict[str, Any]] = []
    drifted: List[int] = []
    truncated: List[int] = []
    for index, task_dir in sorted(canonical_dirs.items()):
        task_text = _read_text(task_dir / "task.txt")
        code_text = _read_text(task_dir / "generated_code.md")

        distinct = set(code_hashes[index].values())
        identical = len(distinct) <= 1
        if not identical:
            drifted.append(index)

        if index < len(tasks) and tasks[index] != task_text:
            raise SystemExit(
                f"{model_slug} task {index}: task.txt does not match "
                f"coding_tasks.jsonl line {index}. Refusing to freeze a corpus "
                f"that does not match the dataset."
            )

        gen_record = gen_index.get(task_text) or {}
        gen_output_tokens = gen_record.get("output_tokens")
        gen_truncated = (
            isinstance(gen_output_tokens, int)
            and isinstance(max_tokens, int)
            and gen_output_tokens >= max_tokens
        )
        if gen_truncated:
            truncated.append(index)

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "model_slug": model_slug,
                "task_index": index,
                "task_slug": task_dir.name,
                "task": task_text,
                "generated_code": code_text,
                "task_sha256": _sha256(task_text),
                "code_sha256": _sha256(code_text),
                "generator_agent_id": generator_agent_id,
                "generator_output_tokens": gen_output_tokens,
                "generator_truncated": gen_truncated,
                "source_cell": str(canonical_rep.relative_to(canonical_rep.parents[3])),
                "kvcomm_dense_input_tokens": reference_tokens.get(task_text),
                "code_identical_across_kv": identical,
                "code_sha256_by_kv": dict(sorted(code_hashes[index].items())),
                "kvcomm_natural_mode_by_kv": dict(sorted(natural_modes[index].items())),
                "kvcomm_has_dense_pair_by_kv": dict(sorted(has_dense_pair[index].items())),
            }
        )

    hit_counts = {
        kv: sum(1 for i in natural_modes if natural_modes[i].get(kv) == "kv_reuse")
        for kv in sorted(kv_points)
    }
    clean = [
        r["task_index"]
        for r in records
        if r["code_identical_across_kv"] and not r["generator_truncated"]
    ]
    audit = {
        "model_slug": model_slug,
        "canonical_kv": canonical_kv,
        "kv_points": sorted(kv_points),
        "n_tasks": len(records),
        "max_tokens": max_tokens,
        "n_tasks_code_drifted_across_kv": len(drifted),
        "drifted_task_indices": drifted,
        # The generator was capped at max_tokens; a capped task means the code
        # under review is cut off mid-output, so the validator reviewed an
        # incomplete program. Not an error, but it must not be averaged over
        # silently.
        "n_tasks_generator_truncated": len(truncated),
        "truncated_task_indices": truncated,
        "n_tasks_clean": len(clean),
        "clean_task_indices": clean,
        "kvcomm_hit_counts_by_kv": hit_counts,
        "n_tasks_missing_reference_tokens": sum(
            1 for r in records if r["kvcomm_dense_input_tokens"] is None
        ),
    }
    return records, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=_REPO_ROOT / "results" / "MarginThresholdRun",
        help="Finished sweep tree to freeze from.",
    )
    parser.add_argument(
        "--canonical-kv",
        default="kv_t0.3_a20_w5",
        help="kv point to take the frozen code from.",
    )
    parser.add_argument("--tasks-path", type=Path, default=_DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    if not args.sweep_root.exists():
        raise SystemExit(f"sweep root not found: {args.sweep_root}")

    tasks = _load_tasks(args.tasks_path)
    cells = _discover_cells(args.sweep_root)
    if not cells:
        raise SystemExit(f"no cells discovered under {args.sweep_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audits: List[Dict[str, Any]] = []
    file_hashes: Dict[str, str] = {}

    for model_slug, kv_points in sorted(cells.items()):
        records, audit = _collect_model(
            model_slug, kv_points, args.canonical_kv, tasks
        )
        out_path = args.out_dir / f"{model_slug}.jsonl"
        payload = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
        out_path.write_text(payload, encoding="utf-8")
        file_hashes[out_path.name] = _sha256(payload)
        audits.append(audit)

        print(
            f"{model_slug}: {audit['n_tasks']} tasks -> {out_path.name}\n"
            f"    drifted_across_kv={audit['n_tasks_code_drifted_across_kv']}  "
            f"generator_truncated={audit['n_tasks_generator_truncated']}  "
            f"clean={audit['n_tasks_clean']}/{audit['n_tasks']}\n"
            f"    kvcomm_hits={audit['kvcomm_hit_counts_by_kv']}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sweep_root": str(args.sweep_root),
        "canonical_kv": args.canonical_kv,
        "tasks_path": str(args.tasks_path),
        "n_tasks_in_dataset": len(tasks),
        "file_sha256": file_hashes,
        "code_stability": audits,
    }
    manifest_path = args.out_dir / "frozen_inputs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nmanifest -> {manifest_path}")

    drifted_models = [a for a in audits if a["n_tasks_code_drifted_across_kv"]]
    if drifted_models:
        print(
            "\nNOTE: generator output is not kv-invariant for "
            + ", ".join(
                f"{a['model_slug']} ({a['n_tasks_code_drifted_across_kv']} tasks)"
                for a in drifted_models
            )
            + f".\nFrozen from {args.canonical_kv}; other kv points are off-corpus "
            "for those tasks and must be excluded or reported separately."
        )

    truncated_models = [a for a in audits if a["n_tasks_generator_truncated"]]
    if truncated_models:
        print(
            "\nNOTE: the generator hit its max_tokens cap on "
            + ", ".join(
                f"{a['model_slug']} ({a['n_tasks_generator_truncated']}/{a['n_tasks']})"
                for a in truncated_models
            )
            + ".\nThose tasks' code is cut off mid-output, so the validator "
            "reviewed an incomplete program. This affects the existing KVCOMM "
            "results as well; report fidelity on the clean subset alongside the "
            "full corpus rather than pooling silently."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
