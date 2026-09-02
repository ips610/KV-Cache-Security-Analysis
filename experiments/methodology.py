"""Emit a self-contained METHODOLOGY.md describing exactly how a sweep was produced.

Answers, for any results tree, the questions a reader (or reviewer, or you in
three months) will ask: what ran, in what order, what each function was
responsible for, and what every controllable setting was fixed to.

Two halves:

* **Static** — the pipeline's call order and the role of each function. This is
  a description of the code path, kept next to the code so it moves with it.
* **Harvested** — the settings actually used, read out of each cell's
  ``env.json``/``meta.json`` rather than restated from memory. Where cells
  disagree on a setting the disagreement is surfaced explicitly, because a
  sweep whose cells silently ran under different configs is not comparable.

Usage:
    python experiments/methodology.py results/FinalRun
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

# Ordered description of the execution path for one task. Kept in sync with
# run_secure_code.py / graph.py / security_validator.py by hand — it documents
# intent and responsibility, which no amount of introspection recovers.
CALL_ORDER: List[Dict[str, str]] = [
    {
        "step": "1",
        "call": "run_secure_code.main()",
        "file": "experiments/run_secure_code.py",
        "role": "Entry point. Parses CLI args, seeds RNGs, pins determinism flags "
                "(cuDNN deterministic, CUBLAS_WORKSPACE_CONFIG), resolves the run "
                "directory, and writes env.json capturing every controllable setting.",
    },
    {
        "step": "2",
        "call": "KVCommConfig.from_env().apply_overrides()",
        "file": "KVCOMM/llm/config.py",
        "role": "Builds the KV-reuse configuration: entropy threshold, max anchor "
                "count, and anchor-pool window size.",
    },
    {
        "step": "3",
        "call": "load_tasks()",
        "file": "datasets/secure_code_dataset.py",
        "role": "Loads the task prompts from the JSONL task file. Task order is "
                "fixed and identical across models and repetitions, so the anchor "
                "pool evolves identically.",
    },
    {
        "step": "4",
        "call": "build_graph()",
        "file": "experiments/run_secure_code.py",
        "role": "Constructs the fixed 2-node chain: node 0 CodeGenerator -> node 1 "
                "SecurityValidator. Spatial mask wires generator into validator; "
                "no temporal edges (single round).",
    },
    {
        "step": "5",
        "call": "_run_pass() [loop over tasks]",
        "file": "experiments/run_secure_code.py",
        "role": "Per task: deep-copies the graph (fresh per-request state, shared "
                "anchor pool), runs it, extracts both agents' outputs, and writes "
                "task.txt / generated_code.md / validation_report.md.",
    },
    {
        "step": "6",
        "call": "Graph.arun(mode='allow_kv_reuse')",
        "file": "KVCOMM/graph/graph.py",
        "role": "Executes the chain in dependency order, awaiting each node.",
    },
    {
        "step": "7",
        "call": "CodeGenerator._async_execute()",
        "file": "KVCOMM/agents/code_generator.py",
        "role": "Agent 1 writes the code for the task. ALWAYS runs dense — this is "
                "the design choice that removes the largest confound: the code "
                "under review is byte-identical across the dense and reuse arms.",
    },
    {
        "step": "8",
        "call": "SecurityValidator._build_validator_prompts()",
        "file": "KVCOMM/agents/security_validator.py",
        "role": "Builds the validator prompt as: [code placeholder] followed by the "
                "NIST rules as EXACT trailing text. Only the code span is eligible "
                "for KV approximation; the rules are never approximated.",
    },
    {
        "step": "9",
        "call": "KVCOMMEngine.predict_as_anchor()",
        "file": "KVCOMM/llm/kvcomm_engine.py",
        "role": "The entropy gate. Computes similarity of the candidate KV against "
                "stored anchors; if entropy exceeds threshold*log2(n) the segment is "
                "declared unsharable and falls back to dense prefill (a 'miss'). "
                "Deterministic: sort + cumsum nucleus selection, no sampling.",
    },
    {
        "step": "10a",
        "call": "KVCOMMEngine.offset_kv_cache_pair()  [on HIT]",
        "file": "KVCOMM/llm/kvcomm_engine.py",
        "role": "Approximates the code span's KV by blending stored anchor deltas, "
                "weighted by a softmax over anchor similarity, then rotates keys to "
                "the correct positions. THIS is the operation under audit.",
    },
    {
        "step": "10b",
        "call": "dense prefill  [on MISS]",
        "file": "KVCOMM/llm/gpt_chat.py",
        "role": "Full exact prefill. A natural miss is already its own dense "
                "reference, so it needs no separate baseline.",
    },
    {
        "step": "11",
        "call": "generate_for_agent(preferred_mode='dense_prefill', measure_only=True)",
        "file": "KVCOMM/agents/security_validator.py",
        "role": "On every anchor HIT, re-runs the SAME validation with dense prefill "
                "as a measurement-only baseline -> validation_report_dense_baseline.md. "
                "measure_only=True suppresses the anchor-pool write so later tasks' "
                "natural hit rate is unbiased. This is the paired dense reference.",
    },
    {
        "step": "12",
        "call": "second dense baseline  [only if --noise-floor-baseline true]",
        "file": "KVCOMM/agents/security_validator.py",
        "role": "A SECOND independent dense run of the same validation -> "
                "validation_report_dense_baseline_b.md. Diffing baseline A against "
                "baseline B measures the dense-vs-dense noise floor on exactly the "
                "cases fidelity is reported over.",
    },
    {
        "step": "13",
        "call": "MetricsCollector / Latency.json",
        "file": "experiments/metrics_collector.py",
        "role": "Records per-generation TTFT, total generation latency, prefill and "
                "preprocess time, mode (kv_reuse / dense_prefill) and the "
                "measure_only flag for every agent call.",
    },
    {
        "step": "14",
        "call": "report.py / compare_validation_reports.py / paper_metrics.py / timing_metrics.py",
        "file": "experiments/",
        "role": "Offline analysis (no GPU): timing aggregates, raw verdict diffing, "
                "unique-case fidelity metrics + silencing rate + noise floor, and the "
                "TTFT-vs-end-to-end deflation table and figure.",
    },
]

METRIC_DEFINITIONS: List[Dict[str, str]] = [
    {"metric": "Anchor-hit rate",
     "definition": "Fraction of validator calls where the entropy gate selected reuse "
                   "over dense prefill. Measurement-only baselines are excluded."},
    {"metric": "Report-change rate",
     "definition": "Fraction of unique paired cases where ANY per-rule verdict differs "
                   "between the dense reference and KV reuse."},
    {"metric": "Rule agreement (overall)",
     "definition": "Matching verdicts / all compared verdicts. Inflated by rules that "
                   "are NOT APPLICABLE in both arms."},
    {"metric": "Rule agreement (applicable-only) [PRIMARY]",
     "definition": "Matching verdicts / comparisons where the dense reference returned "
                   "PASS or FAIL, i.e. where the rule actually applied."},
    {"metric": "Vulnerability silencing rate [HEADLINE]",
     "definition": "(FAIL->PASS + FAIL->NOT APPLICABLE) / total dense FAIL verdicts. "
                   "FAIL->NA counts as silencing because the prompt permits NOT "
                   "APPLICABLE only when the rule's subject matter is absent, so the "
                   "transition asserts the validator can no longer see the code."},
    {"metric": "Noise floor",
     "definition": "The same fidelity metrics computed between two independent dense "
                   "runs. Any nonzero value bounds how much of the reuse divergence "
                   "is attributable to KV reuse rather than pipeline variance."},
    {"metric": "TTFT speedup",
     "definition": "Paired per-task: measured dense-baseline TTFT / kv_reuse TTFT."},
    {"metric": "End-to-end speedup",
     "definition": "The same pairing over complete generation time. Always reported "
                   "alongside TTFT, since decode dominates wall-clock at long outputs."},
]


def harvest_settings(root: Path) -> Dict[str, object]:
    """Collect every cell's env.json and fold identical values together."""
    env_paths = sorted(root.rglob("env.json"))
    cells: List[Dict[str, object]] = []
    for path in env_paths:
        try:
            cells.append({"cell": path.parent.relative_to(root).as_posix(),
                          "env": json.loads(path.read_text(encoding="utf-8"))})
        except Exception as exc:  # noqa: BLE001 — a damaged cell must not abort the report
            cells.append({"cell": path.parent.relative_to(root).as_posix(),
                          "error": repr(exc)})

    def collect(getter) -> Dict[str, List[str]]:
        values: Dict[str, List[str]] = defaultdict(list)
        for cell in cells:
            if "env" not in cell:
                continue
            try:
                value = getter(cell["env"])
            except Exception:  # noqa: BLE001
                value = None
            values[json.dumps(value, sort_keys=True, default=str)].append(cell["cell"])
        return dict(values)

    fields = {
        "model": lambda e: e.get("cli_args", {}).get("llm_name"),
        "seed": lambda e: e.get("seed"),
        "max_tokens": lambda e: e.get("cli_args", {}).get("max_tokens"),
        "tasks_path": lambda e: e.get("cli_args", {}).get("tasks_path"),
        "kv_threshold": lambda e: e.get("cli_args", {}).get("kv_threshold"),
        "kv_max_anchor_num": lambda e: e.get("cli_args", {}).get("kv_max_anchor_num"),
        "kv_window_size": lambda e: e.get("cli_args", {}).get("kv_window_size"),
        "force_exact_nist_prefill": lambda e: e.get("cli_args", {}).get("force_exact_nist_prefill"),
        "approximate_nist_rules": lambda e: e.get("cli_args", {}).get("approximate_nist_rules"),
        "decode": lambda e: e.get("cli_args", {}).get("decode"),
        "determinism": lambda e: e.get("cli_args", {}).get("determinism"),
        "noise_floor_baseline": lambda e: e.get("cli_args", {}).get("noise_floor_baseline"),
        "gpu": lambda e: e.get("gpu_name"),
        "torch_version": lambda e: e.get("torch_version"),
        "transformers_version": lambda e: e.get("transformers_version"),
        "python_version": lambda e: e.get("python_version"),
        "git_commit": lambda e: e.get("git_commit"),
    }
    return {"n_cells": len(cells), "fields": {name: collect(fn) for name, fn in fields.items()}}


def _render_setting(name: str, values: Dict[str, List[str]]) -> List[str]:
    if not values:
        return [f"| `{name}` | *(not recorded)* | — |"]
    if len(values) == 1:
        raw = next(iter(values))
        return [f"| `{name}` | `{raw}` | all cells |"]
    lines = []
    for raw, cells in sorted(values.items(), key=lambda x: -len(x[1])):
        lines.append(f"| `{name}` | `{raw}` | {len(cells)} cells |")
    return lines


def render(root: Path, harvested: Dict[str, object]) -> str:
    lines = [f"# Methodology & provenance — `{root.name}`", ""]
    lines.append(f"Generated from {harvested['n_cells']} cell(s) under `{root}`. "
                 "Every setting below is read from the runs themselves, not restated "
                 "from configuration.")
    lines.append("")

    lines.append("## 1. Pipeline")
    lines.append("")
    lines.append("```")
    lines.append("  task prompt")
    lines.append("      |")
    lines.append("      v")
    lines.append("  [Agent 1: CodeGenerator]  -- ALWAYS DENSE --> generated code")
    lines.append("      |")
    lines.append("      v")
    lines.append("  [Agent 2: SecurityValidator]")
    lines.append("      |-- entropy gate --> HIT  : KV reuse (code span approximated)")
    lines.append("      |                          + measurement-only dense baseline A")
    lines.append("      |                          [+ dense baseline B if noise-floor on]")
    lines.append("      '-- entropy gate --> MISS : dense prefill (its own reference)")
    lines.append("      |")
    lines.append("      v")
    lines.append("  per-rule PASS / FAIL / NOT APPLICABLE verdicts")
    lines.append("```")
    lines.append("")
    lines.append("KV reuse is confined to the **validator**. The generator always runs "
                 "dense, so the code under review is byte-identical across arms and the "
                 "dense report is an exact behavioural reference rather than an "
                 "approximate comparator.")
    lines.append("")

    lines.append("## 2. Call order and the role of each function")
    lines.append("")
    lines.append("| # | Call | Source | Role |")
    lines.append("|---|---|---|---|")
    for step in CALL_ORDER:
        lines.append(f"| {step['step']} | `{step['call']}` | `{step['file']}` | {step['role']} |")
    lines.append("")

    lines.append("## 3. Controllable settings actually used")
    lines.append("")
    lines.append("A row listing more than one value means cells of this sweep ran under "
                 "different settings — check before comparing them.")
    lines.append("")
    lines.append("| Setting | Value | Applies to |")
    lines.append("|---|---|---|")
    for name, values in harvested["fields"].items():
        lines += _render_setting(name, values)
    lines.append("")

    lines.append("### Decoding")
    lines.append("")
    lines.append("Generation is **greedy**: `do_sample=False` is set at every call site in "
                 "`KVCOMM/llm/gpt_chat.py`. The `temperature` setting is therefore "
                 "**inert** — Hugging Face ignores it when sampling is disabled — and is "
                 "recorded for provenance only. A change to `LLM.DEFAULT_TEMPERATURE` "
                 "does not alter generated text.")
    lines.append("")
    lines.append("Note the codebase has a *second*, unrelated temperature: the softmax "
                 "temperature over anchor similarity in "
                 "`KVCOMMEngine.offset_kv_cache_pair` (fixed at 1). That one does affect "
                 "results — it controls how sharply anchor deltas are weighted during "
                 "blending — but it is not the decoding temperature.")
    lines.append("")

    lines.append("### Determinism")
    lines.append("")
    lines.append("Because decoding is greedy, the only remaining source of run-to-run "
                 "variation is non-deterministic GPU kernel reduction order. Runs pin "
                 "`torch.use_deterministic_algorithms(True, warn_only=True)`, "
                 "`cudnn.deterministic=True`, `cudnn.benchmark=False`, and "
                 "`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The measured reproducibility rate is "
                 "reported in `paper_metrics.md`; treat that as the check, not this "
                 "paragraph as the guarantee.")
    lines.append("")

    lines.append("## 4. Metric definitions")
    lines.append("")
    lines.append("| Metric | Definition |")
    lines.append("|---|---|")
    for item in METRIC_DEFINITIONS:
        lines.append(f"| {item['metric']} | {item['definition']} |")
    lines.append("")

    lines.append("## 5. Output files")
    lines.append("")
    lines.append("| File | Contents |")
    lines.append("|---|---|")
    lines.append("| `report/paper_metrics.md` | Unique-case fidelity: change rate, applicable-only agreement, transition matrices, silencing rate, noise floor |")
    lines.append("| `report/timing_metrics.md` | TTFT vs end-to-end speedup per model (the deflation result) |")
    lines.append("| `report/cross_model_matrix.md` | Per-model hit rate, TTFT, throughput, energy |")
    lines.append("| `report/aggregate_stats.md` | Every timing metric as mean ± 95% CI across seeds |")
    lines.append("| `report_discrepancy_summary.md` | Raw per-case verdict diffs (not deduplicated) |")
    lines.append("| `METHODOLOGY.md` | This file |")
    lines.append("| `<model>/.../rep_<n>/warm/<task>/` | Per-task `task.txt`, `generated_code.md`, `validation_report.md`, `validation_report_dense_baseline.md` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_root", type=Path, help="Sweep root, e.g. results/FinalRun")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: <results_root>/METHODOLOGY.md)")
    args = parser.parse_args(argv)

    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    harvested = harvest_settings(root)
    out = args.out or (root / "METHODOLOGY.md")
    out.write_text(render(root, harvested), encoding="utf-8")
    (root / "provenance.json").write_text(
        json.dumps({"call_order": CALL_ORDER, "metrics": METRIC_DEFINITIONS,
                    "harvested": harvested}, indent=2),
        encoding="utf-8",
    )
    print(f"[methodology] harvested {harvested['n_cells']} cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
