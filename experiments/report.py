"""Offline sweep reporting: tables, CI stats, figures, LaTeX, exports.

Reads a sweep directory produced by experiments/sweep.py. No GPU, no
models — safe to run on any machine after syncing results.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.harness.stats import mean_ci
from experiments.secure_code_report import (
    load_latency,
    natural_hit_summary,
    pair_reuse_with_baseline,
)

VALIDATOR_ROLE = "Security Validator"

# Metrics aggregated over repetitions (mean/std/CI each).
AGG_METRICS = [
    "hit_rate", "reuse_mean_ttft", "onhit_baseline_mean_ttft",
    "onhit_mean_speedup", "natural_dense_mean_ttft",
    "reuse_mean_total_latency", "onhit_baseline_mean_total_latency",
    "onhit_mean_total_speedup",
    "validator_tokens_per_sec", "gpu_peak_mem_gb", "pass_duration_s",
    "tasks_per_minute", "energy_joules", "avg_power_w",
]


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def build_cell_row(entry: Dict[str, Any], sweep_dir: Path) -> Dict[str, Any]:
    """Flatten one done cell into a metrics row (honest-baseline semantics)."""
    cell_dir = sweep_dir / entry["cell_dir"]
    records = load_latency(str(cell_dir / "Latency.json"))
    hits = natural_hit_summary(records, agent_role=VALIDATOR_ROLE)
    pairs = pair_reuse_with_baseline(records, agent_role=VALIDATOR_ROLE)

    reuse_vals = [p["reuse_ttft"] for p in pairs
                  if p["natural_mode"] == "kv_reuse" and p["reuse_ttft"]]
    onhit_base_vals = [p["baseline_ttft"] for p in pairs
                       if p["natural_mode"] == "kv_reuse" and p["baseline_ttft"]]
    dense_vals = [p["baseline_ttft"] for p in pairs
                  if p["natural_mode"] == "dense_prefill" and p["baseline_ttft"]]
    speedups = [p["speedup"] for p in pairs if p["speedup"]]
    reuse_total_vals = [p.get("reuse_total") for p in pairs
                        if p["natural_mode"] == "kv_reuse" and p.get("reuse_total")]
    onhit_base_total_vals = [p.get("baseline_total") for p in pairs
                             if p["natural_mode"] == "kv_reuse" and p.get("baseline_total")]
    total_speedups = [p.get("total_speedup") for p in pairs if p.get("total_speedup")]

    natural = [r for r in records
               if r.get("agent_role") == VALIDATOR_ROLE and not r.get("measure_only")]
    out_tokens = sum(int(r.get("output_tokens") or 0) for r in natural)
    gen_time = sum(float(r.get("generation_total_latency") or 0.0) for r in natural)

    metrics: Dict[str, Any] = {}
    metrics_path = cell_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"[report] WARNING: malformed metrics.json skipped: {metrics_path}")
    passes = metrics.get("passes") or [{}]
    p0 = passes[0]

    kv_point = entry["kv_point"]
    return {
        "model": entry["model"],
        "mode": entry["mode"],
        "kv_threshold": kv_point["kv_threshold"],
        "kv_max_anchor_num": kv_point["kv_max_anchor_num"],
        "kv_window_size": kv_point["kv_window_size"],
        "rep_index": entry["rep_index"],
        "seed": entry["seed"],
        "natural_total": hits["natural_total"],
        "natural_hits": hits["hits"],
        "hit_rate": hits["hit_rate"],
        "reuse_mean_ttft": _mean(reuse_vals),
        "onhit_baseline_mean_ttft": _mean(onhit_base_vals),
        "onhit_mean_speedup": _mean(speedups),
        "natural_dense_mean_ttft": _mean(dense_vals),
        "reuse_mean_total_latency": _mean(reuse_total_vals),
        "onhit_baseline_mean_total_latency": _mean(onhit_base_total_vals),
        "onhit_mean_total_speedup": _mean(total_speedups),
        "validator_tokens_per_sec": (out_tokens / gen_time) if gen_time > 0 else None,
        "gpu_peak_mem_gb": (p0.get("gpu_peak_mem_bytes") / 2**30
                            if p0.get("gpu_peak_mem_bytes") is not None else None),
        "pass_duration_s": p0.get("duration_s"),
        "tasks_per_minute": p0.get("tasks_per_minute"),
        "energy_joules": p0.get("energy_joules"),
        "avg_power_w": p0.get("avg_power_w"),
        "kv_bytes_per_token": metrics.get("kv_bytes_per_token"),
    }


def _per_generation_rows(entry: Dict[str, Any], sweep_dir: Path) -> List[Dict[str, Any]]:
    cell_dir = sweep_dir / entry["cell_dir"]
    records = load_latency(str(cell_dir / "Latency.json"))
    rows = []
    for rec in records:
        rows.append({
            "model": entry["model"], "mode_cell": entry["mode"],
            "rep_index": entry["rep_index"], "seed": entry["seed"],
            "agent_role": rec.get("agent_role"), "gen_mode": rec.get("mode"),
            "measure_only": bool(rec.get("measure_only")),
            "ttft": rec.get("ttft"),
            "generation_ttft": rec.get("generation_ttft"),
            "generation_total_latency": rec.get("generation_total_latency"),
            "preprocess_latency": rec.get("preprocess_latency"),
            "input_tokens": rec.get("input_tokens"),
            "output_tokens": rec.get("output_tokens"),
            "anchor_hit": rec.get("anchor_hit"),
            "offset_fallback": rec.get("offset_fallback"),
        })
    return rows


def _write_df(df: pd.DataFrame, base: Path) -> None:
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(df.to_markdown(index=False), encoding="utf-8")


def _aggregate(df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    group_cols = ["model", "mode", "kv_threshold", "kv_max_anchor_num", "kv_window_size"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_reps"] = len(group)
        for metric in AGG_METRICS:
            values = [v for v in group[metric].tolist() if pd.notna(v)]
            mean, std, ci = mean_ci(values, confidence)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_model_matrix(agg: pd.DataFrame) -> pd.DataFrame:
    """Headline per-model matrix (honest-reference semantics, kvcomm-only).

    There is no independent dense-only run to merge in: --mode baseline
    executes the identical allow_kv_reuse natural pass as kvcomm (only the
    pass label differs), so it cannot serve as an independent dense
    reference. Instead the dense reference is derived from within the
    kvcomm pass itself — `natural_dense_mean_ttft_mean` is the mean TTFT of
    natural anchor-miss (dense_prefill) generations, which are unaffected
    by reuse and thus a valid baseline. `speedup_vs_natural_dense` compares
    that reference against the warm reuse TTFT.
    """
    kv = agg[agg["mode"] == "kvcomm"].copy()
    kv["speedup_vs_natural_dense"] = kv.apply(
        lambda r: (r["natural_dense_mean_ttft_mean"] / r["reuse_mean_ttft_mean"])
        if pd.notna(r.get("natural_dense_mean_ttft_mean")) and r.get("reuse_mean_ttft_mean")
        else None,
        axis=1,
    )
    cols = [
        "model", "kv_threshold", "kv_max_anchor_num", "kv_window_size", "n_reps",
        "hit_rate_mean", "reuse_mean_ttft_mean", "onhit_baseline_mean_ttft_mean",
        "onhit_mean_speedup_mean", "natural_dense_mean_ttft_mean",
        "speedup_vs_natural_dense", "onhit_mean_total_speedup_mean",
        "validator_tokens_per_sec_mean",
        "gpu_peak_mem_gb_mean", "energy_joules_mean",
    ]
    return kv[[c for c in cols if c in kv.columns]]


def _fig_bar(df: pd.DataFrame, value_col: str, err_col: Optional[str],
             title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = df["model"].tolist()
    values = df[value_col].fillna(0).tolist()
    errs = df[err_col].fillna(0).tolist() if err_col and err_col in df else None
    ax.bar(range(len(labels)), values, yerr=errs, capsize=4, color="#4878CF")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split("/")[-1] for l in labels], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _figures(agg: pd.DataFrame, gen_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    kv = agg[agg["mode"] == "kvcomm"]
    _fig_bar(kv, "reuse_mean_ttft_mean", "reuse_mean_ttft_ci95",
             "Warm (kv_reuse) TTFT by model", "TTFT (s)",
             fig_dir / "ttft_by_model.pdf")
    _fig_bar(kv, "onhit_mean_speedup_mean", "onhit_mean_speedup_ci95",
             "On-hit speedup (dense baseline / reuse) by model", "speedup (x)",
             fig_dir / "speedup_by_model.pdf")
    _fig_bar(kv, "gpu_peak_mem_gb_mean", "gpu_peak_mem_gb_ci95",
             "Peak GPU memory by model", "GB",
             fig_dir / "memory_by_model.pdf")
    # TTFT distribution: natural validator generations, reuse vs dense.
    fig, ax = plt.subplots(figsize=(7, 4))
    nat = gen_df[(gen_df["agent_role"] == VALIDATOR_ROLE) & (~gen_df["measure_only"])]
    data, labels = [], []
    for model, group in nat.groupby("model"):
        for mode in ("dense_prefill", "kv_reuse"):
            vals = group[group["gen_mode"] == mode]["ttft"].dropna().tolist()
            if vals:
                data.append(vals)
                labels.append(f"{model.split('/')[-1]}\n{mode}")
    if data:
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("TTFT (s)")
        ax.set_title("Validator TTFT distribution (natural generations)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ttft_distribution.pdf")
    plt.close(fig)


def generate_report(sweep_dir: Path, confidence: float = 0.95) -> Path:
    sweep_dir = Path(sweep_dir)
    manifest = json.loads((sweep_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["cells"]

    done = {cid: e for cid, e in entries.items() if e.get("status") == "done"}
    not_done = {cid: e for cid, e in entries.items() if e.get("status") != "done"}
    for cell_id, entry in not_done.items():
        print(f"[report] WARNING: cell not done (status={entry.get('status')}): {cell_id}")

    rows, gen_rows = [], []
    for cell_id, entry in done.items():
        try:
            rows.append(build_cell_row(entry, sweep_dir))
            gen_rows.extend(_per_generation_rows(entry, sweep_dir))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"[report] WARNING: skipping cell {cell_id}: {exc}")
    if not rows:
        raise SystemExit("[report] No completed cells with readable artifacts — nothing to report.")

    report_dir = sweep_dir / "report"
    (report_dir / "export").mkdir(parents=True, exist_ok=True)
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)

    per_run = pd.DataFrame(rows).sort_values(["model", "mode", "rep_index"])
    _write_df(per_run, report_dir / "per_run_comparison")

    agg = _aggregate(per_run, confidence)
    _write_df(agg, report_dir / "aggregate_stats")
    (report_dir / "tables" / "aggregate_stats.tex").write_text(
        agg.to_latex(index=False, float_format="%.4f"), encoding="utf-8")

    matrix = _cross_model_matrix(agg)
    _write_df(matrix, report_dir / "cross_model_matrix")
    (report_dir / "tables" / "cross_model_matrix.tex").write_text(
        matrix.to_latex(index=False, float_format="%.4f"), encoding="utf-8")

    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(report_dir / "export" / "per_generation.csv", index=False)
    gen_df.to_json(report_dir / "export" / "per_generation.json",
                   orient="records", indent=2)
    per_run.to_csv(report_dir / "export" / "per_cell.csv", index=False)
    per_run.to_json(report_dir / "export" / "per_cell.json",
                    orient="records", indent=2)

    _figures(agg, gen_df, report_dir / "figures")
    print(f"[report] report written to {report_dir}")
    return report_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper outputs from a sweep directory.")
    parser.add_argument("sweep_dir", help="Path to results/<sweep_name>.")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    generate_report(Path(args.sweep_dir), confidence=args.confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
