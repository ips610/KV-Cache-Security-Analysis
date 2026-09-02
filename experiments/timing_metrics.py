"""Headline timing metrics: TTFT speedup vs end-to-end speedup (T2 and F2).

The paper's second headline result is a deflation: KV reuse delivers a large
time-to-first-token win that mostly disappears once the full generation is
counted, because at long output lengths decode dominates wall-clock and the
prefill saving is amortized away. Reporting TTFT alone would overstate the
benefit by roughly 4x, so this module always reports the two together.

Pairing is per task and honest: a task's kv_reuse call is compared against the
measurement-only dense prefill of the SAME validation, via
``secure_code_report.pair_reuse_with_baseline``. Natural anchor misses have no
reuse to accelerate and are excluded from speedup (they are their own baseline),
but they are counted in the hit rate.

Usage:
    python experiments/timing_metrics.py results/FinalRun
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

from experiments.secure_code_report import pair_reuse_with_baseline

VALIDATOR_ROLE = "Security Validator"
_CELL_RE = re.compile(r"^[^/]+/[^/]+/[^/]+/rep_\d+$")


def _mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _load_latency(path: Path) -> List[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — one bad cell must not abort the report
        print(f"[timing_metrics] skipping unreadable {path}: {exc!r}")
        return []


def analyze_cell(path: Path) -> Optional[Dict[str, object]]:
    """Per-cell hit rate and paired TTFT / end-to-end speedups."""
    records = _load_latency(path)
    if not records:
        return None
    rows = pair_reuse_with_baseline(records, agent_role=VALIDATOR_ROLE)
    if not rows:
        return None

    hits = [r for r in rows if r.get("reuse_ttft") is not None]
    ttft_speedups = [r["speedup"] for r in hits if r.get("speedup")]
    total_speedups = [r["total_speedup"] for r in hits if r.get("total_speedup")]
    return {
        "tasks": len(rows),
        "hits": len(hits),
        "hit_rate_pct": round(100 * len(hits) / len(rows), 1) if rows else None,
        "mean_reuse_ttft": _mean([r["reuse_ttft"] for r in hits if r.get("reuse_ttft")]),
        "mean_dense_ttft": _mean([r["baseline_ttft"] for r in hits if r.get("baseline_ttft")]),
        "mean_ttft_speedup": _mean(ttft_speedups),
        "mean_reuse_total": _mean([r["reuse_total"] for r in hits if r.get("reuse_total")]),
        "mean_dense_total": _mean([r["baseline_total"] for r in hits if r.get("baseline_total")]),
        "mean_end_to_end_speedup": _mean(total_speedups),
    }


def analyze(root: Path) -> Dict[str, object]:
    """Aggregate cells to per-model means (averaging over repetitions)."""
    per_model: Dict[str, List[Dict]] = defaultdict(list)
    # Sort by the portable path string so report ordering is stable across
    # Windows (case-insensitive Path ordering) and Linux.
    for path in sorted(root.rglob("Latency.json"), key=lambda p: p.as_posix()):
        rel_path = path.parent.relative_to(root).as_posix()
        # Backups such as ``rep_0_interrupted`` may live beside a completed
        # retry. They are not sweep cells and must never enter aggregates.
        if not _CELL_RE.fullmatch(rel_path):
            continue
        rel = rel_path.split("/")
        cell = analyze_cell(path)
        if cell:
            cell["cell"] = path.parent.relative_to(root).as_posix()
            per_model[rel[0]].append(cell)

    def fold(cells: List[Dict]) -> Dict[str, object]:
        def avg(key: str) -> Optional[float]:
            vals = [c[key] for c in cells if c.get(key) is not None]
            return _mean(vals)
        return {
            "n_cells": len(cells),
            "hit_rate_pct": avg("hit_rate_pct"),
            "mean_reuse_ttft": avg("mean_reuse_ttft"),
            "mean_dense_ttft": avg("mean_dense_ttft"),
            "ttft_speedup": avg("mean_ttft_speedup"),
            "end_to_end_speedup": avg("mean_end_to_end_speedup"),
            "cells": cells,
        }

    models = {m: fold(cs) for m, cs in sorted(per_model.items())}
    all_cells = [c for cs in per_model.values() for c in cs]
    return {"per_model": models, "overall": fold(all_cells) if all_cells else None}


def render_markdown(res: Dict[str, object]) -> str:
    lines = ["# Timing metrics — TTFT vs end-to-end", ""]
    lines.append("`ttft_speedup` is paired dense-TTFT / reuse-TTFT on the same task. "
                 "`end_to_end_speedup` is the identical pairing over complete generation "
                 "time. The gap between the two columns is the paper's deflation result.")
    lines.append("")
    lines.append("| Model | Hit rate | Reuse TTFT (s) | Dense TTFT (s) | TTFT speedup | **End-to-end speedup** |")
    lines.append("|---|---|---|---|---|---|")

    def fmt(v, suffix="", digits=3):
        return f"{v:.{digits}f}{suffix}" if isinstance(v, (int, float)) else "—"

    for model, s in res["per_model"].items():
        lines.append(
            f"| {model} | {fmt(s['hit_rate_pct'], '%', 1)} | {fmt(s['mean_reuse_ttft'])} | "
            f"{fmt(s['mean_dense_ttft'])} | {fmt(s['ttft_speedup'], 'x', 2)} | "
            f"**{fmt(s['end_to_end_speedup'], 'x', 2)}** |"
        )
    if res.get("overall"):
        o = res["overall"]
        lines.append(f"| **Overall** | {fmt(o['hit_rate_pct'], '%', 1)} | {fmt(o['mean_reuse_ttft'])} | "
                     f"{fmt(o['mean_dense_ttft'])} | {fmt(o['ttft_speedup'], 'x', 2)} | "
                     f"**{fmt(o['end_to_end_speedup'], 'x', 2)}** |")
    lines.append("")
    lines.append("Per-cell detail:")
    lines.append("")
    lines.append("| Cell | Hit rate | TTFT speedup | End-to-end speedup |")
    lines.append("|---|---|---|---|")
    for model, s in res["per_model"].items():
        for cell in s["cells"]:
            lines.append(f"| {cell['cell']} | {fmt(cell['hit_rate_pct'], '%', 1)} | "
                         f"{fmt(cell['mean_ttft_speedup'], 'x', 2)} | "
                         f"{fmt(cell['mean_end_to_end_speedup'], 'x', 2)} |")
    lines.append("")
    return "\n".join(lines)


def write_figure(res: Dict[str, object], out_dir: Path) -> None:
    """F2 — the deflation figure: paired TTFT vs end-to-end bars per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 — figures are optional
        print(f"[timing_metrics] matplotlib unavailable, skipping figure: {exc!r}")
        return

    models = [m for m, s in res["per_model"].items()
              if s["ttft_speedup"] and s["end_to_end_speedup"]]
    if not models:
        return
    ttft = [res["per_model"][m]["ttft_speedup"] for m in models]
    e2e = [res["per_model"][m]["end_to_end_speedup"] for m in models]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = range(len(models))
    ax.bar([i - 0.2 for i in x], ttft, 0.4, label="TTFT speedup", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], e2e, 0.4, label="End-to-end speedup", color="#C44E52")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.split("_")[-1] for m in models], fontsize=8)
    ax.set_ylabel("Speedup vs dense (x)")
    for i, (a, b) in enumerate(zip(ttft, e2e)):
        ax.text(i - 0.2, a, f"{a:.2f}x", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, f"{b:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "f2_ttft_vs_end_to_end.pdf")
    plt.close(fig)
    print(f"[timing_metrics] figure -> {fig_dir / 'f2_ttft_vs_end_to_end.pdf'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_root", type=Path, help="Sweep root, e.g. results/FinalRun")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write outputs (default: <results_root>/report)")
    args = parser.parse_args(argv)

    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    res = analyze(root)
    if not res["per_model"]:
        raise SystemExit(f"No Latency.json files found under {root}")

    out_dir = (args.out_dir or (root / "report")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timing_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "timing_metrics.md").write_text(render_markdown(res), encoding="utf-8")
    write_figure(res, out_dir)

    for model, s in res["per_model"].items():
        ttft = s["ttft_speedup"]
        e2e = s["end_to_end_speedup"]
        print(f"[timing_metrics] {model:40s} hit {s['hit_rate_pct']}%  "
              f"TTFT {ttft:.2f}x  end-to-end {e2e:.2f}x"
              if ttft and e2e else f"[timing_metrics] {model}: incomplete")
    print(f"[timing_metrics] wrote {out_dir / 'timing_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
