"""One command that turns the two recorded sweeps into everything the paper quotes.

The analysis is spread over four scripts that must run in a specific order, each
with a trap if run wrongly:

* ``make_kv_view.py``   - one view per kv point. Skipping this pools distinct
  configurations as if they were repetitions (see that module's docstring).
* ``paper_metrics.py``  - fidelity numbers, per view. Never per sweep root.
* ``compare_arms_cross_system.py`` - KVCOMM against CacheBlend, coverage-matched
  on both the task and the model axis.
* ``collect_reports.py`` - stage the summaries into ``docs/`` for committing.

Running them by hand is how the arms drifted apart in the first place. This
drives all four, in order, and is **idempotent**: a step whose output already
exists is skipped and reported as such, so re-invoking it collects only what is
missing. ``--force`` redoes everything.

It then writes one consolidated report holding both arms, every model, and an
explicit statement of which models each arm actually covers -- generated from
the coverage data rather than hardcoded, so it cannot go stale.

Usage::

    python experiments/build_paper_report.py            # uses the defaults below
    python experiments/build_paper_report.py --dry-run  # print the plan, run nothing
    python experiments/build_paper_report.py --force    # rebuild everything

The defaults point at the two trees this paper reports. Override with
``--kvcomm-root`` / ``--cacheblend-root`` for a different pair.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

from experiments.make_kv_view import discover_kv_points  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXP = _REPO_ROOT / "experiments"

DEFAULT_KVCOMM_ROOT = _REPO_ROOT / "results" / "MarginThresholdRun"
DEFAULT_CACHEBLEND_ROOT = _REPO_ROOT / "results" / "CacheBlendRun"
DEFAULT_KVCOMM_KV = "kv_t0.3_a20_w5"


# --------------------------------------------------------------------------
# step plumbing
# --------------------------------------------------------------------------

class Runner:
    """Runs steps, or prints them. Counts what it did so the tail can report it."""

    def __init__(self, dry_run: bool, force: bool) -> None:
        self.dry_run = dry_run
        self.force = force
        self.ran: List[str] = []
        self.skipped: List[str] = []
        self.failed: List[str] = []

    def step(self, label: str, produces: Path, cmd: List[str],
             stale_if: Optional[Any] = None) -> bool:
        """Run ``cmd`` unless ``produces`` already exists. Returns True if usable.

        ``stale_if`` is a predicate over the existing output. Presence alone is
        not proof an output is current -- an output written by an older version
        of the analysis is worse than a missing one, because it is reused
        silently.
        """
        if produces.exists() and not self.force:
            reason = stale_if(produces) if stale_if else None
            if not reason:
                print(f"  [skip] {label}  (have {_rel(produces)})")
                self.skipped.append(label)
                return True
            print(f"  [stale] {label}  ({reason})")
        if self.dry_run:
            print(f"  [plan] {label}")
            print(f"         {' '.join(str(c) for c in cmd)}")
            self.ran.append(label)
            return True
        print(f"  [run ] {label}")
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        if result.returncode != 0:
            print(f"  [FAIL] {label}  (exit {result.returncode})", file=sys.stderr)
            self.failed.append(label)
            return False
        self.ran.append(label)
        return True


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _view_root(sweep_root: Path, kv_slug: str) -> Path:
    """Mirrors the naming in make_kv_view.build_view."""
    return sweep_root.parent / f"{sweep_root.name}__view_{kv_slug}"


def _missing_matched_row(path: Path) -> Optional[str]:
    """Flag cross-system output written before the model-axis coverage fix.

    Those files carry a ``kvcomm`` row pooled over models CacheBlend never ran,
    and no ``kvcomm_matched`` row at all, so the only legitimate comparison is
    absent from them. Reusing one would quietly reinstate the bug it fixed.
    """
    payload = _load(path)
    if payload is None:
        return "unreadable"
    if "kvcomm_matched" not in (payload.get("overall") or {}):
        return "predates the model-axis coverage fix, no shared-set row"
    return None


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  warning: {_rel(path)} is not valid JSON ({exc})", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------

_ARM_HEADER = (
    "| scope | n | reports changed | applicable agreement | dense FAIL | "
    "abandoned | created | silencing |\n|---|---|---|---|---|---|---|---|"
)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _stat_row(label: str, stats: Optional[Dict[str, Any]]) -> str:
    if not stats:
        return f"| {label} | — | not measured | | | | | |"
    return (
        f"| {label} | {stats.get('n_cases', '—')} | "
        f"{_pct(stats.get('report_change_rate_pct'))} | "
        f"{_pct(stats.get('applicable_agreement_pct'))} | "
        f"{stats.get('dense_fail_total', '—')} | {stats.get('fail_abandoned', '—')} | "
        f"{stats.get('fail_created', '—')} | {_pct(stats.get('silencing_rate_pct'))} |"
    )


def _arm_section(title: str, cells: Dict[str, Dict[str, Any]], note: str) -> List[str]:
    """One section per arm: overall then per-model, for every cell of that arm."""
    lines = [f"## {title}", "", note, ""]
    if not cells:
        return lines + ["_No metrics found for this arm._", ""]
    for cell_slug, metrics in sorted(cells.items()):
        lines += [
            f"### `{cell_slug}`",
            "",
            _ARM_HEADER,
            _stat_row("all models", metrics.get("overall")),
        ]
        for model, stats in sorted((metrics.get("per_model") or {}).items()):
            lines.append(_stat_row(model, stats))
        floor = (metrics.get("noise_floor") or {})
        rep_floor = (metrics.get("cross_rep_noise_floor") or {})
        lines.append(
            _stat_row("noise floor (2nd dense)", floor.get("overall"))
            if floor.get("available")
            else "| noise floor (2nd dense) | — | not measured | | | | | |"
        )
        if rep_floor.get("available"):
            lines.append(_stat_row("noise floor (across reps)", rep_floor.get("overall")))
        lines += [
            "",
            f"Unique cases {metrics.get('n_unique_cases', '—')} "
            f"(from {metrics.get('n_raw_cases', '—')} paired folders).",
            "",
        ]
    return lines


def _coverage_section(cross: Dict[str, Dict[str, Any]]) -> List[str]:
    """State which models each arm ran, straight from the comparison's own coverage."""
    lines = ["## Coverage and exclusions", ""]
    only_kv: set = set()
    only_cb: set = set()
    for payload in cross.values():
        coverage = (payload.get("overall") or {}).get("coverage") or {}
        only_kv |= set(coverage.get("kvcomm_only_models") or [])
        only_cb |= set(coverage.get("cacheblend_only_models") or [])

    if not only_kv and not only_cb:
        lines += [
            "Both arms cover the same models. The matched rows differ only in "
            "which tasks KVCOMM's anchor gate admitted.",
            "",
        ]
        return lines

    if only_kv:
        lines += [
            f"**Run by KVCOMM only:** {', '.join(sorted(only_kv))}.",
            "",
            "The CacheBlend arm excludes these deliberately, for two reasons that "
            "are independent of each other and both disqualifying:",
            "",
            "1. *The engine cannot run them correctly.* The CacheBlend fork is "
            "pinned to a vLLM 0.4.1-era stack (`cacheblend/pin_env.sh`), which "
            "predates Llama-3.2. It lacks llama3 RoPE scaling and tied-embedding "
            "handling; the latter fails **silently**, loading uninitialised "
            "weights rather than raising. Numbers produced that way cannot be "
            "distinguished from the effect under study.",
            "2. *The corpus cannot support a replayed comparison.* CacheBlend "
            "rebuilds each prompt on a second stack, so it needs token-exact "
            "prompts. Per `experiments/corpus_health.py`, only 22/90 rebuild "
            "token-exactly for this model (37/90 were truncated at the token cap "
            "and 59/90 drift across kv points despite a dense greedy generator), "
            "leaving roughly 9-13 comparable cases per kv point — too few to "
            "clear any floor.",
            "",
            "Consequently the cross-system tables below are computed on the "
            "(model, task) pairs **both** arms cover, and the KVCOMM arm's own "
            "tables retain all of its models. Rows from the two are not "
            "interchangeable: only *KVCOMM, shared set* may be compared against "
            "*CacheBlend, KVCOMM's hit set*.",
            "",
        ]
    if only_cb:
        lines += [f"**Run by CacheBlend only:** {', '.join(sorted(only_cb))}.", ""]
    return lines


def _cross_section(cross: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [
        "## Cross-system: does the effect survive a different reuse mechanism?",
        "",
        "Each arm is scored against **its own** dense reference by identical "
        "code. Compare the two matched rows only — they cover the same models "
        "and the same tasks. The engine floor bounds how much of any remaining "
        "gap is HF-versus-vLLM rather than the reuse mechanism; it is not a "
        "noise floor.",
        "",
    ]
    if not cross:
        return lines + ["_No cross-system comparisons found._", ""]
    for cb_slug, payload in sorted(cross.items()):
        overall = payload.get("overall") or {}
        lines += [
            f"### KVCOMM vs `{cb_slug}`",
            "",
            _ARM_HEADER,
            _stat_row("KVCOMM, all models", overall.get("kvcomm")),
            _stat_row("**KVCOMM, shared set**", overall.get("kvcomm_matched")),
            _stat_row("**CacheBlend, KVCOMM's hit set**", overall.get("cacheblend_matched")),
            _stat_row("CacheBlend, all tasks", overall.get("cacheblend_full")),
            _stat_row("KVCOMM dense vs dense", overall.get("kvcomm_noise_floor")),
            _stat_row("CacheBlend dense vs dense", overall.get("cacheblend_noise_floor")),
            _stat_row("engine floor (HF vs vLLM dense)", overall.get("engine_floor")),
            "",
        ]
    return lines


def _latex_appendix(cross: Dict[str, Dict[str, Any]]) -> List[str]:
    """A paste-ready table of the only rows that may legitimately be differenced."""
    rows = []
    for cb_slug, payload in sorted(cross.items()):
        overall = payload.get("overall") or {}
        kv, cb = overall.get("kvcomm_matched"), overall.get("cacheblend_matched")
        if not kv or not cb:
            continue
        rows.append(
            f"{cb_slug.replace('_', chr(92) + '_')} & {kv['n_cases']} & "
            f"{kv['silencing_rate_pct']:.1f} & {cb['n_cases']} & "
            f"{cb['silencing_rate_pct']:.1f} \\\\"
        )
    if not rows:
        return []
    return [
        "## LaTeX",
        "",
        "Coverage-matched silencing, both arms, for the paper:",
        "",
        "```latex",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{KVCOMM} & \\multicolumn{2}{c}{CacheBlend} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Cell & $n$ & Silencing (\\%) & $n$ & Silencing (\\%) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "```",
        "",
        "Per-arm transition matrices are already emitted as "
        "`report/tables/transitions.tex` inside each view.",
        "",
    ]


def render_report(
    kvcomm_cells: Dict[str, Dict[str, Any]],
    cacheblend_cells: Dict[str, Dict[str, Any]],
    cross: Dict[str, Dict[str, Any]],
    provenance: Dict[str, Any],
) -> str:
    lines = [
        "# KV reuse and vulnerability silencing — consolidated results",
        "",
        "Generated by `experiments/build_paper_report.py`. Every number here is "
        "read from the JSON the analysis scripts wrote; this file computes "
        "nothing of its own.",
        "",
        "| | |",
        "|---|---|",
        f"| KVCOMM tree | `{provenance['kvcomm_root']}` |",
        f"| KVCOMM headline cell | `{provenance['kvcomm_kv']}` |",
        f"| CacheBlend tree | `{provenance['cacheblend_root']}` |",
        f"| git revision | `{provenance['git_rev']}` |",
        "",
    ]
    lines += _arm_section(
        "Arm 1 — KVCOMM (HF transformers, offset-corrected KV reuse)",
        kvcomm_cells,
        "Each cell is one entropy-gate setting. KVCOMM only forms a paired case "
        "on an anchor hit, so `n` is the hit set for that gate, not the corpus.",
    )
    lines += _arm_section(
        "Arm 2 — CacheBlend (vLLM 0.4.1 fork, selective recompute)",
        cacheblend_cells,
        "Each cell is one recompute ratio `r`. `r=0.00` is the maximally lossy "
        "anchor (suffix exact only) and `r=1.00` is a near-dense control — not "
        "dense, since layers 0-1 still read chunk-local KV. CacheBlend covers "
        "every task, so `n` is the corpus.",
    )
    lines += _cross_section(cross)
    lines += _coverage_section(cross)
    lines += _latex_appendix(cross)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_arm(runner: Runner, root: Path, label: str) -> Dict[str, Path]:
    """View + metrics for every cell in one sweep tree. Returns cell -> view root."""
    print(f"\n{label}: {_rel(root)}")
    if not root.is_dir():
        print(f"  missing tree, skipping arm: {_rel(root)}", file=sys.stderr)
        return {}
    cells = discover_kv_points(root)
    if not cells:
        print(f"  no cells discovered (is {_rel(root / 'manifest.json')} present?)",
              file=sys.stderr)
        return {}
    print(f"  cells: {', '.join(cells)}")

    views: Dict[str, Path] = {}
    for cell in cells:
        view = _view_root(root, cell)
        ok = runner.step(
            f"view {cell}",
            view / "manifest.json",
            [sys.executable, str(_EXP / "make_kv_view.py"), str(root), "--kv", cell],
        )
        if not ok:
            continue
        ok = runner.step(
            f"metrics {cell}",
            view / "report" / "paper_metrics.json",
            [sys.executable, str(_EXP / "paper_metrics.py"), str(view)],
        )
        if ok:
            views[cell] = view
    return views


def build_cross(
    runner: Runner, kvcomm_root: Path, kvcomm_kv: str,
    cacheblend_root: Path, cacheblend_cells: List[str],
) -> Dict[str, Path]:
    """Cross-system comparison of the headline KVCOMM cell against every CB cell."""
    print(f"\nCross-system: {kvcomm_kv} vs {len(cacheblend_cells)} CacheBlend cell(s)")
    out_dir = cacheblend_root / "report"
    produced: Dict[str, Path] = {}
    for cb_cell in cacheblend_cells:
        target = out_dir / f"cross_system_{kvcomm_kv}_vs_{cb_cell}.json"
        ok = runner.step(
            f"cross {kvcomm_kv} vs {cb_cell}",
            target,
            [
                sys.executable, str(_EXP / "compare_arms_cross_system.py"),
                "--kvcomm-root", str(kvcomm_root), "--kvcomm-kv", kvcomm_kv,
                "--cacheblend-root", str(cacheblend_root), "--cacheblend-kv", cb_cell,
            ],
            stale_if=_missing_matched_row,
        )
        if ok:
            produced[cb_cell] = target
    return produced


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kvcomm-root", type=Path, default=DEFAULT_KVCOMM_ROOT)
    parser.add_argument("--cacheblend-root", type=Path, default=DEFAULT_CACHEBLEND_ROOT)
    parser.add_argument(
        "--kvcomm-kv", default=None,
        help=f"KVCOMM cell to cross-compare (default: {DEFAULT_KVCOMM_KV} if present, "
             f"else the only cell)",
    )
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "docs" / "PAPER_RESULTS.md",
        help="consolidated report destination",
    )
    parser.add_argument("--force", action="store_true",
                        help="rebuild every step even if its output exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without running anything")
    parser.add_argument("--no-collect", action="store_true",
                        help="skip staging summaries into docs/experiment_reports")
    args = parser.parse_args(argv)

    runner = Runner(dry_run=args.dry_run, force=args.force)

    kv_views = build_arm(runner, args.kvcomm_root, "KVCOMM arm")
    cb_views = build_arm(runner, args.cacheblend_root, "CacheBlend arm")

    kvcomm_kv = args.kvcomm_kv
    if kvcomm_kv is None:
        if DEFAULT_KVCOMM_KV in kv_views:
            kvcomm_kv = DEFAULT_KVCOMM_KV
        elif len(kv_views) == 1:
            kvcomm_kv = next(iter(kv_views))
        elif kv_views:
            raise SystemExit(
                f"--kvcomm-kv required: {args.kvcomm_root.name} holds several cells "
                f"({', '.join(sorted(kv_views))}) and none is the default "
                f"{DEFAULT_KVCOMM_KV}. Pooling them would fold distinct "
                f"configurations together."
            )
    cross_files: Dict[str, Path] = {}
    if kvcomm_kv and cb_views:
        cross_files = build_cross(
            runner, args.kvcomm_root, kvcomm_kv,
            args.cacheblend_root, sorted(cb_views),
        )
    elif not cb_views:
        print("\nCross-system: skipped, no CacheBlend cells.", file=sys.stderr)

    if not args.no_collect:
        print("\nStaging summaries into docs/experiment_reports")
        trees = [args.kvcomm_root, args.cacheblend_root, *kv_views.values(),
                 *cb_views.values()]
        cmd = [sys.executable, str(_EXP / "collect_reports.py"),
               *[str(t) for t in trees if t.is_dir()]]
        if args.dry_run:
            print(f"  [plan] collect\n         {' '.join(cmd)}")
        else:
            subprocess.run(cmd, cwd=_REPO_ROOT)

    if args.dry_run:
        print(f"\nDry run: {len(runner.ran)} step(s) would run, "
              f"{len(runner.skipped)} already present. Nothing was written.")
        print(f"Consolidated report would be written to {_rel(args.out)}")
        return 0

    # ---- consolidate ----
    def _cells_metrics(views: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for cell, view in views.items():
            payload = _load(view / "report" / "paper_metrics.json")
            if payload:
                out[cell] = payload
        return out

    cross = {}
    for cb_cell, path in cross_files.items():
        payload = _load(path)
        if payload:
            cross[cb_cell] = payload

    report = render_report(
        _cells_metrics(kv_views),
        _cells_metrics(cb_views),
        cross,
        {
            "kvcomm_root": _rel(args.kvcomm_root),
            "kvcomm_kv": kvcomm_kv or "n/a",
            "cacheblend_root": _rel(args.cacheblend_root),
            "git_rev": _git_rev(),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(f"\n{len(runner.ran)} step(s) run, {len(runner.skipped)} already present, "
          f"{len(runner.failed)} failed.")
    for label in runner.failed:
        print(f"  FAILED: {label}", file=sys.stderr)
    print(f"[paper-report] wrote {_rel(args.out)}")
    return 1 if runner.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
