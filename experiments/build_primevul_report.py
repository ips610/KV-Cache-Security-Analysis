"""One command that turns the two PrimeVul runs into everything the arm reports.

The PrimeVul arm has two halves and they must not be confused:

* **behavioural** -- does reuse reproduce the dense validator's verdict? Scored
  by the existing ``paper_metrics.py`` on per-kv views, exactly as the VIBE arm
  is, because the report format is deliberately the same.
* **ground truth** -- require the report to identify the PrimeVul-labelled CWE,
  then audit all ten verdicts per artifact as a separately labelled closed-world
  matrix. Scored by ``primevul_metrics.py``, which needs BOTH arms at once.

Order matters. ``make_kv_view.py`` must run before ``paper_metrics.py`` or
distinct thresholds get pooled as if they were repetitions. This drives every
step in order and is idempotent: a step whose output exists is skipped and
reported as such. ``--force`` redoes everything.

Usage::

    python experiments/build_primevul_report.py
    python experiments/build_primevul_report.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root goes to the FRONT of sys.path, not the end: the local
# `datasets/` namespace package must win over any installed HuggingFace
# `datasets` in the environment, or `datasets.primevul_dataset` is
# shadowed and the import fails.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

from experiments.make_kv_view import discover_kv_points  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXP = _REPO_ROOT / "experiments"

DEFAULT_VULN_ROOT = _REPO_ROOT / "results" / "PrimeVulVulnRun"
DEFAULT_PATCHED_ROOT = _REPO_ROOT / "results" / "PrimeVulPatchedRun"
DEFAULT_REPORT = _REPO_ROOT / "docs" / "PRIMEVUL_RESULTS.md"


class Runner:
    """Runs steps, or prints them. Counts what it did so the tail can report it."""

    def __init__(self, dry_run: bool, force: bool) -> None:
        self.dry_run = dry_run
        self.force = force
        self.ran: List[str] = []
        self.skipped: List[str] = []
        self.failed: List[str] = []

    def step(self, label: str, produces: Path, cmd: List[str]) -> bool:
        if produces.exists() and not self.force:
            print(f"  [skip] {label}  (have {_rel(produces)})")
            self.skipped.append(label)
            return True
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


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_behavioural(runner: Runner, root: Path, label: str) -> Dict[str, Path]:
    """Per-kv views plus the shared fidelity metrics. Returns kv point -> view root."""
    print(f"\n{label}: {_rel(root)}")
    if not root.is_dir():
        print(f"  missing tree, skipping: {_rel(root)}", file=sys.stderr)
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
        if not runner.step(f"view {cell}", view / "manifest.json",
                           [sys.executable, str(_EXP / "make_kv_view.py"), str(root),
                            "--kv", cell]):
            continue
        if runner.step(f"fidelity {cell}", view / "report" / "paper_metrics.json",
                       [sys.executable, str(_EXP / "paper_metrics.py"), str(view)]):
            views[cell] = view
    return views


def _fmt(value: Optional[Dict[str, Any]]) -> str:
    if not value or value.get("pct") is None:
        return "n/a"
    ci = value.get("ci95")
    suffix = f" [{ci[0]:.1f}-{ci[1]:.1f}]" if ci else ""
    return f"{value['pct']:.1f}% ({value['k']}/{value['n']}){suffix}"


def render_report(ground_truth: Dict[str, Any], timing: Dict[str, Optional[Dict[str, Any]]],
                  sources: Dict[str, str], git_rev: str) -> str:
    lines = [
        "# PrimeVul ground-truth results",
        "",
        "Every number here is read from the JSON the analysis scripts wrote; this "
        "file computes nothing of its own.",
        "",
        f"- git revision: `{git_rev}`",
    ]
    for key, value in sources.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "Each report contains all ten CWE verdicts. Primary success requires the "
        "specific PrimeVul-labelled CWE to be identified; an unrelated FAIL does "
        "not count as a correct detection.",
        "",
        "## Labelled-CWE identification (primary)",
        "",
        "| Scope | artifacts | Dense target recall | Deployed target recall | "
        "Dense target specificity | Deployed target specificity | "
        "Dense exact-CWE-set | Deployed exact-CWE-set |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(label: str, block: Dict[str, Any]) -> str:
        gt = block["ground_truth"]
        target = gt["target_identification"]
        dense, deployed = target["dense"], target["deployed"]
        exact = gt["exact_cwe_set"]
        return (f"| {label} | {gt['n_paired_scored']} | {_fmt(dense['recall'])} | "
                f"{_fmt(deployed['recall'])} | {_fmt(dense['specificity'])} | "
                f"{_fmt(deployed['specificity'])} | {_fmt(exact['dense']['overall'])} | "
                f"{_fmt(exact['deployed']['overall'])} |")

    lines.append(row("**pooled**", ground_truth["overall"]))
    for kv, block in ground_truth.get("by_kv_point", {}).items():
        lines.append(row(kv, block))
    for name, block in ground_truth.get("per_cell", {}).items():
        lines.append(row(name, block))

    lines += ["", "## Ten-rule audit (closed-world assumption)", "",
              "This expands 100 pairs x 2 versions x 10 rules into 2,000 decisions "
              "per complete cell. PrimeVul verifies the selected CWE but does not "
              "independently annotate the other nine, so their negative labels are "
              "a stated closed-world assumption.", "",
              "| Scope | Rule decisions | Dense accuracy | Deployed accuracy | "
              "Dense recall | Deployed recall | Dense precision | Deployed precision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def rule_row(label: str, block: Dict[str, Any]) -> str:
        rules = block["ground_truth"]["rule_level_closed_world"]
        dense, deployed = rules["dense"], rules["deployed"]
        return (f"| {label} | {dense['n_scored']} | {_fmt(dense['accuracy'])} | "
                f"{_fmt(deployed['accuracy'])} | {_fmt(dense['recall'])} | "
                f"{_fmt(deployed['recall'])} | {_fmt(dense['precision'])} | "
                f"{_fmt(deployed['precision'])} |")

    lines.append(rule_row("**pooled**", ground_truth["overall"]))
    for kv, block in ground_truth.get("by_kv_point", {}).items():
        lines.append(rule_row(kv, block))
    for name, block in ground_truth.get("per_cell", {}).items():
        lines.append(rule_row(name, block))

    lines += ["", "### Per-rule closed-world audit (pooled)", "",
              "| Rule | Decisions | Positive labels | Dense recall | Deployed recall | "
              "Dense FPR | Deployed FPR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    per_rule_gt = ground_truth["overall"]["ground_truth"]["rule_level_closed_world"]["per_rule"]
    for cwe, block in per_rule_gt.items():
        dense, deployed = block["dense"], block["deployed"]
        positives = dense["confusion"]["tp"] + dense["confusion"]["fn"]
        lines.append(f"| Rule {block['rule_number']}: {cwe} | {dense['n_scored']} | "
                     f"{positives} | {_fmt(dense['recall'])} | {_fmt(deployed['recall'])} | "
                     f"{_fmt(dense['fpr'])} | {_fmt(deployed['fpr'])} |")

    lines += ["", "## Dense-to-reuse mechanism metrics (secondary)", "",
              "The dense-conditioned silencing values are retained for mechanism "
              "diagnosis, but they are not the dataset-ground-truth headline.", "",
              "| Scope | Gate hits | Dense recall | Deployed recall | S_GT deployed | S_GT gate-hit | "
              "Dense FPR | Deployed FPR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, block in [("**pooled**", ground_truth["overall"]), *ground_truth.get("by_kv_point", {}).items(), *ground_truth.get("per_cell", {}).items()]:
        v, p = block["vulnerable"], block["patched"]
        lines.append(f"| {label} | {_fmt(v['gate_hit_rate'])} | "
                     f"{_fmt(v['dense_recall'])} | {_fmt(v['reuse_recall'])} | "
                     f"{_fmt(v['gt_silencing_deployed'])} | {_fmt(v['gt_silencing_gate_hit'])} | "
                     f"{_fmt(p['dense_fpr'])} | {_fmt(p['reuse_fpr'])} |")

    lines += ["", "## Per source CWE (vulnerable arm, pooled over cells)", "",
              "| CWE | GT vulns | Dense detected | Reuse detected | Silenced | S_GT |",
              "|---|---:|---:|---:|---:|---:|"]
    for cwe, block in ground_truth["overall"]["per_cwe"].items():
        lines.append(f"| {cwe} | {block['gt_vulnerabilities']} | {block['dense_detected']} | "
                     f"{block['reuse_detected']} | {block['silenced']} | "
                     f"{_fmt(block['gt_silencing'])} |")

    lines += ["", "## Verdict transitions (dense -> reuse)", "",
              "| Arm | Transition | Count |", "|---|---|---:|"]
    for arm in ("vulnerable", "patched"):
        for transition, count in ground_truth["overall"][arm]["transitions"].items():
            lines.append(f"| {arm} | {transition} | {count} |")

    lines += ["", "## Parsing ledger", "",
              "Reports missing any of the ten rule verdicts are excluded from every "
              "rate above and counted here. The "
              "split by side is the load-bearing part: a gate miss has no reuse "
              "pass and can only fail on the dense path, so a gap between the dense "
              "and reuse columns says the approximation destroyed the report "
              "rather than that the model cannot follow the format.", "",
              "| Arm | Parse rate | Dense parsed | Reuse parsed | Reuse parsed (hits) | "
              "Excluded | By side | By gate |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for arm in ("vulnerable", "patched"):
        ledger = ground_truth["overall"]["parsing"][arm]
        lines.append(f"| {arm} | {_fmt(ledger['parse_rate'])} | "
                     f"{_fmt(ledger['dense_parse_rate'])} | "
                     f"{_fmt(ledger['reuse_parse_rate'])} | "
                     f"{_fmt(ledger['reuse_parse_rate_on_hits'])} | "
                     f"{ledger['excluded_count']} | {ledger['excluded_by_side'] or '-'} | "
                     f"{ledger['excluded_by_gate'] or '-'} |")

    lines += ["", "## Coverage", "",
              f"- vulnerable samples scored: {ground_truth['coverage']['vulnerable_samples']}",
              f"- patched samples scored: {ground_truth['coverage']['patched_samples']}",
              f"- samples present in both arms: {ground_truth['coverage']['paired_samples']}",
              ""]

    lines += ["## Efficiency", ""]
    for arm, data in timing.items():
        if not data:
            lines.append(f"- {arm}: timing_metrics.json not found")
            continue
        lines.append(f"### {arm}")
        lines.append("")
        lines.append("| Model | Hit rate | Reuse TTFT | Dense TTFT | TTFT speedup | End-to-end |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model, block in data.get("per_model", {}).items():
            lines.append(
                f"| {model} | {block.get('hit_rate_pct', 0):.1f}% | "
                f"{block.get('mean_reuse_ttft', 0):.3f} s | "
                f"{block.get('mean_dense_ttft', 0):.3f} s | "
                f"{block.get('ttft_speedup', 0):.2f}x | "
                f"{block.get('end_to_end_speedup', 0):.2f}x |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vuln-root", type=Path, default=DEFAULT_VULN_ROOT)
    parser.add_argument("--patched-root", type=Path, default=DEFAULT_PATCHED_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true", help="Rebuild every step.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing.")
    parser.add_argument("--no-collect", action="store_true",
                        help="Skip staging the summaries into docs/experiment_reports/.")
    args = parser.parse_args(argv)

    runner = Runner(dry_run=args.dry_run, force=args.force)

    # 1. behavioural half, per arm, per kv point (views first, always)
    build_behavioural(runner, args.vuln_root, "RUN A (vulnerable)")
    build_behavioural(runner, args.patched_root, "RUN B (patched)")

    # 2. timing, per arm (whole tree; timing does not need a view)
    timing: Dict[str, Optional[Dict[str, Any]]] = {}
    for label, root in (("vulnerable", args.vuln_root), ("patched", args.patched_root)):
        out = root / "report" / "timing_metrics.json"
        if root.is_dir():
            runner.step(f"timing {label}", out,
                        [sys.executable, str(_EXP / "timing_metrics.py"), str(root)])
        timing[label] = _load(out)

    # 3. ground truth, both arms at once
    gt_path = args.vuln_root / "report" / "primevul_metrics.json"
    runner.step("ground truth", gt_path,
                [sys.executable, str(_EXP / "primevul_metrics.py"),
                 "--vuln-root", str(args.vuln_root),
                 "--patched-root", str(args.patched_root)])
    ground_truth = _load(gt_path)

    # 4. consolidated report
    if ground_truth and not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(ground_truth, timing, ground_truth.get("sources", {}), _git_rev()),
            encoding="utf-8")
        print(f"\n[report] {_rel(args.report)}")
    elif not ground_truth and not args.dry_run:
        print("\n[report] skipped: ground-truth metrics missing", file=sys.stderr)

    # 5. stage summaries for committing (results/ is gitignored)
    if not args.no_collect:
        for root in (args.vuln_root, args.patched_root):
            if root.is_dir():
                runner.step(f"collect {root.name}",
                            _REPO_ROOT / "docs" / "experiment_reports" / root.name / "report",
                            [sys.executable, str(_EXP / "collect_reports.py"), str(root)])

    print(f"\n[summary] ran={len(runner.ran)} skipped={len(runner.skipped)} "
          f"failed={len(runner.failed)}")
    if runner.failed:
        for label in runner.failed:
            print(f"  FAILED: {label}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
