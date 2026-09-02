"""Paper-ready fidelity metrics for the KV-reuse security-review study.

Consumes a sweep tree and emits every verdict-level number the paper reports:
unique-case counts, report-change rate, applicable-only agreement, the full
dense->reuse transition matrix (overall and per rule), the headline
**vulnerability silencing rate**, and the measured dense-vs-dense noise floor.

Three ideas drive the design:

1. **Unique cases, not repetitions.** Decoding is greedy and (with the
   determinism flags in ``run_secure_code.py``) reproducible, so the three
   repetitions of a task are not independent samples. Counting them as such
   would inflate n threefold. We collapse each (model, task) to one case and
   report ``n_unique``. A reproducibility guard records any task whose reps
   disagree instead of silently collapsing it.

2. **Applicable-only agreement is the honest number.** Overall agreement is
   inflated by rules that are NOT APPLICABLE in both arms — agreement on a
   non-judgment. Restricting to cases where the dense baseline actually ruled
   (PASS or FAIL) is the primary metric.

3. **The noise floor must be measured, not assumed.** When a run was made with
   ``--noise-floor-baseline true`` each anchor hit has two independent dense
   reports. Diffing them gives dense-vs-dense divergence on exactly the cases
   fidelity is reported over. Without it, "all divergence is due to KV reuse"
   is an assumption; with it, it is a measurement.

Usage:
    python experiments/paper_metrics.py results/FinalRun
    python experiments/paper_metrics.py results/FinalRun --out-dir custom/dir
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

from experiments.compare_validation_reports import parse_verdicts

KV_REPORT = "validation_report.md"
BASELINE_A = "validation_report_dense_baseline.md"
BASELINE_B = "validation_report_dense_baseline_b.md"

PASS, FAIL, NA = "PASS", "FAIL", "NOT APPLICABLE"
VERDICTS = (PASS, FAIL, NA)
RULED = (PASS, FAIL)  # dense actually made a judgment

# results/<sweep>/<model>/<mode>/<kv_slug>/rep_<n>/<pass>/<task_slug>
_CASE_RE = re.compile(r"^(?P<model>[^/]+)/(?P<mode>[^/]+)/(?P<kv>[^/]+)/rep_(?P<rep>\d+)/[^/]+/(?P<task>.+)$")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _rel_parts(folder: Path, root: Path) -> Optional[Dict[str, str]]:
    rel = folder.relative_to(root).as_posix()
    match = _CASE_RE.match(rel)
    return match.groupdict() if match else None


def load_cases(root: Path) -> List[Dict]:
    """Return one record per (model, kv, rep, task) folder holding a paired report."""
    cases: List[Dict] = []
    for baseline_path in sorted(root.rglob(BASELINE_A)):
        folder = baseline_path.parent
        if not (folder / KV_REPORT).is_file():
            continue
        parts = _rel_parts(folder, root)
        if parts is None:
            continue
        dense_b_path = folder / BASELINE_B
        cases.append({
            **parts,
            "folder": folder.relative_to(root).as_posix(),
            "dense": parse_verdicts(baseline_path),
            "reuse": parse_verdicts(folder / KV_REPORT),
            "dense_b": parse_verdicts(dense_b_path) if dense_b_path.is_file() else None,
        })
    return cases


def _fingerprint(case: Dict, arm: str) -> str:
    """Stable representation of a case's verdicts, for cross-rep comparison."""
    return json.dumps({str(k): v for k, v in sorted((case[arm] or {}).items())}, sort_keys=True)


def collapse_to_unique(cases: List[Dict]) -> Tuple[List[Dict], Dict[str, object]]:
    """Collapse repetitions of the same (model, task) into one case.

    Returns ``(unique_cases, reproducibility)``. Repetitions are expected to be
    identical; any that are not are counted and named in the reproducibility
    block rather than being averaged away, because a task whose own repetitions
    disagree cannot support a clean causal attribution to KV reuse.
    """
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for case in cases:
        grouped[(case["model"], case["task"])].append(case)

    unique: List[Dict] = []
    per_model = defaultdict(lambda: {"tasks": 0, "reproducible": 0, "unstable_tasks": []})
    for (model, task), reps in sorted(grouped.items()):
        stats = per_model[model]
        stats["tasks"] += 1
        stable = len({_fingerprint(r, "dense") + "|" + _fingerprint(r, "reuse") for r in reps}) == 1
        if stable:
            stats["reproducible"] += 1
        else:
            stats["unstable_tasks"].append(task)
        chosen = dict(reps[0])
        chosen["n_reps"] = len(reps)
        chosen["reps_identical"] = stable
        unique.append(chosen)

    repro = {
        "per_model": {
            m: {
                "tasks": s["tasks"],
                "reproducible": s["reproducible"],
                "pct": round(100 * s["reproducible"] / s["tasks"], 1) if s["tasks"] else None,
                "unstable_tasks": s["unstable_tasks"],
            }
            for m, s in sorted(per_model.items())
        }
    }
    total = sum(s["tasks"] for s in per_model.values())
    ok = sum(s["reproducible"] for s in per_model.values())
    repro["overall"] = {
        "tasks": total,
        "reproducible": ok,
        "pct": round(100 * ok / total, 1) if total else None,
    }
    return unique, repro


# --------------------------------------------------------------------------
# Core metric computation
# --------------------------------------------------------------------------

def compare_arms(cases: List[Dict], ref: str, test: str) -> Dict[str, object]:
    """Compute all fidelity metrics for ``test`` measured against ``ref``.

    Used twice: (dense -> reuse) for the paper's headline results, and
    (dense -> dense_b) for the noise floor, so both are computed by identical
    code and are directly comparable.
    """
    n_cases = 0
    changed_reports = 0
    total = agreed = 0
    applicable_total = applicable_agreed = 0
    transitions: Counter = Counter()
    per_rule_transitions: Dict[str, Counter] = defaultdict(Counter)
    per_rule: Dict[str, List[int]] = defaultdict(lambda: [0, 0])            # [agreed, total]
    per_rule_applicable: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    dense_verdict_counts: Counter = Counter()

    for case in cases:
        ref_v, test_v = case[ref], case[test]
        if not ref_v or not test_v:
            continue
        shared = sorted(set(ref_v) & set(test_v))
        if not shared:
            continue
        n_cases += 1
        case_changed = False
        for rule in shared:
            a, b = ref_v[rule], test_v[rule]
            key = str(rule)
            total += 1
            per_rule[key][1] += 1
            dense_verdict_counts[a] += 1
            match = a == b
            if match:
                agreed += 1
                per_rule[key][0] += 1
            else:
                case_changed = True
                transitions[f"{a}->{b}"] += 1
                per_rule_transitions[key][f"{a}->{b}"] += 1
            if a in RULED:
                applicable_total += 1
                per_rule_applicable[key][1] += 1
                if match:
                    applicable_agreed += 1
                    per_rule_applicable[key][0] += 1
        if case_changed:
            changed_reports += 1

    dense_fail = dense_verdict_counts[FAIL]
    silenced = transitions[f"{FAIL}->{PASS}"] + transitions[f"{FAIL}->{NA}"]
    created = transitions[f"{PASS}->{FAIL}"] + transitions[f"{NA}->{FAIL}"]

    def pct(num: int, den: int) -> Optional[float]:
        return round(100 * num / den, 1) if den else None

    return {
        "n_cases": n_cases,
        "reports_changed": changed_reports,
        "report_change_rate_pct": pct(changed_reports, n_cases),
        "exact_match_cases": n_cases - changed_reports,
        "rules_compared": total,
        "rules_agreed": agreed,
        "agreement_pct": pct(agreed, total),
        "applicable_compared": applicable_total,
        "applicable_agreed": applicable_agreed,
        "applicable_agreement_pct": pct(applicable_agreed, applicable_total),
        "dense_verdict_counts": dict(dense_verdict_counts),
        "transitions": dict(transitions),
        "per_rule_transitions": {k: dict(v) for k, v in sorted(per_rule_transitions.items(), key=lambda x: int(x[0]))},
        "dense_fail_total": dense_fail,
        "fail_abandoned": silenced,
        "fail_created": created,
        # The headline metric: of every FAIL the dense reference issued, what
        # fraction did the test arm stop reporting as a failure?
        "silencing_rate_pct": pct(silenced, dense_fail),
        "asymmetry_ratio": round(silenced / created, 2) if created else None,
        "per_rule_agreement": {
            k: {"agreed": v[0], "total": v[1], "pct": pct(v[0], v[1])}
            for k, v in sorted(per_rule.items(), key=lambda x: int(x[0]))
        },
        "per_rule_applicable_agreement": {
            k: {"agreed": v[0], "total": v[1], "pct": pct(v[0], v[1])}
            for k, v in sorted(per_rule_applicable.items(), key=lambda x: int(x[0]))
        },
    }


def cross_rep_noise_floor(cases: List[Dict]) -> Dict[str, object]:
    """Measure dense-vs-dense divergence by pairing repetitions of the same task.

    When the validator samples, each repetition is an independent dense draw for
    the same input, so repetition k's dense report versus repetition k+1's dense
    report is exactly a dense-vs-dense comparison — the noise floor, obtained
    without spending a second dense generation inside every cell.

    Under greedy decoding the repetitions are identical and this correctly
    reports a floor of zero.
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict]] = defaultdict(dict)
    for case in cases:
        grouped[(case["model"], case["kv"], case["task"])][case["rep"]] = case

    pairs: List[Dict] = []
    per_model: Dict[str, List[Dict]] = defaultdict(list)
    for (model, _kv, _task), reps in grouped.items():
        ordered = [reps[r] for r in sorted(reps)]
        for first, second in zip(ordered, ordered[1:]):
            pair = {"dense": first["dense"], "dense_other": second["dense"]}
            pairs.append(pair)
            per_model[model].append(pair)

    if not pairs:
        return {"available": False, "n_pairs": 0, "overall": None, "per_model": {}}
    return {
        "available": True,
        "n_pairs": len(pairs),
        "overall": compare_arms(pairs, "dense", "dense_other"),
        "per_model": {m: compare_arms(ps, "dense", "dense_other")
                      for m, ps in sorted(per_model.items())},
    }


def analyze(cases: List[Dict]) -> Dict[str, object]:
    """Full analysis: overall + per model, reuse fidelity + noise floor."""
    unique, repro = collapse_to_unique(cases)
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for case in unique:
        by_model[case["model"]].append(case)

    cross_rep = cross_rep_noise_floor(cases)
    with_b = [c for c in unique if c["dense_b"]]
    result = {
        "n_raw_cases": len(cases),
        "n_unique_cases": len(unique),
        "reproducibility": repro,
        "overall": compare_arms(unique, "dense", "reuse"),
        "per_model": {m: compare_arms(cs, "dense", "reuse") for m, cs in sorted(by_model.items())},
        "cross_rep_noise_floor": cross_rep,
        "noise_floor": {
            "available": bool(with_b),
            "n_cases_with_second_baseline": len(with_b),
            "overall": compare_arms(with_b, "dense", "dense_b") if with_b else None,
            "per_model": {
                m: compare_arms([c for c in cs if c["dense_b"]], "dense", "dense_b")
                for m, cs in sorted(by_model.items())
                if any(c["dense_b"] for c in cs)
            } if with_b else {},
        },
    }
    return result


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _transition_table(transitions: Dict[str, int], total_comparisons: int) -> List[str]:
    lines = ["| Transition | Count | Share of all comparisons |", "|---|---|---|"]
    for key, count in sorted(transitions.items(), key=lambda x: -x[1]):
        share = f"{100 * count / total_comparisons:.1f}%" if total_comparisons else "—"
        lines.append(f"| {key} | {count} | {share} |")
    return lines


def _headline_block(name: str, stats: Dict[str, object]) -> List[str]:
    lines = [f"### {name}", ""]
    lines.append(f"- Unique paired cases: **{stats['n_cases']}** "
                 f"(exact match: {stats['exact_match_cases']})")
    lines.append(f"- **Report-change rate: {stats['report_change_rate_pct']}%** "
                 f"({stats['reports_changed']}/{stats['n_cases']} reports altered)")
    lines.append(f"- Overall rule agreement: {stats['rules_agreed']}/{stats['rules_compared']} "
                 f"= {stats['agreement_pct']}%")
    lines.append(f"- **Applicable-only agreement (primary): "
                 f"{stats['applicable_agreed']}/{stats['applicable_compared']} "
                 f"= {stats['applicable_agreement_pct']}%**")
    lines.append(f"- Dense FAIL verdicts: {stats['dense_fail_total']}; "
                 f"abandoned: {stats['fail_abandoned']}; created: {stats['fail_created']}")
    lines.append(f"- **Vulnerability silencing rate: {stats['silencing_rate_pct']}%** "
                 f"(asymmetry {stats['asymmetry_ratio']}:1 toward suppression)")
    lines.append("")
    lines += _transition_table(stats["transitions"], stats["rules_compared"])
    lines.append("")
    lines.append("| Rule | Applicable agreement | Overall agreement |")
    lines.append("|---|---|---|")
    for rule, app in stats["per_rule_applicable_agreement"].items():
        overall = stats["per_rule_agreement"].get(rule, {})
        app_txt = f"{app['pct']}% ({app['agreed']}/{app['total']})" if app["total"] else "n/a"
        lines.append(f"| {rule} | {app_txt} | {overall.get('pct')}% ({overall.get('agreed')}/{overall.get('total')}) |")
    lines.append("")
    return lines


def render_markdown(res: Dict[str, object]) -> str:
    lines = ["# Paper fidelity metrics — KV reuse vs dense reference", ""]
    lines.append(f"Raw paired folders: {res['n_raw_cases']} -> "
                 f"**{res['n_unique_cases']} unique (model, task) cases** after collapsing repetitions.")
    lines.append("")

    repro = res["reproducibility"]
    lines.append("## Reproducibility across repetitions")
    lines.append("")
    lines.append(f"Overall: {repro['overall']['reproducible']}/{repro['overall']['tasks']} "
                 f"= {repro['overall']['pct']}% of tasks produced identical verdicts in every repetition.")
    lines.append("")
    lines.append("| Model | Reproducible tasks | % |")
    lines.append("|---|---|---|")
    for model, s in repro["per_model"].items():
        lines.append(f"| {model} | {s['reproducible']}/{s['tasks']} | {s['pct']}% |")
    lines.append("")
    if repro["overall"]["pct"] is not None and repro["overall"]["pct"] < 100:
        lines.append("> **Warning:** some tasks disagree across their own repetitions. Divergence on "
                     "those tasks cannot be attributed to KV reuse alone. Confirm the determinism "
                     "flags in `run_secure_code.py` were active for this run.")
        lines.append("")

    cr = res.get("cross_rep_noise_floor") or {}
    lines.append("## Noise floor A — dense vs dense across repetitions")
    lines.append("")
    if not cr.get("available"):
        lines.append("Not available — needs at least two repetitions per task.")
    else:
        c = cr["overall"]
        lines.append(f"Pairs compared: {cr['n_pairs']} (repetition k's dense report vs "
                     f"repetition k+1's, same task).")
        lines.append("")
        lines.append(f"- Report-change rate: **{c['report_change_rate_pct']}%**")
        lines.append(f"- Rule agreement: **{c['agreement_pct']}%**")
        lines.append(f"- Applicable-only agreement: **{c['applicable_agreement_pct']}%**")
        lines.append(f"- Silencing rate: **{c['silencing_rate_pct']}%**")
        lines.append("")
        lines.append("| Model | Change rate | Agreement | Silencing |")
        lines.append("|---|---|---|---|")
        for model, s in cr["per_model"].items():
            lines.append(f"| {model} | {s['report_change_rate_pct']}% | "
                         f"{s['agreement_pct']}% | {s['silencing_rate_pct']}% |")
        lines.append("")
        lines.append("Under greedy decoding this is 0% by construction. Under sampling it is "
                     "the bar the KV-reuse divergence below must clear to be a real effect.")
    lines.append("")

    nf = res["noise_floor"]
    lines.append("## Noise floor B — dense vs a second dense in the same cell")
    lines.append("")
    if not nf["available"]:
        lines.append("Not measured — this run recorded one dense report per task. That is "
                     "intentional when repetitions are stochastic, since noise floor A above "
                     "measures the same quantity without the extra generation. Enable "
                     "`--noise-floor-baseline true` only if you need a within-cell floor.")
    else:
        n = nf["overall"]
        lines.append(f"Measured on {nf['n_cases_with_second_baseline']} cases holding two independent "
                     f"dense reports.")
        lines.append("")
        lines.append(f"- Dense-vs-dense report-change rate: **{n['report_change_rate_pct']}%**")
        lines.append(f"- Dense-vs-dense rule agreement: **{n['agreement_pct']}%**")
        lines.append(f"- Dense-vs-dense silencing rate: **{n['silencing_rate_pct']}%**")
        lines.append("")
        lines.append("A zero (or near-zero) noise floor is what licenses attributing the divergence "
                     "below to KV reuse.")
    lines.append("")

    lines.append("## Overall (all models)")
    lines.append("")
    lines += _headline_block("Aggregate", res["overall"])

    lines.append("## Per model")
    lines.append("")
    for model, stats in res["per_model"].items():
        lines += _headline_block(model, stats)

    lines.append("## Per-rule transition matrices")
    lines.append("")
    for rule, trans in res["overall"]["per_rule_transitions"].items():
        total = res["overall"]["per_rule_agreement"].get(rule, {}).get("total", 0)
        lines.append(f"### Rule {rule}")
        lines.append("")
        lines += _transition_table(trans, total)
        lines.append("")
    return "\n".join(lines)


def render_latex_transition_table(stats: Dict[str, object]) -> str:
    # NOTE: the arrow substitution and the trailing row separator are built
    # outside the f-strings on purpose. Python < 3.12 rejects backslashes inside
    # f-string expressions, and the H100 runs 3.10.
    arrow = " $\\rightarrow$ "
    row_end = " \\\\"
    compared = stats["rules_compared"] or 1
    rows = "\n".join(
        arrow.join(key.split("->")) + f" & {count} & "
        + f"{100 * count / compared:.1f}" + "\\%" + row_end
        for key, count in sorted(stats["transitions"].items(), key=lambda x: -x[1])
    )
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Verdict transitions from the dense reference to KV reuse "
        f"($n={stats['n_cases']}$ unique paired cases).}}\n"
        "\\label{tab:transitions}\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Transition & Count & Share \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def write_figures(res: Dict[str, object], out_dir: Path) -> None:
    """F3 (per-rule applicable agreement by model) and F4 (speedup vs silencing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 — figures are optional
        print(f"[paper_metrics] matplotlib unavailable, skipping figures: {exc!r}")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = list(res["per_model"])
    rules = sorted(
        {r for m in models for r in res["per_model"][m]["per_rule_applicable_agreement"]},
        key=int,
    )
    if models and rules:
        fig, ax = plt.subplots(figsize=(9, 4))
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            per_rule = res["per_model"][model]["per_rule_applicable_agreement"]
            vals = [(per_rule.get(r) or {}).get("pct") or 0 for r in rules]
            ax.bar([x + i * width for x in range(len(rules))], vals, width, label=model.split("_")[-1])
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(rules))])
        ax.set_xticklabels([f"R{r}" for r in rules])
        ax.set_ylabel("Applicable-only agreement (%)")
        ax.set_xlabel("NIST rule")
        ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "f3_per_rule_applicable_agreement.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for model in models:
        stats = res["per_model"][model]
        change_rate = stats["report_change_rate_pct"]
        silencing_rate = stats["silencing_rate_pct"]
        # A model that emitted no dense FAIL verdicts has no silencing-rate
        # denominator.  Keep that value undefined in the metrics and omit the
        # point instead of passing None to Matplotlib (or misreporting it as 0).
        if change_rate is None or silencing_rate is None:
            continue
        ax.scatter(change_rate, silencing_rate, s=70)
        ax.annotate(model.split("_")[-1], (change_rate, silencing_rate),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Report-change rate (%)")
    ax.set_ylabel("Vulnerability silencing rate (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "f4_change_vs_silencing.pdf")
    plt.close(fig)
    print(f"[paper_metrics] figures -> {fig_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_root", type=Path, help="Sweep root, e.g. results/FinalRun")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write outputs (default: <results_root>/report)")
    args = parser.parse_args(argv)

    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    cases = load_cases(root)
    if not cases:
        raise SystemExit(
            f"No paired reports under {root}. Expected folders holding both "
            f"{KV_REPORT} and {BASELINE_A}."
        )

    res = analyze(cases)
    out_dir = (args.out_dir or (root / "report")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "paper_metrics.md").write_text(render_markdown(res), encoding="utf-8")
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "transitions.tex").write_text(
        render_latex_transition_table(res["overall"]), encoding="utf-8"
    )
    write_figures(res, out_dir)

    overall = res["overall"]
    print(f"[paper_metrics] {res['n_unique_cases']} unique cases "
          f"(from {res['n_raw_cases']} paired folders)")
    print(f"[paper_metrics] report-change rate    : {overall['report_change_rate_pct']}%")
    print(f"[paper_metrics] applicable agreement  : {overall['applicable_agreement_pct']}%")
    print(f"[paper_metrics] silencing rate        : {overall['silencing_rate_pct']}%")
    cr = res.get("cross_rep_noise_floor") or {}
    if cr.get("available"):
        print(f"[paper_metrics] noise floor (across reps): "
              f"{cr['overall']['report_change_rate_pct']}% change, "
              f"{cr['overall']['agreement_pct']}% agreement, n={cr['n_pairs']} pairs")
    if res["noise_floor"]["available"]:
        print(f"[paper_metrics] noise floor (2nd dense)  : "
              f"{res['noise_floor']['overall']['report_change_rate_pct']}%")
    elif not cr.get("available"):
        print("[paper_metrics] noise floor        : NOT MEASURED")
    print(f"[paper_metrics] wrote {out_dir / 'paper_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
