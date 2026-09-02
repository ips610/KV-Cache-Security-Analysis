"""Compare KV-reuse validator reports against their dense-baseline reports.

On a cache hit the harness saves two reports side by side in each task folder
(see ``experiments/secure_code_artifacts.py``):

* ``validation_report.md``                 — validator ran with kv_reuse prefill
* ``validation_report_dense_baseline.md``  — same task, dense prefill (ground truth)

This script walks a results tree, parses the per-rule **PASS** / **FAIL** /
**NOT APPLICABLE** verdicts out of every folder that contains both files, and
reports how far the KV-reuse report deviates from the dense baseline. Results are written into the
scanned tree as ``report_discrepancy_summary.md`` and
``report_discrepancy_summary.json``.

Usage:
    python experiments/compare_validation_reports.py <results_root>
    python experiments/compare_validation_reports.py paper_main_v2/paper_main_v2
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

KV_REPORT = "validation_report.md"
BASELINE_REPORT = "validation_report_dense_baseline.md"

# Rule headings: "#### Rule 1: ..." (strict format) or "**Rule 1: ...**"
# (older Llama-style output).
_RULE_RE = re.compile(r"^(?:#{2,6}\s*|\*\*)Rule\s*(\d+)", re.M)
# Verdicts: "**PASS**" / "**FAIL**" / "**NOT APPLICABLE**" (strict) or a bare
# verdict line; also tolerates the N/A and NOT-APPLICABLE spellings.
_VERDICT_PATTERN = r"(PASS|FAIL|NOT[ _-]APPLICABLE|N/A)"
_VERDICT_RE = re.compile(
    rf"(?:\*\*{_VERDICT_PATTERN}\*\*|^{_VERDICT_PATTERN}\b)", re.M
)


def _normalize_verdict(raw: str) -> str:
    verdict = raw.upper()
    if verdict in ("PASS", "FAIL"):
        return verdict
    return "NOT APPLICABLE"


def parse_verdicts(
    report_path: Path,
    *,
    rule_number_aliases: Optional[Mapping[int, int]] = None,
) -> Dict[int, str]:
    """Return ``{rule_number: "PASS"|"FAIL"|"NOT APPLICABLE"}`` from a report.

    The verdict is taken from the first verdict marker after each rule
    heading; summary sections later in the report are ignored because they
    occasionally contradict the per-rule verdict lines.
    """
    text = report_path.read_text(encoding="utf-8", errors="replace")
    verdicts: Dict[int, str] = {}
    headings = list(_RULE_RE.finditer(text))
    for i, heading in enumerate(headings):
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        match = _VERDICT_RE.search(text[start:end])
        raw_rule = int(heading.group(1))
        rule = rule_number_aliases.get(raw_rule, raw_rule) if rule_number_aliases else raw_rule
        if match and rule not in verdicts:
            verdicts[rule] = _normalize_verdict(match.group(1) or match.group(2))
    return verdicts


def find_pairs(root: Path) -> List[Path]:
    """Return every task folder under ``root`` holding both reports (cache hits)."""
    return sorted(
        p.parent
        for p in root.rglob(BASELINE_REPORT)
        if (p.parent / KV_REPORT).is_file()
    )


def compare_folder(folder: Path) -> Dict[str, object]:
    """Compare one task folder's KV-reuse report against its dense baseline."""
    baseline = parse_verdicts(folder / BASELINE_REPORT)
    kv = parse_verdicts(folder / KV_REPORT)
    rules: Dict[int, Dict[str, Optional[str]]] = {}
    for rule in sorted(set(baseline) | set(kv)):
        rules[rule] = {
            "baseline": baseline.get(rule),
            "kv_reuse": kv.get(rule),
            "match": baseline.get(rule) is not None
            and baseline.get(rule) == kv.get(rule),
        }
    comparable = [r for r in rules.values() if r["baseline"] and r["kv_reuse"]]
    agreed = sum(1 for r in comparable if r["match"])
    return {
        "folder": str(folder),
        "rules": {str(k): v for k, v in rules.items()},
        "rules_compared": len(comparable),
        "rules_agreed": agreed,
        "exact_match": bool(comparable) and agreed == len(comparable),
    }


def _flip_key(rule_info: Dict[str, Optional[str]]) -> str:
    return f"{rule_info['baseline']}->{rule_info['kv_reuse']}"


def summarize(root: Path, cases: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-case comparisons overall and per top-level model folder."""
    def new_bucket() -> Dict[str, object]:
        return {
            "cases": 0,
            "exact_match_cases": 0,
            "rules_compared": 0,
            "rules_agreed": 0,
            "flips": defaultdict(int),
            "per_rule": defaultdict(lambda: [0, 0]),  # rule -> [agreed, total]
        }

    overall = new_bucket()
    per_model: Dict[str, Dict[str, object]] = defaultdict(new_bucket)
    for case in cases:
        rel = Path(case["folder"]).relative_to(root)
        model = rel.parts[0] if rel.parts else "unknown"
        for bucket in (overall, per_model[model]):
            bucket["cases"] += 1
            bucket["exact_match_cases"] += case["exact_match"]
            bucket["rules_compared"] += case["rules_compared"]
            bucket["rules_agreed"] += case["rules_agreed"]
            for rule, info in case["rules"].items():
                if not (info["baseline"] and info["kv_reuse"]):
                    continue
                bucket["per_rule"][rule][1] += 1
                if info["match"]:
                    bucket["per_rule"][rule][0] += 1
                else:
                    bucket["flips"][_flip_key(info)] += 1

    def finalize(bucket: Dict[str, object]) -> Dict[str, object]:
        compared = bucket["rules_compared"]
        return {
            "cases": bucket["cases"],
            "exact_match_cases": bucket["exact_match_cases"],
            "rules_compared": compared,
            "rules_agreed": bucket["rules_agreed"],
            "agreement_pct": round(100 * bucket["rules_agreed"] / compared, 1)
            if compared
            else None,
            "flips": dict(bucket["flips"]),
            "per_rule_agreement": {
                rule: {"agreed": a, "total": n, "pct": round(100 * a / n, 1)}
                for rule, (a, n) in sorted(bucket["per_rule"].items())
            },
        }

    return {
        "results_root": str(root),
        "overall": finalize(overall),
        "per_model": {m: finalize(b) for m, b in sorted(per_model.items())},
        "cases": cases,
    }


def render_markdown(summary: Dict[str, object]) -> str:
    lines = ["# KV-reuse vs dense-baseline report discrepancy", ""]
    lines.append(
        "Dense baseline (`validation_report_dense_baseline.md`) is treated as "
        "ground truth; deviation is measured on the per-rule PASS/FAIL/NOT APPLICABLE verdicts "
        "of the cache-hit KV-reuse report (`validation_report.md`)."
    )
    lines.append("")

    def section(title: str, stats: Dict[str, object]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            f"- Paired (cache-hit) cases: {stats['cases']} "
            f"(exact match: {stats['exact_match_cases']})"
        )
        lines.append(
            f"- Rule-level agreement: {stats['rules_agreed']}/{stats['rules_compared']}"
            + (f" = {stats['agreement_pct']}%" if stats["agreement_pct"] is not None else "")
        )
        if stats["flips"]:
            flips = ", ".join(f"{k}: {v}" for k, v in sorted(stats["flips"].items()))
            lines.append(f"- Verdict flips (baseline->kv_reuse): {flips}")
        if stats["per_rule_agreement"]:
            lines.append("")
            lines.append("| Rule | Agreed | Total | Agreement |")
            lines.append("|---|---|---|---|")
            for rule, r in stats["per_rule_agreement"].items():
                lines.append(f"| {rule} | {r['agreed']} | {r['total']} | {r['pct']}% |")
        lines.append("")

    section("Overall", summary["overall"])
    for model, stats in summary["per_model"].items():
        section(model, stats)

    disagreements = [c for c in summary["cases"] if not c["exact_match"]]
    lines.append("## Disagreeing cases")
    lines.append("")
    if not disagreements:
        lines.append("None — every cache-hit report matched its dense baseline.")
    for case in disagreements:
        rel = Path(case["folder"]).relative_to(summary["results_root"])
        diffs = [
            f"Rule {rule}: baseline={info['baseline']}, kv_reuse={info['kv_reuse']}"
            for rule, info in case["rules"].items()
            if not info["match"]
        ]
        lines.append(f"- `{rel}` — " + "; ".join(diffs))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_root", type=Path, help="Run/results directory to scan")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without writing files into the results tree",
    )
    args = parser.parse_args()
    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    pairs = find_pairs(root)
    if not pairs:
        raise SystemExit(f"No folders under {root} contain both {KV_REPORT} and {BASELINE_REPORT}")

    summary = summarize(root, [compare_folder(folder) for folder in pairs])
    markdown = render_markdown(summary)
    print(markdown)
    if not args.no_write:
        (root / "report_discrepancy_summary.md").write_text(markdown, encoding="utf-8")
        (root / "report_discrepancy_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Wrote {root / 'report_discrepancy_summary.md'}")
        print(f"Wrote {root / 'report_discrepancy_summary.json'}")


if __name__ == "__main__":
    main()
