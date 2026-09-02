"""Ground-truth scoring for the PrimeVul arm.

The validator emits all ten CWE rule verdicts. Primary success requires the
specific PrimeVul-labelled CWE to be FAIL on vulnerable code and not FAIL on its
patched twin; an unrelated FAIL does not count. A second closed-world audit
expands 100 pairs x 2 versions x 10 rules into 2,000 decisions per cell.

    S_GT = |{y=1, D=1, R=0}| / |{y=1, D=1}|

Dense-conditioned silencing remains a secondary mechanism metric with two
denominators that are reported and never mixed:

* **deployed** -- every labelled-vulnerable sample the dense validator detected.
  A gate miss executes densely, so R == D there by construction; this is what a
  deployment with reuse switched on actually loses.
* **gate-hit** -- restricted to samples where the entropy gate fired and reuse
  really happened. This is the mechanism's own rate, and it is the honest number
  to compare against the VIBE arm's paired cases.

Reports missing any of the ten required rule verdicts are excluded from rates
and listed in an explicit ledger, so format failures cannot quietly move the
ground-truth values.

    python experiments/primevul_metrics.py \
        --vuln-root results/PrimeVulVulnRun --patched-root results/PrimeVulPatchedRun
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root goes to the FRONT of sys.path, not the end: the local
# `datasets/` namespace package must win over any installed HuggingFace
# `datasets` in the environment, or `datasets.primevul_dataset` is
# shadowed and the import fails.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap(require_kvcomm=True)

from experiments.compare_validation_reports import parse_verdicts  # noqa: E402
from experiments.secure_code_artifacts import task_slug  # noqa: E402
from datasets.primevul_dataset import DATA_DIR, load_tasks  # noqa: E402
from KVCOMM.prompt.primevul_common import CWE_IDS, CWE_RULE_NUMBER  # noqa: E402

KV_REPORT = "validation_report.md"
BASELINE_REPORT = "validation_report_dense_baseline.md"

# results/<sweep>/<model>/<mode>/<kv_slug>/rep_<n>/<pass>/<task_slug>
_CASE_RE = re.compile(
    r"^(?P<model>[^/]+)/(?P<mode>[^/]+)/(?P<kv>[^/]+)/rep_(?P<rep>\d+)/[^/]+/(?P<task>.+)$"
)

# A sample is detected when at least one of its ten rule blocks is FAIL.
DETECTED = "FAIL"
EXPECTED_RULES = frozenset(range(1, len(CWE_IDS) + 1))
CWE_NUMBER_TO_RULE = {
    int(cwe.removeprefix("CWE-")): CWE_RULE_NUMBER[cwe]
    for cwe in CWE_IDS
}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson(successes: int, total: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval; the normal approximation is unusable at these n."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def rate(successes: int, total: int) -> Dict[str, Any]:
    """A rate that always carries its numerator, denominator and interval."""
    if total <= 0:
        return {"pct": None, "n": 0, "k": successes, "ci95": None}
    low, high = wilson(successes, total)
    return {
        "pct": round(100.0 * successes / total, 1),
        "k": successes,
        "n": total,
        "ci95": [round(100.0 * low, 1), round(100.0 * high, 1)],
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _slug_to_sample(tasks_path: Path) -> Dict[str, Dict[str, Any]]:
    """Artifact folder slug -> task record, recomputed from the arm's task file.

    ``task_slug`` is deterministic in (index, task text), so this reproduces the
    run's folder names exactly without trusting anything written at run time.
    """
    return {task_slug(i, t["task"]): t for i, t in enumerate(load_tasks(str(tasks_path)))}


def load_arm(root: Path, tasks_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read every task folder under ``root`` into a scored case.

    A folder with a dense baseline is a gate HIT: the natural report is the
    reuse verdict and the baseline is the dense reference. A folder without one
    is a gate MISS: the natural pass ran densely, so it IS the dense verdict and
    the deployed reuse verdict equals it.
    """
    slug_map = _slug_to_sample(tasks_path)
    cases: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    stale_prompt_folders: List[str] = []

    for natural_path in sorted(root.rglob(KV_REPORT)):
        folder = natural_path.parent
        rel = folder.relative_to(root).as_posix()
        match = _CASE_RE.match(rel)
        if match is None:
            continue
        parts = match.groupdict()
        task = slug_map.get(parts["task"])
        if task is None:
            unmatched.append(rel)
            continue
        # Slugs intentionally use only a short task prefix, so the old
        # target-CWE prompt and the new neutral all-ten-rules prompt can share a
        # folder name. Refuse to rescore such stale outputs under the new
        # contract when the recorded task text is available.
        recorded_task_path = folder / "task.txt"
        if recorded_task_path.is_file():
            recorded_task = recorded_task_path.read_text(encoding="utf-8", errors="replace")
            if recorded_task != task["task"]:
                stale_prompt_folders.append(rel)
                continue

        baseline_path = folder / BASELINE_REPORT
        gate_hit = baseline_path.is_file()
        # Some otherwise complete reports label sections with catalogue IDs
        # (``Rule 119`` for CWE-119) instead of catalogue ordinals
        # (``Rule 1``). Both identify the same displayed rule, so normalize the
        # former before checking whether all ten verdicts are present.
        natural = parse_verdicts(
            natural_path, rule_number_aliases=CWE_NUMBER_TO_RULE
        )
        dense = (
            parse_verdicts(
                baseline_path, rule_number_aliases=CWE_NUMBER_TO_RULE
            )
            if gate_hit
            else natural
        )

        dense_verdict, dense_note = _classify_all_rules(dense)
        reuse_verdict, reuse_note = (_classify_all_rules(natural) if gate_hit
                                     else (dense_verdict, dense_note))

        label = task["label"]
        if label not in (0, 1):
            raise ValueError(f"{task['sample_id']}: PrimeVul label must be 0 or 1, got {label!r}")

        target_rule = task["cwe_rule_number"]
        dense_complete = dense_verdict is not None
        reuse_rules = natural if gate_hit else dense
        reuse_complete = reuse_verdict is not None
        cases.append({
            **parts,
            "folder": rel,
            "sample_id": task["sample_id"],
            "pair_id": task["pair_id"],
            "cwe": task["cwe"],
            "target_rule": target_rule,
            "label": label,
            "gate_hit": gate_hit,
            "dense": dense_verdict,
            "reuse": reuse_verdict,
            "dense_rule_verdicts": dense,
            "reuse_rule_verdicts": reuse_rules,
            # Ground-truth identification is about the dataset-labelled CWE,
            # not merely whether some rule somewhere was FAIL.
            "dense_target": dense.get(target_rule) if dense_complete else None,
            "reuse_target": reuse_rules.get(target_rule) if reuse_complete else None,
            "dense_failed_rules": (sorted(r for r, v in dense.items() if v == DETECTED)
                                   if dense_complete else None),
            "reuse_failed_rules": (sorted(r for r, v in reuse_rules.items() if v == DETECTED)
                                   if reuse_complete else None),
            "dense_parsed": dense_complete,
            "reuse_parsed": reuse_complete,
            "parse_notes": sorted({n for n in (dense_note, reuse_note) if n}),
        })

    return cases, {
        "unmatched_folders": unmatched,
        "stale_prompt_folders": stale_prompt_folders,
    }


def _classify_all_rules(
    verdicts: Dict[int, str],
) -> Tuple[Optional[str], Optional[str]]:
    """Convert the required ten rule verdicts into one binary prediction.

    PrimeVul supplies one binary label per code artifact. The validator therefore
    predicts vulnerable when *any* of the ten CWE rules is FAIL, and predicts
    non-vulnerable when all ten are PASS/NOT APPLICABLE. A report that omits,
    duplicates, or invents rule numbers is not silently treated as a clean sample.
    """
    present = set(verdicts)
    missing = sorted(EXPECTED_RULES - present)
    unexpected = sorted(present - EXPECTED_RULES)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append("missing_rules=" + ",".join(map(str, missing)))
        if unexpected:
            parts.append("unexpected_rules=" + ",".join(map(str, unexpected)))
        return None, ";".join(parts)
    return (DETECTED if any(verdicts[r] == DETECTED for r in EXPECTED_RULES)
            else "PASS"), None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _detected(verdict: Optional[str]) -> Optional[bool]:
    if verdict is None:
        return None
    return verdict == DETECTED


def _require_label(cases: List[Dict[str, Any]], expected: int, arm: str) -> None:
    wrong = [c["sample_id"] for c in cases if c.get("label") != expected]
    if wrong:
        preview = ", ".join(wrong[:5])
        raise ValueError(
            f"{arm} scorer received {len(wrong)} case(s) whose dataset label is not "
            f"{expected}: {preview}"
        )


def score_vulnerable(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recall, ground-truth silencing, and the dense->reuse transition matrix."""
    _require_label(cases, 1, "vulnerable")
    scored = [c for c in cases if c["dense_target"] and c["reuse_target"]]
    n = len(scored)
    dense_hits = [c for c in scored if _detected(c["dense_target"])]
    reuse_hits = [c for c in scored if _detected(c["reuse_target"])]
    silenced = [c for c in dense_hits if not _detected(c["reuse_target"])]

    gate = [c for c in scored if c["gate_hit"]]
    gate_dense_hits = [c for c in gate if _detected(c["dense_target"])]
    gate_silenced = [c for c in gate_dense_hits if not _detected(c["reuse_target"])]

    return {
        "n_scored": n,
        "n_excluded": len(cases) - n,
        # Over ALL cases: the gate fired independently of whether the resulting
        # report could be parsed, and counting only parsed cases here would hide
        # reuse executions whose output was destroyed.
        "gate_hit_rate": rate(len([c for c in cases if c["gate_hit"]]), len(cases)),
        "gate_hit_rate_scored_only": rate(len(gate), n),
        "dense_recall": rate(len(dense_hits), n),
        "reuse_recall": rate(len(reuse_hits), n),
        "dense_false_negatives": n - len(dense_hits),
        "reuse_false_negatives": n - len(reuse_hits),
        # The headline. Both denominators, always labelled.
        "gt_silencing_deployed": rate(len(silenced), len(dense_hits)),
        "gt_silencing_gate_hit": rate(len(gate_silenced), len(gate_dense_hits)),
        "transitions": _transitions(scored),
        "transitions_gate_hit": _transitions(gate),
    }


def score_patched(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Specificity, false-positive rate, and reuse-induced false alarms."""
    _require_label(cases, 0, "patched")
    scored = [c for c in cases if c["dense_target"] and c["reuse_target"]]
    n = len(scored)
    dense_clean = [c for c in scored if not _detected(c["dense_target"])]
    reuse_clean = [c for c in scored if not _detected(c["reuse_target"])]
    induced = [c for c in dense_clean if _detected(c["reuse_target"])]

    gate = [c for c in scored if c["gate_hit"]]
    gate_clean = [c for c in gate if not _detected(c["dense_target"])]
    gate_induced = [c for c in gate_clean if _detected(c["reuse_target"])]

    return {
        "n_scored": n,
        "n_excluded": len(cases) - n,
        "gate_hit_rate": rate(len([c for c in cases if c["gate_hit"]]), len(cases)),
        "gate_hit_rate_scored_only": rate(len(gate), n),
        "dense_specificity": rate(len(dense_clean), n),
        "reuse_specificity": rate(len(reuse_clean), n),
        "dense_fpr": rate(n - len(dense_clean), n),
        "reuse_fpr": rate(n - len(reuse_clean), n),
        "reuse_induced_false_alarms_deployed": rate(len(induced), len(dense_clean)),
        "reuse_induced_false_alarms_gate_hit": rate(len(gate_induced), len(gate_clean)),
        "transitions": _transitions(scored),
        "transitions_gate_hit": _transitions(gate),
    }


def _score_predictions(rows: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Binary metrics for ``(ground_truth, prediction)`` rows."""
    tp = sum(y and pred for y, pred in rows)
    fn = sum(y and not pred for y, pred in rows)
    tn = sum(not y and not pred for y, pred in rows)
    fp = sum(not y and pred for y, pred in rows)
    return {
        "n_scored": len(rows),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "accuracy": rate(tp + tn, len(rows)),
        "recall": rate(tp, tp + fn),
        "specificity": rate(tn, tn + fp),
        "precision": rate(tp, tp + fp),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "fpr": rate(fp, fp + tn),
    }


def _sample_side(cases: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    """Legacy sample-level prediction: any FAIL means vulnerable."""
    return _score_predictions([
        (c["label"] == 1, _detected(c[side]))
        for c in cases
    ])


def _target_side(cases: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    """Whether the validator found the specific PrimeVul-labelled CWE."""
    key = f"{side}_target"
    return _score_predictions([
        (c["label"] == 1, _detected(c[key]))
        for c in cases
    ])


def _rule_side(
    cases: List[Dict[str, Any]], side: str, rules: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Expand artifacts to ten rule rows under the closed-world label vector."""
    key = f"{side}_rule_verdicts"
    rows: List[Tuple[bool, bool]] = []
    selected_rules = rules or sorted(EXPECTED_RULES)
    for case in cases:
        for rule in selected_rules:
            expected = case["label"] == 1 and rule == case["target_rule"]
            rows.append((expected, _detected(case[key][rule])))
    return _score_predictions(rows)


def _exact_identification(cases: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    """Exact predicted CWE set: {target} for vulnerable, empty for patched."""
    key = f"{side}_failed_rules"
    correct = 0
    vulnerable_correct = 0
    patched_correct = 0
    vulnerable_n = sum(c["label"] == 1 for c in cases)
    patched_n = sum(c["label"] == 0 for c in cases)
    for case in cases:
        expected = {case["target_rule"]} if case["label"] == 1 else set()
        matched = set(case[key]) == expected
        correct += matched
        vulnerable_correct += matched and case["label"] == 1
        patched_correct += matched and case["label"] == 0
    return {
        "overall": rate(correct, len(cases)),
        "vulnerable": rate(vulnerable_correct, vulnerable_n),
        "patched": rate(patched_correct, patched_n),
    }


def score_ground_truth(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Target identification plus a transparent 10-rule closed-world audit."""
    bad = [c["sample_id"] for c in cases if c.get("label") not in (0, 1)]
    if bad:
        raise ValueError(f"invalid PrimeVul labels for: {', '.join(bad[:5])}")
    # Use the common complete-report subset for paired dense/deployed comparison.
    paired = [c for c in cases if c.get("dense") is not None and c.get("reuse") is not None]
    per_rule = {}
    for rule, cwe in enumerate(CWE_IDS, start=1):
        per_rule[cwe] = {
            "rule_number": rule,
            "dense": _rule_side(paired, "dense", [rule]),
            "deployed": _rule_side(paired, "reuse", [rule]),
        }

    return {
        "n_cases": len(cases),
        "n_paired_scored": len(paired),
        "n_excluded": len(cases) - len(paired),
        "positive_labels": sum(c["label"] == 1 for c in cases),
        "negative_labels": sum(c["label"] == 0 for c in cases),
        # Primary: the named PrimeVul CWE must itself be FAIL on vulnerable code
        # and not FAIL on its patched twin.
        "target_identification": {
            "dense": _target_side(paired, "dense"),
            "deployed": _target_side(paired, "reuse"),
        },
        # Stronger artifact-level metric. This treats PrimeVul's selected CWE as
        # the only positive class for the vulnerable artifact and none as
        # positive for the patch.
        "exact_cwe_set": {
            "dense": _exact_identification(paired, "dense"),
            "deployed": _exact_identification(paired, "reuse"),
        },
        # 200 artifacts x 10 rules = 2,000 rows per cell. The nine non-target
        # classes are not independently annotated by PrimeVul, so this block is
        # explicitly named closed-world rather than presented as verified GT.
        "rule_level_closed_world": {
            "n_expected_per_complete_cell": 2000,
            "dense": _rule_side(paired, "dense"),
            "deployed": _rule_side(paired, "reuse"),
            "per_rule": per_rule,
        },
        # Kept only to show how the previous any-FAIL reduction differs from
        # correct CWE identification.
        "sample_any_fail": {
            "dense": _sample_side(paired, "dense"),
            "deployed": _sample_side(paired, "reuse"),
        },
    }


def _transitions(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for case in cases:
        if not (case["dense_target"] and case["reuse_target"]):
            continue
        counts[f"{case['dense_target']}->{case['reuse_target']}"] += 1
    return dict(sorted(counts.items()))


def per_cwe(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-CWE detection and silencing, numerators and denominators in every cell."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["cwe"]].append(case)
    table = {}
    for cwe, group in sorted(grouped.items()):
        scored = [c for c in group if c["dense_target"] and c["reuse_target"]]
        dense_hits = [c for c in scored if _detected(c["dense_target"])]
        reuse_hits = [c for c in scored if _detected(c["reuse_target"])]
        silenced = [c for c in dense_hits if not _detected(c["reuse_target"])]
        table[cwe] = {
            "gt_vulnerabilities": len(group),
            "scored": len(scored),
            "dense_detected": len(dense_hits),
            "reuse_detected": len(reuse_hits),
            "silenced": len(silenced),
            "gt_silencing": rate(len(silenced), len(dense_hits)),
        }
    return table


def parse_ledger(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every report that could not be scored, and why. Never silently dropped."""
    notes: Counter = Counter()
    excluded: List[Dict[str, str]] = []
    by_side: Counter = Counter()
    by_gate: Counter = Counter()
    for case in cases:
        for note in case["parse_notes"]:
            notes[note] += 1
        if case["dense_parsed"] and case["reuse_parsed"]:
            continue
        if not case["dense_parsed"] and not case["reuse_parsed"]:
            side = "both"
        elif not case["reuse_parsed"]:
            side = "reuse_only"
        else:
            side = "dense_only"
        by_side[side] += 1
        by_gate["gate_hit" if case["gate_hit"] else "gate_miss"] += 1
        excluded.append({"folder": case["folder"], "sample_id": case["sample_id"],
                         "side": side, "gate_hit": case["gate_hit"],
                         "notes": ",".join(case["parse_notes"]) or "no_verdict"})
    # On a gate miss the natural report IS the dense verdict, so a miss can only
    # fail on the dense path. Splitting the exclusions this way is what shows
    # whether unparsable output is a property of reuse or of the model.
    return {
        "notes": dict(notes),
        "excluded_count": len(excluded),
        "excluded_by_side": dict(by_side),
        "excluded_by_gate": dict(by_gate),
        "dense_parse_rate": rate(sum(1 for c in cases if c["dense_parsed"]), len(cases)),
        "reuse_parse_rate": rate(sum(1 for c in cases if c["reuse_parsed"]), len(cases)),
        "reuse_parse_rate_on_hits": rate(
            sum(1 for c in cases if c["gate_hit"] and c["reuse_parsed"]),
            sum(1 for c in cases if c["gate_hit"])),
        "excluded": excluded,
        "parse_rate": rate(len(cases) - len(excluded), len(cases)),
    }


def analyse(vuln_cases: List[Dict[str, Any]], patched_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    def by_cell(cases):
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for case in cases:
            grouped[(case["model"], case["kv"])].append(case)
        return grouped

    vuln_cells, patched_cells = by_cell(vuln_cases), by_cell(patched_cases)
    cells: Dict[str, Any] = {}
    for key in sorted(set(vuln_cells) | set(patched_cells)):
        model, kv = key
        v, p = vuln_cells.get(key, []), patched_cells.get(key, [])
        cells[f"{model}|{kv}"] = {
            "model": model,
            "kv_point": kv,
            "ground_truth": score_ground_truth(v + p),
            "vulnerable": score_vulnerable(v),
            "patched": score_patched(p),
            "per_cwe": per_cwe(v),
            "parsing": {"vulnerable": parse_ledger(v), "patched": parse_ledger(p)},
        }

    by_kv: Dict[str, Any] = {}
    for kv in sorted({c["kv"] for c in vuln_cases} | {c["kv"] for c in patched_cases}):
        v = [c for c in vuln_cases if c["kv"] == kv]
        p = [c for c in patched_cases if c["kv"] == kv]
        by_kv[kv] = {
            "ground_truth": score_ground_truth(v + p),
            "vulnerable": score_vulnerable(v),
            "patched": score_patched(p),
            "per_cwe": per_cwe(v),
        }

    return {
        "schema_version": 3,
        "overall": {
            "ground_truth": score_ground_truth(vuln_cases + patched_cases),
            "vulnerable": score_vulnerable(vuln_cases),
            "patched": score_patched(patched_cases),
            "per_cwe": per_cwe(vuln_cases),
            "parsing": {"vulnerable": parse_ledger(vuln_cases),
                        "patched": parse_ledger(patched_cases)},
        },
        "by_kv_point": by_kv,
        "per_cell": cells,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt(value: Dict[str, Any]) -> str:
    if not value or value.get("pct") is None:
        return "n/a"
    return f"{value['pct']:.1f}% ({value['k']}/{value['n']})"


def render_markdown(result: Dict[str, Any], sources: Dict[str, str]) -> str:
    lines = ["# PrimeVul ground-truth results", ""]
    lines.append("Primary ground truth requires the validator to identify the specific ")
    lines.append("PrimeVul-labelled CWE. Merely failing an unrelated rule is not a hit. ")
    lines.append("All ten verdicts are also expanded into a 2,000-row closed-world audit ")
    lines.append("per complete cell. `Deployed` is KV reuse on a hit and dense fallback on a miss.")
    lines.append("")
    for key, value in sources.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines += ["## Labelled-CWE identification (primary)", "",
              "| Scope | artifacts | Dense target recall | Deployed target recall | "
              "Dense target specificity | Deployed target specificity | "
              "Dense exact-CWE-set | Deployed exact-CWE-set |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def gt_row(label: str, block: Dict[str, Any]) -> str:
        gt = block["ground_truth"]
        target = gt["target_identification"]
        dense, deployed = target["dense"], target["deployed"]
        exact = gt["exact_cwe_set"]
        return (f"| {label} | {gt['n_paired_scored']} | {_fmt(dense['recall'])} | "
                f"{_fmt(deployed['recall'])} | {_fmt(dense['specificity'])} | "
                f"{_fmt(deployed['specificity'])} | {_fmt(exact['dense']['overall'])} | "
                f"{_fmt(exact['deployed']['overall'])} |")

    lines.append(gt_row("all cells", result["overall"]))
    for kv, block in result["by_kv_point"].items():
        lines.append(gt_row(kv, block))
    for name, block in result["per_cell"].items():
        lines.append(gt_row(name, block))
    lines.append("")

    lines += ["## Ten-rule audit (closed-world assumption)", "",
              "PrimeVul verifies the selected CWE, but does not independently annotate "
              "the other nine classes. The table below treats those nine as negative "
              "only to provide the requested 100 x 2 x 10 audit.", "",
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

    lines.append(rule_row("all cells", result["overall"]))
    for kv, block in result["by_kv_point"].items():
        lines.append(rule_row(kv, block))
    for name, block in result["per_cell"].items():
        lines.append(rule_row(name, block))
    lines.append("")

    lines += ["### Per-rule closed-world audit (pooled)", "",
              "| Rule | Decisions | Positive labels | Dense recall | Deployed recall | "
              "Dense FPR | Deployed FPR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    per_rule_gt = result["overall"]["ground_truth"]["rule_level_closed_world"]["per_rule"]
    for cwe, block in per_rule_gt.items():
        dense, deployed = block["dense"], block["deployed"]
        positives = dense["confusion"]["tp"] + dense["confusion"]["fn"]
        lines.append(f"| Rule {block['rule_number']}: {cwe} | {dense['n_scored']} | "
                     f"{positives} | {_fmt(dense['recall'])} | {_fmt(deployed['recall'])} | "
                     f"{_fmt(dense['fpr'])} | {_fmt(deployed['fpr'])} |")
    lines.append("")

    lines += ["## Dense-to-reuse mechanism metrics (secondary)", "",
              "| Scope | n | Hit rate | Dense recall | Reuse recall | "
              "S_GT (deployed) | S_GT (gate-hit) | Dense FPR | Reuse FPR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(label: str, block: Dict[str, Any]) -> str:
        v, p = block["vulnerable"], block["patched"]
        return (f"| {label} | {v['n_scored']} | {_fmt(v['gate_hit_rate'])} | "
                f"{_fmt(v['dense_recall'])} | {_fmt(v['reuse_recall'])} | "
                f"{_fmt(v['gt_silencing_deployed'])} | {_fmt(v['gt_silencing_gate_hit'])} | "
                f"{_fmt(p['dense_fpr'])} | {_fmt(p['reuse_fpr'])} |")

    lines.append(row("all cells", result["overall"]))
    for kv, block in result["by_kv_point"].items():
        lines.append(row(kv, block))
    for name, block in result["per_cell"].items():
        lines.append(row(name, block))
    lines.append("")

    lines += ["## Per CWE (pooled over cells, vulnerable arm)", "",
              "| CWE | GT vulns | Dense detected | Reuse detected | Silenced | S_GT |",
              "|---|---:|---:|---:|---:|---:|"]
    for cwe, block in result["overall"]["per_cwe"].items():
        lines.append(f"| {cwe} | {block['gt_vulnerabilities']} | {block['dense_detected']} | "
                     f"{block['reuse_detected']} | {block['silenced']} | "
                     f"{_fmt(block['gt_silencing'])} |")
    lines.append("")

    lines += ["## Transitions (dense -> reuse)", "",
              "| Arm | Transition | Count |", "|---|---|---:|"]
    for arm in ("vulnerable", "patched"):
        for transition, count in result["overall"][arm]["transitions"].items():
            lines.append(f"| {arm} | {transition} | {count} |")
    lines.append("")

    lines += ["## Reuse-induced false alarms (patched arm)", "",
              f"- deployed: {_fmt(result['overall']['patched']['reuse_induced_false_alarms_deployed'])}",
              f"- gate-hit: {_fmt(result['overall']['patched']['reuse_induced_false_alarms_gate_hit'])}",
              ""]

    lines += ["## Parsing ledger", "",
              "A report is parseable only when all ten rule verdicts are present. "
              "Split by which side failed. A gate miss has no reuse pass, so it can "
              "only fail on the dense path; comparing the two columns is what "
              "distinguishes an unparsable model from unparsable reuse.", "",
              "| Arm | Parse rate | Dense parsed | Reuse parsed | Reuse parsed (hits) | "
              "Excluded | By side | Notes |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for arm in ("vulnerable", "patched"):
        ledger = result["overall"]["parsing"][arm]
        lines.append(f"| {arm} | {_fmt(ledger['parse_rate'])} | "
                     f"{_fmt(ledger['dense_parse_rate'])} | "
                     f"{_fmt(ledger['reuse_parse_rate'])} | "
                     f"{_fmt(ledger['reuse_parse_rate_on_hits'])} | "
                     f"{ledger['excluded_count']} | {ledger['excluded_by_side'] or '-'} | "
                     f"{ledger['notes'] or '-'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vuln-root", required=True, help="RUN A tree (or a single-kv view).")
    parser.add_argument("--patched-root", required=True, help="RUN B tree (or a single-kv view).")
    parser.add_argument("--vuln-tasks", default=str(DATA_DIR / "tasks_vulnerable.jsonl"))
    parser.add_argument("--patched-tasks", default=str(DATA_DIR / "tasks_patched.jsonl"))
    parser.add_argument("--out-dir", default=None,
                        help="Where to write primevul_metrics.{json,md} (default: <vuln-root>/report).")
    args = parser.parse_args()

    vuln_root, patched_root = Path(args.vuln_root), Path(args.patched_root)
    vuln_cases, vuln_info = load_arm(vuln_root, Path(args.vuln_tasks))
    patched_cases, patched_info = load_arm(patched_root, Path(args.patched_tasks))
    if not vuln_cases:
        raise SystemExit(f"no scorable task folders under {vuln_root}")

    result = analyse(vuln_cases, patched_cases)
    result["sources"] = {
        "vulnerable_root": str(vuln_root),
        "patched_root": str(patched_root),
        "pairs_table": str(DATA_DIR / "primevul_pairs.jsonl"),
    }
    result["loading"] = {"vulnerable": vuln_info, "patched": patched_info}
    # Sanity: the two arms must cover the same samples, or the pairing is broken.
    result["coverage"] = {
        "vulnerable_samples": len({c["sample_id"] for c in vuln_cases}),
        "patched_samples": len({c["sample_id"] for c in patched_cases}),
        "paired_samples": len({c["sample_id"] for c in vuln_cases}
                              & {c["sample_id"] for c in patched_cases}),
    }

    out_dir = Path(args.out_dir) if args.out_dir else (vuln_root / "report")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primevul_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "primevul_metrics.md").write_text(
        render_markdown(result, result["sources"]), encoding="utf-8")

    overall = result["overall"]["vulnerable"]
    print(f"[primevul-metrics] {out_dir / 'primevul_metrics.md'}")
    print(f"[primevul-metrics] dense recall {_fmt(overall['dense_recall'])}, "
          f"S_GT deployed {_fmt(overall['gt_silencing_deployed'])}, "
          f"S_GT gate-hit {_fmt(overall['gt_silencing_gate_hit'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
