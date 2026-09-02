"""Put the KVCOMM and CacheBlend arms side by side against the same question.

Does the vulnerability-silencing effect survive a *different* lossy KV-reuse
mechanism? KVCOMM approximates by adding a cross-task learned delta to cached
KV; CacheBlend prefills chunks independently and selectively recomputes the
highest-deviation tokens. If both silence, the effect is a property of lossy
reuse. If only KVCOMM does, it is a property of that approximation.

Four numbers are needed to answer this honestly, and this script reports all of
them together because quoting any one alone is misleading:

1. **Each arm vs its own dense reference.** The headline. Within-system paired,
   so engine and tokenizer differences cancel.
2. **Each arm's own noise floor.** dense vs dense. KVCOMM's HF runs measured
   0.0%; vLLM in bf16 can flip greedy tokens between arithmetically identical
   computations, so CacheBlend's floor must be measured, not assumed. A
   silencing rate that does not clear its own floor is not an effect.
3. **Coverage-matched rows, on both axes.** KVCOMM only produces a paired case
   on an anchor hit (41-70 of 90 depending on the gate); CacheBlend covers every
   task. The arms also run different model sets -- the CacheBlend sweep excludes
   Llama-3.2-3B deliberately (see cacheblend/run_sweep.sh). Differencing rows
   built on different (model, task) sets confounds the mechanism with the
   selection, so each arm is *also* reported restricted to the (model, task)
   pairs both arms cover. Only those two rows may be compared to each other.
4. **The engine floor.** KVCOMM's dense reference (HF transformers, eager) and
   CacheBlend's dense reference (vLLM 0.4.1, paged xformers) are *different
   references*. Their disagreement bounds how much of any cross-system gap is
   the serving stack rather than the algorithm. Never compare the two arms
   without quoting it.

Usage::

    python experiments/compare_arms_cross_system.py \\
        --kvcomm-root results/MarginThresholdRun --kvcomm-kv kv_t0.3_a20_w5 \\
        --cacheblend-root results/CacheBlendRun --cacheblend-kv cb_r0.16_sufB
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

from experiments.paper_metrics import compare_arms, load_cases  # noqa: E402


def _by_kv(cases: List[Dict], kv_slug: Optional[str]) -> List[Dict]:
    """Keep one kv point. Pooling them repeats the collapse make_kv_view fixes."""
    if kv_slug is None:
        return cases
    return [c for c in cases if c["kv"] == kv_slug]


def _index(cases: List[Dict]) -> Dict[tuple, Dict]:
    """Map (model, task) -> case, keeping the first repetition."""
    out: Dict[tuple, Dict] = {}
    for case in cases:
        out.setdefault((case["model"], case["task"]), case)
    return out


def _by_model(cases: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for case in cases:
        grouped[case["model"]].append(case)
    return dict(grouped)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _row(label: str, stats: Dict[str, Any]) -> str:
    return (
        f"| {label} | {stats['n_cases']} | {_fmt(stats['report_change_rate_pct'])} | "
        f"{_fmt(stats['applicable_agreement_pct'])} | {stats['dense_fail_total']} | "
        f"{stats['fail_abandoned']} | {stats['fail_created']} | "
        f"{_fmt(stats['silencing_rate_pct'])} |"
    )


_HEADER = (
    "| arm | n | reports changed | applicable agreement | dense FAIL | "
    "abandoned | created | silencing |\n|---|---|---|---|---|---|---|---|"
)


def _coverage_note(coverage: Dict[str, Any]) -> str:
    """Spell out which arm ran which models, so no row is read as like-for-like."""
    only_kv = coverage.get("kvcomm_only_models") or []
    only_cb = coverage.get("cacheblend_only_models") or []
    if not only_kv and not only_cb:
        return (
            "Both arms cover the same models, so the two matched rows differ "
            "only in which tasks KVCOMM's anchor gate admitted."
        )
    parts = []
    if only_kv:
        parts.append(f"KVCOMM only: {', '.join(only_kv)}")
    if only_cb:
        parts.append(f"CacheBlend only: {', '.join(only_cb)}")
    return (
        "**The arms do not cover the same models** — "
        + "; ".join(parts)
        + ". The all-models KVCOMM row is therefore scored over models the other "
        "arm never ran, and must not be differenced against any CacheBlend row. "
        "Compare *shared set* against *hit set*: same models, same tasks, each "
        "against its own dense reference. Per-model tables below cover the "
        "shared models only."
    )


def analyse(
    kvcomm_cases: List[Dict], cb_cases: List[Dict]
) -> Dict[str, Any]:
    kv_index = _index(kvcomm_cases)
    cb_index = _index(cb_cases)

    # KVCOMM only forms a paired case on an anchor hit, so its key set IS the
    # hit set for this gate setting.
    hit_keys = set(kv_index)
    shared_keys = hit_keys & set(cb_index)
    matched_cb = [c for k, c in cb_index.items() if k in shared_keys]
    # The mirror of matched_cb. The arms do not cover the same models -- the
    # CacheBlend sweep excludes Llama-3.2-3B on purpose (its corpus does not
    # support a replayed comparison, see cacheblend/run_sweep.sh) -- so the
    # full-coverage KVCOMM row is scored over models CacheBlend never ran. Only
    # this row is comparable to the CacheBlend rows below it.
    matched_kv = [c for k, c in kv_index.items() if k in shared_keys]

    # Engine floor: the two systems' DENSE references on the same task. This is
    # not a noise floor; it bounds the serving-stack contribution.
    engine_cases = [
        {
            "model": key[0],
            "task": key[1],
            "kvcomm_dense": kv_index[key]["dense"],
            "cb_dense": cb_index[key]["dense"],
        }
        for key in sorted(shared_keys)
    ]

    result: Dict[str, Any] = {
        "kvcomm": compare_arms(kvcomm_cases, "dense", "reuse"),
        "kvcomm_matched": compare_arms(matched_kv, "dense", "reuse"),
        "cacheblend_full": compare_arms(cb_cases, "dense", "blend_alias"),
        "cacheblend_matched": compare_arms(matched_cb, "dense", "blend_alias"),
        "engine_floor": compare_arms(engine_cases, "kvcomm_dense", "cb_dense"),
        "coverage": {
            "kvcomm_cases": len(kvcomm_cases),
            "cacheblend_cases": len(cb_cases),
            "kvcomm_hit_tasks": len(hit_keys),
            "coverage_matched_cases": len(matched_cb),
            "engine_floor_cases": len(engine_cases),
            "kvcomm_only_models": sorted(
                set(_by_model(kvcomm_cases)) - set(_by_model(cb_cases))
            ),
            "cacheblend_only_models": sorted(
                set(_by_model(cb_cases)) - set(_by_model(kvcomm_cases))
            ),
        },
    }

    # Noise floors, where a second dense report exists.
    for name, cases in (("kvcomm", kvcomm_cases), ("cacheblend", cb_cases)):
        with_b = [c for c in cases if c.get("dense_b")]
        result[f"{name}_noise_floor"] = (
            compare_arms(with_b, "dense", "dense_b") if with_b else None
        )
    return result


def render(result: Dict[str, Any], per_model: Dict[str, Dict[str, Any]],
           kv_slug: str, cb_slug: str) -> str:
    lines: List[str] = [
        "# KVCOMM vs CacheBlend — does lossy KV reuse silence findings?",
        "",
        f"KVCOMM cell `{kv_slug}` against CacheBlend cell `{cb_slug}`.",
        "",
        "Each arm is scored against **its own** dense reference by identical "
        "code, so no difference here comes from a different analysis path.",
        "",
        "## Overall",
        "",
        _HEADER,
        _row("KVCOMM, all models", result["kvcomm"]),
        _row("KVCOMM, shared set", result["kvcomm_matched"]),
        _row("CacheBlend, all tasks", result["cacheblend_full"]),
        _row("CacheBlend, KVCOMM's hit set", result["cacheblend_matched"]),
        "",
        _coverage_note(result["coverage"]),
        "",
        "### Floors — read these before the rows above",
        "",
        _HEADER,
    ]
    for name, label in (
        ("kvcomm_noise_floor", "KVCOMM dense vs dense"),
        ("cacheblend_noise_floor", "CacheBlend dense vs dense"),
        ("engine_floor", "HF dense vs vLLM dense (engine floor)"),
    ):
        stats = result.get(name)
        lines.append(
            _row(label, stats) if stats
            else f"| {label} | — | not measured | | | | | |"
        )

    lines += [
        "",
        "A silencing rate that does not clear its own arm's dense-vs-dense floor "
        "is not evidence of anything. The engine floor bounds how much of any "
        "KVCOMM-vs-CacheBlend difference is HF-versus-vLLM rather than the "
        "reuse mechanism; it is **not** a noise floor and the two arms must "
        "never be compared through each other's dense reference.",
        "",
        "## Per model",
        "",
    ]
    for model, res in sorted(per_model.items()):
        lines += [
            f"### {model}",
            "",
            _HEADER,
            _row("KVCOMM", res["kvcomm"]),
            _row("CacheBlend, all tasks", res["cacheblend_full"]),
            _row("CacheBlend, hit set", res["cacheblend_matched"]),
            _row("engine floor", res["engine_floor"]),
        ]
        floor = res.get("cacheblend_noise_floor")
        lines.append(
            _row("CacheBlend dense vs dense", floor) if floor
            else "| CacheBlend dense vs dense | — | not measured | | | | | |"
        )
        lines += [
            "",
            f"Coverage: KVCOMM {res['coverage']['kvcomm_cases']} paired cases "
            f"(anchor hits), CacheBlend {res['coverage']['cacheblend_cases']} "
            f"(every task), matched {res['coverage']['coverage_matched_cases']}.",
            "",
        ]

    lines += [
        "## Reading this",
        "",
        "- **Both arms silence well above their floors** — the effect is a "
        "property of lossy KV reuse, not of KVCOMM's offset approximation.",
        "- **Only KVCOMM silences** — the effect depends on the approximation "
        "strategy. CacheBlend's selective exact recompute avoids it, which is a "
        "concrete mitigation and a stronger result than a null.",
        "- **Neither clears its floor** — the harness cannot resolve the "
        "question at this precision; fix determinism or sample and vote before "
        "claiming anything.",
        "",
        "Note the mechanisms differ in more than their loss: KVCOMM reuses the "
        "generator's own KV plus a delta learned from *other tasks*, while "
        "CacheBlend prefills each chunk standalone with no cross-task "
        "information. A null result for CacheBlend therefore exonerates this "
        "reuse strategy, not lossy reuse in general.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvcomm-root", type=Path, required=True)
    parser.add_argument("--kvcomm-kv", default=None,
                        help="kv slug, e.g. kv_t0.3_a20_w5 (required if the tree has several)")
    parser.add_argument("--cacheblend-root", type=Path, required=True)
    parser.add_argument("--cacheblend-kv", default=None,
                        help="cell slug, e.g. cb_r0.16_sufB")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    kvcomm_cases = _by_kv(load_cases(args.kvcomm_root), args.kvcomm_kv)
    cb_cases = _by_kv(load_cases(args.cacheblend_root), args.cacheblend_kv)
    if not kvcomm_cases:
        raise SystemExit(f"no KVCOMM cases in {args.kvcomm_root} for kv={args.kvcomm_kv}")
    if not cb_cases:
        raise SystemExit(f"no CacheBlend cases in {args.cacheblend_root} for kv={args.cacheblend_kv}")

    for kv_slugs, label in (
        ({c["kv"] for c in kvcomm_cases}, "KVCOMM"),
        ({c["kv"] for c in cb_cases}, "CacheBlend"),
    ):
        if len(kv_slugs) > 1:
            raise SystemExit(
                f"{label} tree spans several cells {sorted(kv_slugs)}. Pass the "
                f"matching --*-kv; pooling them would fold distinct "
                f"configurations together as if they were repetitions."
            )

    # load_cases names the reuse arm "reuse"; alias it so compare_arms can be
    # called identically for both systems.
    for case in cb_cases:
        case["blend_alias"] = case["reuse"]

    overall = analyse(kvcomm_cases, cb_cases)
    per_model: Dict[str, Dict[str, Any]] = {}
    kv_models, cb_models = _by_model(kvcomm_cases), _by_model(cb_cases)
    for model in sorted(set(kv_models) & set(cb_models)):
        per_model[model] = analyse(kv_models[model], cb_models[model])

    kv_slug = args.kvcomm_kv or kvcomm_cases[0]["kv"]
    cb_slug = args.cacheblend_kv or cb_cases[0]["kv"]
    markdown = render(overall, per_model, kv_slug, cb_slug)

    out_dir = args.out_dir or args.cacheblend_root / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cross_system_{kv_slug}_vs_{cb_slug}"
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"overall": overall, "per_model": per_model},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(markdown)
    print(f"[cross-system] wrote {out_dir / f'{stem}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
