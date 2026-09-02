"""Reporting helpers for the secure-code experiment.

These functions interpret the existing `Latency.json` (written by
`LLMChat.agen_kvcomm` via `_append_latency_record`) — they do not perform any
KV computation themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Single source of truth for cold/warm semantics.
_MODE_FLAGS = {
    "dense_prefill": {"anchor_hit": False, "fallback_used": True},
    "kv_reuse": {"anchor_hit": True, "fallback_used": False},
}


def derive_flags(mode: str) -> Dict[str, bool]:
    """Map a generation mode to (anchor_hit, fallback_used)."""
    if mode not in _MODE_FLAGS:
        raise ValueError(
            f"Unexpected generation mode '{mode}'. Expected one of {sorted(_MODE_FLAGS)}."
        )
    return dict(_MODE_FLAGS[mode])


def load_latency(path: str) -> List[Dict[str, Any]]:
    """Load the Latency.json list written during an allow_kv_reuse run."""
    latency_path = Path(path)
    if not latency_path.exists():
        raise FileNotFoundError(f"Latency file not found: {latency_path}")
    with open(latency_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Latency.json must contain a list of records.")
    return data


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_measurement(rec: Dict[str, Any]) -> bool:
    """True for on-hit dense baseline runs (measure_only), which are NOT natural runs."""
    return bool(rec.get("measure_only"))


def natural_hit_summary(
    records: List[Dict[str, Any]],
    *,
    agent_role: str,
) -> Dict[str, Any]:
    """Honest anchor-hit rate over the single natural pass for one agent role.

    Only *natural* generations count (each task ran once). The measurement-only
    dense baselines added after a hit are excluded, so the hit rate reflects real
    anchor behavior rather than being inflated by any pre-seeding.
    """
    natural = [
        r for r in records
        if r.get("agent_role") == agent_role and not _is_measurement(r)
    ]
    hits = [r for r in natural if r.get("mode") == "kv_reuse"]
    misses = [r for r in natural if r.get("mode") == "dense_prefill"]
    total = len(natural)
    return {
        "agent_role": agent_role,
        "natural_total": total,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": (len(hits) / total) if total else None,
    }


def pair_reuse_with_baseline(
    records: List[Dict[str, Any]],
    *,
    agent_role: str,
) -> List[Dict[str, Any]]:
    """Pair each task's natural kv_reuse with its measurement-only dense baseline.

    One row per task (ordered by first appearance):
    - Natural **hit** (kv_reuse): ``reuse_ttft`` is the natural kv_reuse TTFT and
      ``baseline_ttft`` is the measurement-only dense_prefill run of the SAME task;
      ``speedup`` = baseline / reuse. If the baseline is missing, speedup is None.
    - Natural **miss** (dense_prefill): there is no reuse to accelerate, so
      ``reuse_ttft``/``speedup`` are None and ``baseline_ttft`` is the task's own
      natural dense TTFT (it is already its baseline).

    End-to-end timing is paired the same way from ``generation_total_latency``:
    ``reuse_total``, ``baseline_total``, and ``total_speedup`` = baseline_total /
    reuse_total (None where the record lacks a total).
    """
    order: List[str] = []
    natural_mode: Dict[str, str] = {}
    reuse_ttft: Dict[str, float] = {}
    natural_dense_ttft: Dict[str, float] = {}
    measured_baseline_ttft: Dict[str, float] = {}
    reuse_total: Dict[str, float] = {}
    natural_dense_total: Dict[str, float] = {}
    measured_baseline_total: Dict[str, float] = {}

    for rec in records:
        if rec.get("agent_role") != agent_role:
            continue
        message = rec.get("message")
        mode = rec.get("mode")
        ttft = float(rec.get("ttft", 0.0))
        total_raw = rec.get("generation_total_latency")
        total = float(total_raw) if total_raw is not None else None
        if message not in order:
            order.append(message)
        if _is_measurement(rec):
            if mode == "dense_prefill":
                measured_baseline_ttft[message] = ttft
                if total is not None:
                    measured_baseline_total[message] = total
            continue
        # natural record
        natural_mode.setdefault(message, mode)
        if mode == "kv_reuse":
            reuse_ttft[message] = ttft
            if total is not None:
                reuse_total[message] = total
        elif mode == "dense_prefill":
            natural_dense_ttft[message] = ttft
            if total is not None:
                natural_dense_total[message] = total

    rows: List[Dict[str, Any]] = []
    for message in order:
        mode = natural_mode.get(message)
        if mode == "kv_reuse":
            reuse = reuse_ttft.get(message)
            baseline = measured_baseline_ttft.get(message)
            speedup = (baseline / reuse) if (baseline is not None and reuse) else None
            r_total = reuse_total.get(message)
            b_total = measured_baseline_total.get(message)
            total_speedup = (b_total / r_total) if (b_total is not None and r_total) else None
            rows.append({
                "message": message,
                "natural_mode": "kv_reuse",
                "reuse_ttft": reuse,
                "baseline_ttft": baseline,
                "speedup": speedup,
                "reuse_total": r_total,
                "baseline_total": b_total,
                "total_speedup": total_speedup,
            })
        elif mode == "dense_prefill":
            rows.append({
                "message": message,
                "natural_mode": "dense_prefill",
                "reuse_ttft": None,
                "baseline_ttft": natural_dense_ttft.get(message),
                "speedup": None,
                "reuse_total": None,
                "baseline_total": natural_dense_total.get(message),
                "total_speedup": None,
            })
    return rows


def summarize_agent_records(
    records: List[Dict[str, Any]],
    *,
    agent_role: str,
) -> Dict[str, Any]:
    """Cold (dense_prefill) vs warm (kv_reuse) timing summary for one agent role.

    Reports averages of total TTFT, the dense-prefill / generation TTFT
    (``generation_ttft``), full generation time (``generation_total_latency``),
    and the kv_reuse preprocess latency, so cold and warm are comparable.
    """
    agent = [r for r in records if r.get("agent_role") == agent_role]
    buckets: Dict[str, Dict[str, List]] = {
        "dense_prefill": {"ttft": [], "gen_ttft": [], "total": [], "preprocess": [],
                          "anchor_making": [], "input_tokens": [], "output_tokens": []},
        "kv_reuse": {"ttft": [], "gen_ttft": [], "total": [], "preprocess": [],
                     "input_tokens": [], "output_tokens": [],
                     "anchor_hit": [], "offset_fallback": []},
    }
    per_generation: List[Dict[str, Any]] = []
    for rec in agent:
        mode = rec.get("mode")
        flags = derive_flags(mode)
        ttft = float(rec.get("ttft", 0.0))
        gen_ttft = float(rec.get("generation_ttft", 0.0))
        total = float(rec.get("generation_total_latency", 0.0))
        preprocess = float(rec.get("preprocess_latency") or 0.0)
        in_tok = int(rec.get("input_tokens") or 0)
        out_tok = int(rec.get("output_tokens") or 0)
        b = buckets[mode]
        b["ttft"].append(ttft)
        b["gen_ttft"].append(gen_ttft)
        b["total"].append(total)
        b["preprocess"].append(preprocess)
        b["input_tokens"].append(in_tok)
        b["output_tokens"].append(out_tok)
        row: Dict[str, Any] = {
            "message": rec.get("message"),
            "mode": mode,
            "ttft": ttft,
            "generation_ttft": gen_ttft,
            "generation_total_latency": total,
            "preprocess_latency": preprocess,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            **flags,
        }
        if mode == "dense_prefill":
            aml = rec.get("anchor_making_latency")
            if aml is not None:
                b["anchor_making"].append(float(aml))
                row["anchor_making_latency"] = float(aml)
        elif mode == "kv_reuse":
            ah = rec.get("anchor_hit")
            of = rec.get("offset_fallback")
            if ah is not None:
                b["anchor_hit"].append(bool(ah))
            if of is not None:
                b["offset_fallback"].append(bool(of))
            row["anchor_hit"] = ah
            row["offset_fallback"] = of
        per_generation.append(row)

    cold, warm = buckets["dense_prefill"], buckets["kv_reuse"]
    cold_avg_ttft = _avg(cold["ttft"])
    warm_avg_ttft = _avg(warm["ttft"])
    cold_avg_total = _avg(cold["total"])
    warm_avg_total = _avg(warm["total"])
    warm_anchor_hits = sum(warm["anchor_hit"])
    warm_anchor_total = len(warm["anchor_hit"])
    return {
        "agent_role": agent_role,
        "cold_count": len(cold["ttft"]),
        "warm_count": len(warm["ttft"]),
        "cold_avg_ttft": cold_avg_ttft,
        "warm_avg_ttft": warm_avg_ttft,
        "cold_avg_dense_prefill_ttft": _avg(cold["gen_ttft"]),
        "warm_avg_gen_ttft": _avg(warm["gen_ttft"]),
        "cold_avg_total_generation": cold_avg_total,
        "warm_avg_total_generation": warm_avg_total,
        "warm_avg_preprocess": _avg(warm["preprocess"]),
        "cold_avg_anchor_making_latency": _avg(cold["anchor_making"]),
        "cold_avg_input_tokens": _avg(cold["input_tokens"]),
        "cold_avg_output_tokens": _avg(cold["output_tokens"]),
        "warm_avg_input_tokens": _avg(warm["input_tokens"]),
        "warm_avg_output_tokens": _avg(warm["output_tokens"]),
        "warm_anchor_hit_count": warm_anchor_hits,
        "warm_anchor_total": warm_anchor_total,
        "warm_anchor_hit_rate": (warm_anchor_hits / warm_anchor_total) if warm_anchor_total else None,
        "warm_offset_fallback_count": sum(warm["offset_fallback"]),
        "ttft_speedup": (cold_avg_ttft / warm_avg_ttft) if warm_avg_ttft > 0 else None,
        "total_generation_speedup": (cold_avg_total / warm_avg_total) if warm_avg_total > 0 else None,
        "per_generation": per_generation,
    }


def summarize_all_agents(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-agent-role timing summaries, keyed by role (order preserved)."""
    roles: List[str] = []
    for rec in records:
        role = rec.get("agent_role")
        if role is not None and role not in roles:
            roles.append(role)
    return {role: summarize_agent_records(records, agent_role=role) for role in roles}


def summarize_validator_records(
    records: List[Dict[str, Any]],
    *,
    validator_role: str = "Security Validator",
) -> Dict[str, Any]:
    """Summarize the validator's cold (dense_prefill) vs warm (kv_reuse) TTFT."""
    validator = [r for r in records if r.get("agent_role") == validator_role]
    per_generation: List[Dict[str, Any]] = []
    cold_ttfts: List[float] = []
    warm_ttfts: List[float] = []
    for rec in validator:
        mode = rec.get("mode")
        flags = derive_flags(mode)
        ttft = float(rec.get("ttft", 0.0))
        per_generation.append({
            "message": rec.get("message"),
            "mode": mode,
            "ttft": ttft,
            **flags,
        })
        if mode == "dense_prefill":
            cold_ttfts.append(ttft)
        elif mode == "kv_reuse":
            warm_ttfts.append(ttft)

    cold_avg = sum(cold_ttfts) / len(cold_ttfts) if cold_ttfts else 0.0
    warm_avg = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else 0.0
    speedup = (cold_avg / warm_avg) if warm_avg > 0 else None
    return {
        "validator_role": validator_role,
        "cold_count": len(cold_ttfts),
        "warm_count": len(warm_ttfts),
        "cold_avg_ttft": cold_avg,
        "warm_avg_ttft": warm_avg,
        "speedup": speedup,
        "per_generation": per_generation,
    }
