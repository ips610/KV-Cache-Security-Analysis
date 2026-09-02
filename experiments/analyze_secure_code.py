import argparse
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()
import json
from pathlib import Path

from experiments.secure_code_report import (
    load_latency,
    natural_hit_summary,
    pair_reuse_with_baseline,
    summarize_all_agents,
    summarize_validator_records,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze secure-code KVCOMM run")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Run output dir containing Latency.json.")
    parser.add_argument("--validator_role", type=str, default="Security Validator")
    return parser.parse_args()


def _fmt_secs(value):
    return f"{value:.4f}s" if value is not None else "   n/a  "


def _fmt_speedup(value):
    return f"{value:.2f}x" if value is not None else "  -  "


def print_honest_report(records, role):
    """Real anchor-hit rate over the single natural pass, plus each hit's
    kv_reuse next to its measurement-only dense baseline."""
    hit = natural_hit_summary(records, agent_role=role)
    rate = hit["hit_rate"]
    print("=" * 78)
    print(f"Natural single-pass anchor hits for {role}:")
    print(f"  hits={hit['hits']}/{hit['natural_total']}"
          + (f" ({rate:.0%})" if rate is not None else "")
          + f"   misses={hit['misses']}  (measurement-only baselines excluded)")
    rows = pair_reuse_with_baseline(records, agent_role=role)
    if not rows:
        return
    print("-" * 78)
    print(f"  {'task':40s} {'natural':9s} {'reuse':>9s} {'baseline':>10s} {'speedup':>9s}")
    for r in rows:
        msg = (r.get("message") or "")[:38]
        tag = "HIT" if r["natural_mode"] == "kv_reuse" else "miss"
        print(f"  {msg:40s} {tag:9s} {_fmt_secs(r['reuse_ttft']):>9s} "
              f"{_fmt_secs(r['baseline_ttft']):>10s} {_fmt_speedup(r['speedup']):>9s}")
    print("  (miss = no anchor hit, ran dense naturally; it is already its own baseline)")


def main():
    args = parse_args()
    latency_path = Path(args.output_dir) / "Latency.json"
    records = load_latency(str(latency_path))
    summary = summarize_validator_records(records, validator_role=args.validator_role)

    print_honest_report(records, args.validator_role)

    print("=" * 70)
    print(f"Validator role : {summary['validator_role']}")
    print(f"Cold (dense_prefill) generations : {summary['cold_count']}  avg TTFT={summary['cold_avg_ttft']:.4f}s")
    print(f"Warm (kv_reuse)     generations : {summary['warm_count']}  avg TTFT={summary['warm_avg_ttft']:.4f}s")
    speedup = summary["speedup"]
    print(f"Speedup (cold/warm) : {speedup:.2f}x" if speedup else "Speedup : n/a (no warm runs)")
    print("-" * 70)
    print(f"{'message':40s} {'mode':14s} {'anchor_hit':10s} {'fallback':9s} ttft")
    for row in summary["per_generation"]:
        msg = (row.get("message") or "")[:38]
        print(f"{msg:40s} {row['mode']:14s} {str(row['anchor_hit']):10s} "
              f"{str(row['fallback_used']):9s} {row['ttft']:.4f}")

    # Per-agent timing: complete generation time + dense-prefill, both agents.
    per_agent = summarize_all_agents(records)
    print("=" * 70)
    print("Per-agent timing (cold = dense_prefill, warm = kv_reuse):")
    for role, s in per_agent.items():
        print("-" * 70)
        print(f"Agent role : {role}  (cold={s['cold_count']}, warm={s['warm_count']})")
        print(f"  TTFT              cold={s['cold_avg_ttft']:.4f}s  warm={s['warm_avg_ttft']:.4f}s"
              + (f"  speedup={s['ttft_speedup']:.2f}x" if s['ttft_speedup'] else "  speedup=n/a"))
        print(f"  Dense-prefill TTFT cold={s['cold_avg_dense_prefill_ttft']:.4f}s  "
              f"warm(gen_ttft)={s['warm_avg_gen_ttft']:.4f}s")
        print(f"  Total generation  cold={s['cold_avg_total_generation']:.4f}s  "
              f"warm={s['warm_avg_total_generation']:.4f}s"
              + (f"  speedup={s['total_generation_speedup']:.2f}x"
                 if s['total_generation_speedup'] else "  speedup=n/a"))
        print(f"  Warm preprocess   {s['warm_avg_preprocess']:.4f}s")
        aml = s.get("cold_avg_anchor_making_latency", 0.0)
        print(f"  Anchor making     cold={aml:.4f}s (cost of materialising anchors)")
        ci = s.get("cold_avg_input_tokens", 0)
        co = s.get("cold_avg_output_tokens", 0)
        wi = s.get("warm_avg_input_tokens", 0)
        wo = s.get("warm_avg_output_tokens", 0)
        print(f"  Token stats       cold: input={ci:.0f}  output={co:.0f} | warm: input={wi:.0f}  output={wo:.0f}")
        ah = s.get("warm_anchor_hit_count", 0)
        at = s.get("warm_anchor_total", 0)
        rate = s.get("warm_anchor_hit_rate")
        fb = s.get("warm_offset_fallback_count", 0)
        print(f"  Anchor stats      hit={ah}/{at}"
              + (f" ({rate:.0%})" if rate is not None else "")
              + f"  offset_fallback={fb}")
    print("=" * 70)
    print(json.dumps({"validator_summary": summary, "per_agent": per_agent},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
