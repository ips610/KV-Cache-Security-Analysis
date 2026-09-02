import argparse
import asyncio
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap(require_kvcomm=True)
sys.stdout.reconfigure(encoding='utf-8')
import copy
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# cuBLAS reads this when it creates its workspace (first cuBLAS call), so it
# MUST be set before torch initializes CUDA. Without it, cuBLAS may pick
# non-deterministic reduction splits and identical inputs can yield different
# logits run to run — observed as ~54% of Llama-3.2-3B tasks disagreeing across
# repetitions, which would otherwise be misread as KV-reuse divergence.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from KVCOMM.graph.graph import Graph
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import configure_logging, logger
from datasets.secure_code_dataset import load_tasks
from experiments.metrics_collector import MetricsCollector, write_env
from experiments.secure_code_artifacts import write_run_meta, write_task_artifacts
from experiments.secure_code_report import (
    derive_flags,
    load_latency,
    natural_hit_summary,
    pair_reuse_with_baseline,
    summarize_validator_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Decoding is already greedy (do_sample=False everywhere in gpt_chat.py), so the
# only remaining source of run-to-run variation is non-deterministic GPU kernel
# reduction order. Pinning it is what makes the dense-vs-dense noise floor
# genuinely zero, which in turn lets every observed divergence be attributed to
# KV reuse. warn_only=True: an op without a deterministic implementation warns
# instead of aborting a multi-hour run.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception as exc:  # noqa: BLE001 — determinism is best-effort, never fatal
    print(f"[determinism] could not enable deterministic algorithms: {exc!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="KVCOMM two-agent secure-code experiment")
    parser.add_argument("--mode", type=str, default="kvcomm", choices=["baseline", "kvcomm"],
                        help="kvcomm = single natural allow_kv_reuse pass (anchors build up "
                             "online; each hit also gets a measurement-only dense baseline). "
                             "baseline runs the SAME allow_kv_reuse natural pass as kvcomm — "
                             "only the pass label differs — it is NOT an independent dense "
                             "reference; deprecated, retained for compatibility only.")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--domain", type=str, default="SecureCode")
    parser.add_argument("--tasks_path", type=str, default=None, help="Override path to the tasks JSONL (default: coding_tasks.jsonl).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks.")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--prefix", type=str, default="The task is: ",
                        help="Prefix text for the input query (kept identical to dense prefill).")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "result" / "secure_code"))
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run; folder is <output_dir>/<run_name>. "
                             "If omitted, you are prompted (timestamp fallback when non-interactive).")
    parser.add_argument("--force-exact-nist-prefill", dest="force_exact_nist_prefill",
                        type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--approximate-nist-rules", dest="approximate_nist_rules",
                        type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--kv-threshold", type=float, default=0.3)
    parser.add_argument("--kv-max-anchor-num", type=int, default=20)
    parser.add_argument("--kv-window-size", type=int, default=5)
    parser.add_argument("--metrics_out", type=str, default=None,
                        help="Path for metrics.json (default: <run_dir>/metrics.json).")
    parser.add_argument("--generator-dense", dest="generator_dense",
                        type=lambda s: s.lower() == "true", default=False,
                        help="Force the CodeGenerator to always dense-prefill (never KV reuse), "
                             "confining the optimization under study to the validator and "
                             "keeping the code under review identical across all arms.")
    parser.add_argument("--dense-reference-always", dest="dense_reference_always",
                        type=lambda s: s.lower() == "true", default=False,
                        help="Per task, run the validator dense FIRST as the reference, then "
                             "the natural pass (kv_reuse when the gate fires). Gives every "
                             "prompt a paired dense reference instead of only anchor hits.")
    parser.add_argument("--sample-roles", dest="sample_roles", type=str, default="",
                        help="Comma-separated agent roles that decode with sampling instead of "
                             "greedy, e.g. 'Security Validator'. Leave empty for fully greedy "
                             "(seeds are then inert and runs are byte-identical).")
    parser.add_argument("--sample-temperature", dest="sample_temperature", type=float, default=0.7,
                        help="Sampling temperature for the roles named by --sample-roles.")
    parser.add_argument("--noise-floor-baseline", dest="noise_floor_baseline",
                        type=lambda s: s.lower() == "true", default=False,
                        help="On every anchor hit, record a SECOND measurement-only dense "
                             "baseline of the same validation. Diffing the two dense reports "
                             "measures the dense-vs-dense noise floor on exactly the paired "
                             "cases fidelity is reported over. Costs one extra full dense "
                             "generation per hit.")
    args = parser.parse_args()
    if args.approximate_nist_rules:
        parser.error(
            "--approximate-nist-rules=true is not supported: security rules must be exact-prefilled. "
            "Approximating them would require inventing new KV logic, which this experiment forbids."
        )
    if not args.force_exact_nist_prefill:
        parser.error("--force-exact-nist-prefill=false is not supported in this experiment.")
    return args


def _extract_answer(answers: Any, task_text: str) -> str:
    """arun returns a dict keyed by task (allow_kv_reuse) or a list (default)."""
    if isinstance(answers, dict):
        items = answers.get(task_text, [])
    elif isinstance(answers, list):
        items = answers
    else:
        items = []
    return items[-1] if items else ""


def _sanitize_run_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name.strip()).strip("_")


def resolve_run_name(args) -> str:
    """Pick the run name: --run_name, else interactive prompt, else timestamp."""
    if args.run_name:
        cleaned = _sanitize_run_name(args.run_name)
        if cleaned:
            return cleaned
    if sys.stdin.isatty():
        cleaned = _sanitize_run_name(input("Enter a name for this run: "))
        if cleaned:
            return cleaned
    return "run_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def build_graph(args, kv_config) -> Graph:
    # Chain topology over 2 agents: node 0 (CodeGenerator) -> node 1 (SecurityValidator).
    fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(2)] for j in range(2)]
    fixed_temporal_masks = [[0 for _ in range(2)] for _ in range(2)]
    node_kwargs = [{"role": "Code Generator"}, {"role": "Security Validator"}]
    return Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=["CodeGenerator", "SecurityValidator"],
        decision_method=None,
        fixed_spatial_masks=fixed_spatial_masks,
        fixed_temporal_masks=fixed_temporal_masks,
        node_kwargs=node_kwargs,
        kv_config=kv_config,
    )


async def _run_pass(graph: Graph, tasks: List[Dict[str, str]], *, pass_name: str,
                    prefix: str, output_dir: str, max_tokens: int,
                    artifacts_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks):
        input_dict = dict(task)
        realized_graph = copy.deepcopy(graph)
        realized_graph.spatial_logits = graph.spatial_logits
        realized_graph.temporal_logits = graph.temporal_logits
        result = await realized_graph.arun(
            input_dict,
            1,
            mode="allow_kv_reuse",
            prefix=prefix,
            output_dir=output_dir,
            max_tokens=max_tokens,
        )
        report = _extract_answer(result.get("answers", {}), task["task"])
        # Node 0 is the CodeGenerator (chain: generator -> validator).
        node_ids = list(realized_graph.nodes.keys())
        gen_node = realized_graph.nodes[node_ids[0]]
        gen_text = _extract_answer(gen_node.outputs, task["task"])
        # On an anchor hit the validator also ran a measurement-only dense
        # baseline of the SAME task; persist its report for side-by-side
        # dense-vs-reuse comparison (None on misses).
        val_node = realized_graph.nodes[node_ids[-1]]
        dense_baseline_report = getattr(val_node, "dense_baseline_reports", {}).get(task["task"])
        # Second dense baseline (only when --noise-floor-baseline true); diffing
        # it against the first gives the dense-vs-dense noise floor.
        dense_baseline_report_b = getattr(val_node, "dense_baseline_reports_b", {}).get(task["task"])
        if artifacts_dir is not None:
            write_task_artifacts(
                run_dir=artifacts_dir,
                pass_name=pass_name,
                index=index,
                task_text=task["task"],
                generator_text=gen_text,
                report_text=report,
                dense_baseline_report_text=dense_baseline_report,
                dense_baseline_report_b_text=dense_baseline_report_b,
            )
        collected.append({"pass": pass_name, "task": task["task"], "report": report,
                          "dense_baseline_report": dense_baseline_report})
        logger.opt(colors=True).info("<blue>[{}]</blue> task done: {}", pass_name, task["task"][:60])
    return collected


async def main():
    args = parse_args()
    if args.mode == "baseline":
        logger.opt(colors=True).warning(
            "<yellow>[DEPRECATED]</yellow> --mode baseline is not a dense reference: it runs "
            "the SAME allow_kv_reuse natural pass as kvcomm (only the pass label differs). "
            "Prefer --mode kvcomm — the dense reference comes from natural anchor-miss "
            "records (natural_dense_mean_ttft) and the per-task measurement-only on-hit "
            "dense baselines."
        )
    run_name = resolve_run_name(args)
    output_dir = Path(args.output_dir).expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_dir / "logs/log.txt")
    logger.opt(colors=True).info("<blue>[RUN]</blue> name={} dir={}", run_name, str(output_dir))

    # Start from a fresh Latency.json so the summary reflects only this run.
    latency_file = output_dir / "Latency.json"
    if latency_file.exists():
        latency_file.unlink()

    kv_config = KVCommConfig.from_env().apply_overrides(
        threshold=args.kv_threshold,
        max_anchor_num=args.kv_max_anchor_num,
        window_size=args.kv_window_size,
    )
    tasks = load_tasks(args.tasks_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    # Sampling is opt-in per role and read from the environment inside
    # LLMChat.sampling_kwargs; set it before the graph builds any LLM.
    if args.sample_roles:
        os.environ["KVCOMM_SAMPLE_ROLES"] = args.sample_roles
        os.environ["KVCOMM_SAMPLE_TEMPERATURE"] = str(args.sample_temperature)
        logger.opt(colors=True).info(
            "<green>[SAMPLING]</green> roles=[{}] temperature={} (all other roles stay greedy)",
            args.sample_roles, args.sample_temperature,
        )

    graph = build_graph(args, kv_config)
    # Set on the base graph so every per-task deepcopy inherits it.
    node_ids = list(graph.nodes.keys())
    validator_node = graph.nodes[node_ids[-1]]
    if args.generator_dense:
        graph.nodes[node_ids[0]].force_dense = True
        logger.opt(colors=True).info(
            "<green>[GENERATOR]</green> forced dense prefill (no KV reuse) — "
            "the code under review is identical across all arms."
        )
    if args.noise_floor_baseline:
        validator_node.noise_floor_baseline = True
        logger.opt(colors=True).info(
            "<green>[NOISE-FLOOR]</green> second dense baseline ENABLED "
            "(one extra dense generation per task)."
        )
    if args.dense_reference_always:
        validator_node.dense_reference_always = True
        logger.opt(colors=True).info(
            "<green>[REFERENCE]</green> dense-first ordering ENABLED "
            "(every task gets a paired dense reference)."
        )

    metrics_path = Path(args.metrics_out) if args.metrics_out else (output_dir / "metrics.json")
    collector = MetricsCollector(metrics_path, llm_name=args.llm_name, mode=args.mode)
    # Record the decode + determinism knobs alongside the CLI args so every
    # result carries the full set of controllable settings that produced it.
    # `temperature` is captured for completeness but is INERT: generation runs
    # with do_sample=False (greedy) in KVCOMM/llm/gpt_chat.py, so HF ignores it.
    from KVCOMM.llm.llm import LLM
    write_env(output_dir / "env.json", {
        **vars(args),
        "effective_seed": SEED,
        "decode": {
            "sample_roles": args.sample_roles or None,
            "sample_temperature": args.sample_temperature if args.sample_roles else None,
            "greedy_roles": "all" if not args.sample_roles else "all except sample_roles",
            "temperature_setting": LLM.DEFAULT_TEMPERATURE,
            "temperature_note": (
                "DEFAULT_TEMPERATURE is inert for greedy roles (do_sample=False); "
                "sampled roles use sample_temperature"
            ),
            "max_tokens": args.max_tokens,
            "seed_is_inert": not bool(args.sample_roles),
        },
        "dense_reference_always": args.dense_reference_always,
        "generator_dense": args.generator_dense,
        "determinism": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": True,
        },
        "kv": {
            "threshold": args.kv_threshold,
            "max_anchor_num": args.kv_max_anchor_num,
            "window_size": args.kv_window_size,
            "anchor_blend_softmax_temperature": 1,
            "force_exact_nist_prefill": args.force_exact_nist_prefill,
            "approximate_nist_rules": args.approximate_nist_rules,
        },
        "noise_floor_baseline": args.noise_floor_baseline,
    })

    reports: List[Dict[str, Any]] = []
    if args.mode == "baseline":
        # NOTE: this runs the SAME allow_kv_reuse natural pass as kvcomm (see
        # _run_pass — mode="allow_kv_reuse" is hardcoded there); only the pass
        # label ("cold" vs "warm") differs. This is NOT an independent dense
        # reference run. Retained for compatibility; deprecated (see warning
        # above). The honest dense reference is derived from within the
        # kvcomm pass: natural anchor-miss records plus the measurement-only
        # on-hit dense baselines.
        collector.start_pass("cold")
        reports += await _run_pass(graph, tasks, pass_name="cold", prefix=args.prefix,
                                   output_dir=str(output_dir), max_tokens=args.max_tokens,
                                   artifacts_dir=output_dir)
        collector.end_pass(task_count=len(tasks))
    if args.mode == "kvcomm":
        # Single natural pass: anchors build up online; each hit also runs a
        # measurement-only dense baseline (the honest on-hit comparison).
        collector.start_pass("warm")
        reports += await _run_pass(graph, tasks, pass_name="warm", prefix=args.prefix,
                                   output_dir=str(output_dir), max_tokens=args.max_tokens,
                                   artifacts_dir=output_dir)
        collector.end_pass(task_count=len(tasks))

    # Build summary from the existing Latency.json.
    summary: Dict[str, Any] = {}
    hit_summary: Dict[str, Any] = {}
    reuse_pairs: List[Dict[str, Any]] = []
    try:
        records = load_latency(str(latency_file))
        summary = summarize_validator_records(records, validator_role="Security Validator")
        hit_summary = natural_hit_summary(records, agent_role="Security Validator")
        reuse_pairs = pair_reuse_with_baseline(records, agent_role="Security Validator")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not summarize Latency.json: {}", exc)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    payload = {
        "run_name": run_name,
        "mode": args.mode,
        "llm_name": args.llm_name,
        "force_exact_nist_prefill": args.force_exact_nist_prefill,
        "approximate_nist_rules": args.approximate_nist_rules,
        "num_tasks": len(tasks),
        "summary": summary,
        "natural_hit_summary": hit_summary,
        "reuse_vs_baseline": reuse_pairs,
        "reports": reports,
        "timestamp": timestamp,
    }
    result_file = output_dir / f"secure_code_{args.mode}_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_run_meta(output_dir, {k: payload[k] for k in (
        "run_name", "mode", "llm_name", "force_exact_nist_prefill",
        "approximate_nist_rules", "num_tasks", "summary", "timestamp",
    )})
    collector.write(extra={"num_tasks": len(tasks), "run_name": run_name})
    logger.opt(colors=True).info("<blue>[RESULT SAVED]</blue> {}", str(result_file))
    if hit_summary:
        logger.opt(colors=True).info(
            "<magenta>[HITS]</magenta> natural anchor hits={}/{} ({}) — each hit has a "
            "measurement-only dense baseline for comparison.",
            hit_summary.get("hits", 0),
            hit_summary.get("natural_total", 0),
            (f"{hit_summary['hit_rate']:.0%}" if hit_summary.get("hit_rate") is not None else "n/a"),
        )


if __name__ == "__main__":
    asyncio.run(main())
