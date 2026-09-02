"""PrimeVul ground-truth arm: dense vs KV-reuse validation of labelled code.

PrimeVul supplies the function, so Agent 1 is a promptless dense encoder: it
performs one direct forward pass over raw code tokens and returns no text. The
runner privately retains dataset metadata, gates only Agent 2, and supplies
Agent 2 either the request-scoped opaque KV package or a verified dense code
payload. Generic ``Graph.arun`` record broadcast is intentionally not used.

One arm per run (plan section 9): all vulnerable versions in one run with a
fresh anchor pool, all patched versions in another, matched offline by
``pair_id``. A vulnerable function and its patched twin can differ by a handful
of tokens, so they must never share an anchor pool.

    python experiments/run_primevul.py --mode kvcomm \
        --llm_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --tasks_path datasets/primevul/tasks_vulnerable.jsonl \
        --output_dir results/PrimeVulVulnRun/... --run_name rep_0

CLI flag names are kept identical to ``run_secure_code.py`` so
``experiments/sweep.py --runner`` drives this file unchanged.
"""
import argparse
import asyncio
import sys, os
# Repo root goes to the FRONT of sys.path, not the end: the local
# `datasets/` namespace package must win over any installed HuggingFace
# `datasets` in the environment, or `datasets.primevul_dataset` is
# shadowed and the import fails.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap(require_kvcomm=True)
sys.stdout.reconfigure(encoding='utf-8')
import json
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# cuBLAS reads this when it creates its workspace (first cuBLAS call), so it
# MUST be set before torch initializes CUDA. Without it, cuBLAS may pick
# non-deterministic reduction splits and identical inputs can yield different
# logits run to run, which would be misread as KV-reuse divergence.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from KVCOMM.graph.graph import Graph
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import configure_logging, logger
from KVCOMM.utils.metrics import GenerationResult, metrics_recorder
from datasets.primevul_dataset import DATA_DIR, load_tasks
from experiments.metrics_collector import MetricsCollector, write_env
from experiments.secure_code_artifacts import (
    task_slug,
    write_primevul_task_artifacts,
    write_run_meta,
)
from experiments.secure_code_report import (
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
except Exception as exc:  # noqa: BLE001 - determinism is best-effort, never fatal
    print(f"[determinism] could not enable deterministic algorithms: {exc!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="KVCOMM PrimeVul ground-truth experiment")
    parser.add_argument("--mode", type=str, default="kvcomm", choices=["kvcomm"],
                        help="kvcomm = single natural allow_kv_reuse pass (anchors build up "
                             "online; each hit also gets a measurement-only dense baseline).")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--domain", type=str, default="PrimeVul")
    parser.add_argument("--tasks_path", type=str, default=None,
                        help="Arm task file (default: datasets/primevul/tasks_vulnerable.jsonl).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--max_tokens", type=int, default=6000,
                        help="Validator report budget for all ten rule blocks.")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "result" / "primevul"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--arm", type=str, default=None, choices=["vulnerable", "patched"],
                        help="Recorded for provenance; inferred from the task file when omitted.")
    parser.add_argument("--force-exact-cwe-prefill", dest="force_exact_cwe_prefill",
                        type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--force-exact-nist-prefill", dest="force_exact_nist_prefill",
                        type=lambda s: s.lower() != "false", default=True,
                        help="Accepted for sweep-runner compatibility; the PrimeVul arm has no "
                             "NIST rules and this maps onto --force-exact-cwe-prefill.")
    parser.add_argument("--approximate-nist-rules", dest="approximate_nist_rules",
                        type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--kv-threshold", type=float, default=0.3)
    parser.add_argument("--kv-max-anchor-num", type=int, default=20)
    parser.add_argument("--kv-window-size", type=int, default=5)
    parser.add_argument("--metrics_out", type=str, default=None)
    parser.add_argument("--generator-dense", dest="generator_dense",
                        type=lambda s: s.lower() == "true", default=True,
                        help="Provider never uses KV reuse. Always true here: a supplied "
                             "artifact's KV must be computed exactly.")
    parser.add_argument("--dense-reference-always", dest="dense_reference_always",
                        type=lambda s: s.lower() == "true", default=False,
                        help="Per sample, run the validator dense as a reference in addition to "
                             "the natural pass. Off by default: a gate miss already executes "
                             "densely and is its own reference.")
    parser.add_argument("--sample-roles", dest="sample_roles", type=str, default="")
    parser.add_argument("--sample-temperature", dest="sample_temperature", type=float, default=0.7)
    parser.add_argument("--noise-floor-baseline", dest="noise_floor_baseline",
                        type=lambda s: s.lower() == "true", default=False,
                        help="On every anchor hit, record a SECOND measurement-only dense "
                             "baseline. Diffing the two dense reports measures the "
                             "dense-vs-dense noise floor.")
    parser.add_argument("--self-test", action="store_true",
                        help="Verify the provider donates correct KV, then exit without a run.")
    parser.add_argument("--self-test-samples", type=int, default=3)
    args = parser.parse_args()
    if args.approximate_nist_rules:
        parser.error(
            "--approximate-nist-rules=true is not supported: the CWE catalogue must be "
            "exact-prefilled. Approximating it would confound a corrupted definition with "
            "corrupted code."
        )
    if not (args.force_exact_cwe_prefill and args.force_exact_nist_prefill):
        parser.error("exact rubric prefill cannot be disabled in this experiment.")
    if not args.generator_dense:
        parser.error(
            "--generator-dense=false is not supported: the PrimeVul artifact's KV is donated "
            "by an exact forward pass so the code under review is identical across all arms."
        )
    return args


def _sanitize_run_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name.strip()).strip("_")


def resolve_run_name(args) -> str:
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
    # Chain topology over 2 agents: node 0 (CodeProvider) -> node 1 (CweValidator).
    fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(2)] for j in range(2)]
    fixed_temporal_masks = [[0 for _ in range(2)] for _ in range(2)]
    node_kwargs = [{"role": "Code Provider"}, {"role": "Security Validator"}]
    return Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=["CodeProvider", "CweValidator"],
        decision_method=None,
        fixed_spatial_masks=fixed_spatial_masks,
        fixed_temporal_masks=fixed_temporal_masks,
        node_kwargs=node_kwargs,
        kv_config=kv_config,
    )


async def _run_pass(graph: Graph, tasks: List[Dict[str, Any]], *, pass_name: str,
                    output_dir: Optional[str], max_tokens: int,
                    artifacts_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Run the corrected two-stage PrimeVul chain without record broadcast."""
    from KVCOMM.llm.gpt_chat import LLMChat

    graph.construct_spatial_connection()
    node_ids = list(graph.nodes.keys())
    provider = graph.nodes[node_ids[0]]
    validator = graph.nodes[node_ids[-1]]
    if validator not in provider.spatial_successors:
        raise RuntimeError("PrimeVul graph must retain the CodeProvider -> CweValidator edge.")

    collected: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks):
        request_uid = uuid.uuid4().hex
        metrics_recorder.start_request(
            request_uid=request_uid,
            batch_index=index,
            task=task["task"],
            execution_mode="primevul_staged_v2",
        )
        package = None
        natural = None
        dense_baseline_report = None
        dense_baseline_report_b = None
        audit: Dict[str, Any] = {}
        try:
            # Agent 1 is invoked with exactly one model-visible value: raw code.
            package = await provider.encode_code(
                task["code"], request_uid=request_uid, output_dir=output_dir
            )
            metrics_recorder.record_agent_output(
                request_uid=request_uid,
                agent_id=provider.id,
                agent_name=provider.agent_name,
                agent_role=provider.role,
                generation=GenerationResult(
                    text="",
                    mode="dense_code_encode",
                    ttft=0.0,
                    metadata=provider.last_package_metadata,
                ),
            )

            # The opaque hash is an audit/anchor key. No dataset task metadata
            # enters Agent 2's user prompt or KV placeholder resolution.
            audit_key = package.code_sha256
            await validator.ensure_primevul_prefix(source_agent_id=provider.id)
            preferred_mode = validator.llm.select_bare_code_mode(
                request_uid=request_uid,
                source_agent_id=provider.id,
                audit_key=audit_key,
            )
            agent_2_payload = (
                {"opaque_package": package}
                if preferred_mode == "kv_reuse"
                else {"code": task["code"]}
            )
            natural = await validator.run_primevul_stage(
                request_uid=request_uid,
                source_agent_id=provider.id,
                audit_key=audit_key,
                preferred_mode=preferred_mode,
                payload=agent_2_payload,
                max_tokens=max_tokens,
                output_dir=output_dir,
            )
            metrics_recorder.record_agent_output(
                request_uid=request_uid,
                agent_id=validator.id,
                agent_name=validator.agent_name,
                agent_role=validator.role,
                generation=natural,
            )

            if validator.dense_reference_always or preferred_mode == "kv_reuse":
                reference = await validator.run_primevul_stage(
                    request_uid=request_uid,
                    source_agent_id=provider.id,
                    audit_key=audit_key,
                    preferred_mode="dense_prefill",
                    payload={"code": task["code"]},
                    max_tokens=max_tokens,
                    output_dir=output_dir,
                    measure_only=True,
                )
                dense_baseline_report = reference.text
                if validator.noise_floor_baseline:
                    reference_b = await validator.run_primevul_stage(
                        request_uid=request_uid,
                        source_agent_id=provider.id,
                        audit_key=audit_key,
                        preferred_mode="dense_prefill",
                        payload={"code": task["code"]},
                        max_tokens=max_tokens,
                        output_dir=output_dir,
                        measure_only=True,
                    )
                    dense_baseline_report_b = reference_b.text

            audit = {
                "schema_version": 2,
                "handoff_schema": "primevul_bare_code_kv_v2",
                **package.audit_metadata(),
                "agent_1_mode": "dense_code_encode",
                "agent_2_mode": natural.mode,
                "genuine_anchor": natural.metadata.get("genuine_anchor", False),
                "fallback_status": natural.metadata.get("fallback_status", False),
            }
            if artifacts_dir is not None:
                write_primevul_task_artifacts(
                    run_dir=artifacts_dir,
                    pass_name=pass_name,
                    index=index,
                    task_text=task["task"],
                    source_code=task["code"],
                    report_text=natural.text,
                    audit_metadata=audit,
                    dense_baseline_report_text=dense_baseline_report,
                    dense_baseline_report_b_text=dense_baseline_report_b,
                )
        finally:
            # This drops the only tensor-bearing package reference owned by the
            # request state. Anchor deltas may persist; raw Agent 1 KV may not.
            LLMChat.finalize_request(request_uid)
            metrics_recorder.finalize_request(request_uid)
            package = None
            if request_uid in LLMChat._request_states:
                raise RuntimeError("PrimeVul request-scoped KV package was not erased.")

        report = natural.text if natural is not None else ""
        collected.append({
            "pass": pass_name,
            "task": task["task"],
            "sample_id": task["sample_id"],
            "pair_id": task["pair_id"],
            "cwe": task["cwe"],
            "label": task["label"],
            "report": report,
            "dense_baseline_report": dense_baseline_report,
            "audit": audit,
        })
        logger.opt(colors=True).info("<blue>[{}]</blue> {} done ({})",
                                     pass_name, task["sample_id"], task["cwe"])
    return collected


def write_sample_index(run_dir: Path, pass_name: str, tasks: List[Dict[str, Any]]) -> None:
    """Map artifact folder slug -> ground truth, so scoring never guesses."""
    index = {
        task_slug(i, task["task"]): {
            "sample_id": task["sample_id"],
            "pair_id": task["pair_id"],
            "cwe": task["cwe"],
            "cwe_rule_number": task["cwe_rule_number"],
            "arm": task["arm"],
            "label": task["label"],
        }
        for i, task in enumerate(tasks)
    }
    payload = {"schema_version": 2, "pass": pass_name, "samples": index}
    (run_dir / "primevul_index.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


async def run_self_test(args, kv_config, tasks: List[Dict[str, Any]]) -> int:
    """Compare the corrected dense pipeline with its exact exported token ids."""
    import torch as _torch

    from experiments.export_primevul_prompt_spec import (
        DEFAULT_PROVIDER_AGENT_ID, build_spec,
    )

    arm = tasks[0]["arm"]
    tasks_path = Path(args.tasks_path) if args.tasks_path else (
        DATA_DIR / f"tasks_{arm}.jsonl")
    print(f"[self-test] building the reference token sequences for {args.llm_name} ...")
    spec = build_spec(args.llm_name, args.llm_name, arm, tasks_path,
                      DEFAULT_PROVIDER_AGENT_ID, None)
    spec_by_sample = {t["sample_id"]: t for t in spec["tasks"]}

    graph = build_graph(args, kv_config)
    node_ids = list(graph.nodes.keys())
    validator = graph.nodes[node_ids[-1]]
    sample_tasks = tasks[: max(1, args.self_test_samples)]
    reports = await _run_pass(
        graph,
        sample_tasks,
        pass_name="self_test",
        output_dir=None,
        max_tokens=args.max_tokens,
    )
    reports_by_sample = {item["sample_id"]: item["report"] for item in reports}
    failures: List[str] = []

    for task in sample_tasks:
        problems: List[str] = []
        pipeline_report = reports_by_sample.get(task["sample_id"], "")
        spec_task = spec_by_sample.get(task["sample_id"])
        if spec_task is None:
            problems.append("sample missing from the reference spec")
        else:
            ids = _torch.tensor([spec_task["full_ids"]], device=validator.llm.model.device)
            with _torch.no_grad():
                out = validator.llm.model.generate(
                    input_ids=ids,
                    attention_mask=_torch.ones_like(ids),
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            reference_text = validator.llm.tokenizer.decode(
                out[0, ids.shape[-1]:], skip_special_tokens=True)
            produced, expected = str(pipeline_report), reference_text
            overlap = min(len(produced), len(expected))
            if overlap < 32:
                problems.append(f"only {overlap} characters generated; nothing to compare")
            elif produced[:overlap] != expected[:overlap]:
                first = next((i for i in range(overlap) if produced[i] != expected[i]), overlap)
                problems.append(
                    "dense pipeline report diverges from a plain pass over the same "
                    f"ids at character {first}")
                print("  --- pipeline ---")
                print(produced[max(0, first - 120):first + 200])
                print("  --- reference ---")
                print(expected[max(0, first - 120):first + 200])
            else:
                print(f"[self-test] {task['sample_id']} dense stream matches over "
                      f"{overlap} characters (pipeline {len(produced)}, "
                      f"reference {len(expected)})")

        status = "OK" if not problems else "FAIL"
        print(f"[self-test] {task['sample_id']} {status}"
              + ("" if not problems else " :: " + "; ".join(problems)))
        failures += [f"{task['sample_id']}: {p}" for p in problems]

    if failures:
        print("[self-test] FAILED:")
        for line in failures:
            print("  -", line)
        return 1
    print(f"[self-test] passed on {len(sample_tasks)} samples "
          f"(validator: {type(validator).__name__})")
    return 0


async def main():
    args = parse_args()

    # The self-test writes no artifacts, so it must not claim a run name or a
    # results directory -- it is meant to be cheap to run before a real sweep.
    if args.self_test:
        kv_config = KVCommConfig.from_env().apply_overrides(
            threshold=args.kv_threshold,
            max_anchor_num=args.kv_max_anchor_num,
            window_size=args.kv_window_size,
        )
        tasks = load_tasks(args.tasks_path)
        if args.limit is not None:
            tasks = tasks[: args.limit]
        raise SystemExit(await run_self_test(args, kv_config, tasks))

    run_name = resolve_run_name(args)
    output_dir = Path(args.output_dir).expanduser() / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix corrected PrimeVul schema v2 with an existing run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "primevul_schema.json").write_text(
        json.dumps({
            "schema_version": 2,
            "handoff_schema": "primevul_bare_code_kv_v2",
            "legacy_results_comparable": False,
            "requires_fresh_cacheblend_run": True,
        }, indent=2),
        encoding="utf-8",
    )
    configure_logging(log_path=output_dir / "logs/log.txt")
    logger.opt(colors=True).info("<blue>[RUN]</blue> name={} dir={}", run_name, str(output_dir))

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
    arms = {task["arm"] for task in tasks}
    if len(arms) != 1:
        raise ValueError(f"a run must cover exactly one arm, found {sorted(arms)}")
    arm = arms.pop()
    if args.arm and args.arm != arm:
        raise ValueError(f"--arm {args.arm} but the task file holds the {arm} arm")
    args.arm = arm
    logger.opt(colors=True).info(
        "<green>[ARM]</green> {} - {} samples, fresh anchor pool", arm, len(tasks)
    )

    if args.sample_roles:
        os.environ["KVCOMM_SAMPLE_ROLES"] = args.sample_roles
        os.environ["KVCOMM_SAMPLE_TEMPERATURE"] = str(args.sample_temperature)
        logger.opt(colors=True).info(
            "<green>[SAMPLING]</green> roles=[{}] temperature={} (all other roles stay greedy)",
            args.sample_roles, args.sample_temperature,
        )

    graph = build_graph(args, kv_config)
    node_ids = list(graph.nodes.keys())
    validator_node = graph.nodes[node_ids[-1]]
    graph.nodes[node_ids[0]].force_dense = True
    logger.opt(colors=True).info(
        "<green>[PROVIDER]</green> code KV donated by an exact forward pass (never reused) - "
        "the code under review is identical across all arms."
    )
    if args.noise_floor_baseline:
        validator_node.noise_floor_baseline = True
        logger.opt(colors=True).info(
            "<green>[NOISE-FLOOR]</green> second dense baseline ENABLED "
            "(one extra dense generation per hit)."
        )
    if args.dense_reference_always:
        validator_node.dense_reference_always = True
        logger.opt(colors=True).info(
            "<green>[REFERENCE]</green> dense reference for EVERY sample ENABLED."
        )

    metrics_path = Path(args.metrics_out) if args.metrics_out else (output_dir / "metrics.json")
    collector = MetricsCollector(metrics_path, llm_name=args.llm_name, mode=args.mode)
    from KVCOMM.llm.llm import LLM
    write_env(output_dir / "env.json", {
        **{k: v for k, v in vars(args).items()},
        "effective_seed": SEED,
        "arm": arm,
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
            "force_exact_cwe_prefill": args.force_exact_cwe_prefill,
        },
        "noise_floor_baseline": args.noise_floor_baseline,
        "dense_reference_always": args.dense_reference_always,
    })

    collector.start_pass("warm")
    reports = await _run_pass(graph, tasks, pass_name="warm",
                              output_dir=str(output_dir), max_tokens=args.max_tokens,
                              artifacts_dir=output_dir)
    collector.end_pass(task_count=len(tasks))
    write_sample_index(output_dir, "warm", tasks)

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
        "schema_version": 2,
        "handoff_schema": "primevul_bare_code_kv_v2",
        "legacy_results_comparable": False,
        "run_name": run_name,
        "mode": args.mode,
        "arm": arm,
        "llm_name": args.llm_name,
        "force_exact_cwe_prefill": args.force_exact_cwe_prefill,
        "num_tasks": len(tasks),
        "summary": summary,
        "natural_hit_summary": hit_summary,
        "reuse_vs_baseline": reuse_pairs,
        "reports": reports,
        "timestamp": timestamp,
    }
    result_file = output_dir / f"primevul_{arm}_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_run_meta(output_dir, {k: payload[k] for k in (
        "schema_version", "handoff_schema", "legacy_results_comparable",
        "run_name", "mode", "arm", "llm_name", "force_exact_cwe_prefill",
        "num_tasks", "summary", "timestamp",
    )})
    collector.write(extra={"num_tasks": len(tasks), "run_name": run_name, "arm": arm})
    logger.opt(colors=True).info("<blue>[RESULT SAVED]</blue> {}", str(result_file))
    if hit_summary:
        logger.opt(colors=True).info(
            "<magenta>[HITS]</magenta> natural anchor hits={}/{} ({}) - each hit has a "
            "measurement-only dense baseline for comparison.",
            hit_summary.get("hits", 0),
            hit_summary.get("natural_total", 0),
            (f"{hit_summary['hit_rate']:.0%}" if hit_summary.get("hit_rate") is not None else "n/a"),
        )


if __name__ == "__main__":
    asyncio.run(main())
