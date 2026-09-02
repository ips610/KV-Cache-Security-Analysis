"""Run the Security Validator through CacheBlend and emit KVCOMM-shaped results.

Per task, in this order (matching KVCOMM commit 5620f1a, which put the
approximate pass before its reference so the two arms share an ordering):

    1. collect  - prefill each chunk independently, harvest its per-layer KV
    2. blend    - one generation with check=True at the given recompute ratio
    3. dense    - the same token ids with check=False, a true full prefill
    4. dense_b  - optional second dense, giving a within-cell noise floor

Steps 2-4 receive byte-identical ``prompt_token_ids``. That is the whole design:
the headline metric is within-system paired, so nothing about tokenizer drift or
engine differences can leak into it.

No generator runs here. The code under review comes from the frozen corpus, so
both arms review the same program.

Requires the patched CacheBlend fork and the XFormers backend -- the blending
code exists only in ``vllm/attention/backends/xformers.py``, and FlashAttention
would silently bypass it and produce a "blend" identical to a full prefill.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Must be set before vllm is imported: the selector caches the backend choice.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifacts  # noqa: E402
import prompt_builder  # noqa: E402


def _require_xformers() -> None:
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend != "XFORMERS":
        raise SystemExit(
            f"VLLM_ATTENTION_BACKEND={backend!r}; blending is implemented only in "
            "the XFormers backend. Any other backend silently disables it and "
            "the 'blend' arm would be an ordinary full prefill."
        )


def build_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs: Dict[str, Any] = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # No CUDA graphs: removes a source of run-to-run variation, and the
        # blend path reshapes the sequence mid-forward anyway.
        enforce_eager=True,
        # Must stay off: prefix caching would reuse KV across the collect,
        # blend and dense passes and silently contaminate every arm.
        enable_prefix_caching=False,
        swap_space=4,
        seed=args.seed,
        dtype=args.dtype,
    )
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    return LLM(**kwargs)


def model_handles(llm) -> Tuple[Any, Dict[str, Any], List[Any]]:
    """Reach the patched model, its control dict, and its layers.

    CacheBlend drives blending by mutating a plain dict hanging off the model,
    so this private path is the supported interface. It breaks under tensor
    parallelism (there would be more than one worker) -- hence TP=1 throughout.
    """
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
    except AttributeError as exc:
        raise SystemExit(
            "could not reach the model through driver_worker; CacheBlend's "
            "blending hooks require a single-worker (TP=1) engine."
        ) from exc
    if not hasattr(model, "cache_fuse_metadata"):
        raise SystemExit(
            f"{type(model).__name__} has no cache_fuse_metadata: this model file "
            "is unpatched, so blending would do nothing. Apply "
            "cacheblend/patches/qwen2_cacheblend.patch and reinstall."
        )
    return model, model.cache_fuse_metadata, model.layers


def collect_chunk_kvs(
    llm, model, meta: Dict[str, Any], layers: List[Any], chunks: List[List[int]],
    expected_total: Optional[int] = None,
) -> List[List[Any]]:
    """Prefill each chunk alone and concatenate its per-layer K/V.

    Each chunk is prefilled with no cross-chunk attention -- that independence is
    exactly what CacheBlend then repairs by selective recompute.

    ``chunks`` must span the ENTIRE sequence, suffix included. The suffix is
    forced into the recompute set, but the kernel still indexes ``value_old``
    over the full length::

        temp_diff = torch.sum((value[:-last_len] - value_old[:-last_len])**2, ...)

    so an ``old_kvs`` that stops short of the suffix fails with a bare tensor
    shape mismatch deep in xformers. ``example/blend_musique.py`` appends the
    query+``[/INST]`` span as a final chunk for exactly this reason.

    Note this passes ``prompt_token_ids`` rather than decoded text. The upstream
    examples decode chunks back to strings, which lets the engine prepend a BOS
    (hence their ``[1:len+1]`` slicing) and makes the final prompt a re-encode of
    a decode. Feeding ids keeps the sequence exact and removes the offset.
    """
    from vllm import SamplingParams

    import torch

    total = sum(len(c) for c in chunks)
    if expected_total is not None and total != expected_total:
        raise ValueError(
            f"chunks span {total} tokens but the prompt is {expected_total}. "
            f"old_kvs must cover the whole sequence including the suffix; "
            f"short by {expected_total - total}."
        )

    sp = SamplingParams(temperature=0, max_tokens=1)
    meta["collect"] = True
    meta["check"] = False

    per_layer: List[List[Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        llm.generate(prompt_token_ids=[list(chunk)], sampling_params=sp, use_tqdm=False)
        for layer_index in range(len(layers)):
            hack_kv = layers[layer_index].self_attn.hack_kv
            if not hack_kv:
                raise RuntimeError(
                    f"layer {layer_index} produced no hack_kv during collection; "
                    "the model file is not carrying the CacheBlend patch."
                )
            k = hack_kv[0][: len(chunk)].clone()
            v = hack_kv[1][: len(chunk)].clone()
            if k.shape[0] != len(chunk):
                raise RuntimeError(
                    f"chunk {chunk_index} layer {layer_index}: captured "
                    f"{k.shape[0]} KV rows for {len(chunk)} tokens."
                )
            if chunk_index == 0:
                per_layer.append([k, v])
            else:
                per_layer[layer_index][0] = torch.cat(
                    (per_layer[layer_index][0], k), dim=0
                )
                per_layer[layer_index][1] = torch.cat(
                    (per_layer[layer_index][1], v), dim=0
                )

    meta["collect"] = False
    model.old_kvs = per_layer
    return per_layer


def generate_once(
    llm, meta: Dict[str, Any], full_ids: List[int], *, blend: bool,
    suffix_len: int, recomp_ratio: float, max_tokens: int,
) -> Dict[str, Any]:
    from vllm import SamplingParams

    meta["collect"] = False
    meta["check"] = bool(blend)
    if blend:
        meta["suffix_len"] = suffix_len
        meta["recomp_ratio"] = recomp_ratio
        meta["recomp_ratios"] = [recomp_ratio]

    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens)
    started = time.perf_counter()
    output = llm.generate(
        prompt_token_ids=[list(full_ids)], sampling_params=sp, use_tqdm=False
    )[0]
    wall = time.perf_counter() - started

    metrics = output.metrics
    ttft = float(metrics.first_token_time - metrics.first_scheduled_time)
    completion = output.outputs[0]
    return {
        "text": completion.text,
        "token_ids": list(completion.token_ids),
        "output_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "generation_ttft": ttft,
        "generation_total_latency": wall,
    }


def common_prefix_len(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run_task(
    llm, model, meta, layers, spec, task, args, request_uid: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    chunks, suffix_ids, full_ids = prompt_builder.build(spec, task, args.suffix_config)
    audit_key = task.get("audit_key", task["seg_T"])
    budget = prompt_builder.exact_fraction(chunks, suffix_ids, args.recomp_ratio)

    collect_started = time.perf_counter()
    # The suffix is collected too: it is forced exact during the blend, but
    # old_kvs is indexed over the full sequence length regardless.
    collect_chunk_kvs(
        llm, model, meta, layers, chunks + [suffix_ids], expected_total=len(full_ids)
    )
    preprocess_latency = time.perf_counter() - collect_started

    blend = generate_once(
        llm, meta, full_ids, blend=True, suffix_len=len(suffix_ids),
        recomp_ratio=args.recomp_ratio, max_tokens=args.max_tokens,
    )
    dense = generate_once(
        llm, meta, full_ids, blend=False, suffix_len=len(suffix_ids),
        recomp_ratio=args.recomp_ratio, max_tokens=args.max_tokens,
    )
    dense_b = None
    if args.noise_floor_baseline:
        dense_b = generate_once(
            llm, meta, full_ids, blend=False, suffix_len=len(suffix_ids),
            recomp_ratio=args.recomp_ratio, max_tokens=args.max_tokens,
        )

    records = [
        artifacts.latency_record(
            mode=artifacts.MODE_BLEND, message=audit_key,
            input_tokens=len(full_ids), output_tokens=blend["output_tokens"],
            generation_ttft=blend["generation_ttft"],
            generation_total_latency=blend["generation_total_latency"],
            preprocess_latency=preprocess_latency, measure_only=False,
            request_uid=request_uid,
        ),
        artifacts.latency_record(
            mode=artifacts.MODE_DENSE, message=audit_key,
            input_tokens=len(full_ids), output_tokens=dense["output_tokens"],
            generation_ttft=dense["generation_ttft"],
            generation_total_latency=dense["generation_total_latency"],
            preprocess_latency=0.0, measure_only=True, request_uid=request_uid,
        ),
    ]
    if dense_b is not None:
        records.append(
            artifacts.latency_record(
                mode=artifacts.MODE_DENSE, message=audit_key,
                input_tokens=len(full_ids), output_tokens=dense_b["output_tokens"],
                generation_ttft=dense_b["generation_ttft"],
                generation_total_latency=dense_b["generation_total_latency"],
                preprocess_latency=0.0, measure_only=True, request_uid=request_uid,
            )
        )

    debug = {
        "task_index": task["task_index"],
        "n_chunks": len(chunks),
        "chunk_lens": [len(c) for c in chunks],
        "recomp_ratio": args.recomp_ratio,
        "suffix_config": args.suffix_config,
        **budget,
        "preprocess_latency": preprocess_latency,
        "blend_finish_reason": blend["finish_reason"],
        "dense_finish_reason": dense["finish_reason"],
        "blend_ttft": blend["generation_ttft"],
        "dense_ttft": dense["generation_ttft"],
        "identical_to_dense": blend["text"] == dense["text"],
        # Carried through so downstream analysis can restrict to the clean subset
        # without re-reading the frozen corpus.
        "generator_truncated": task.get("generator_truncated"),
        "code_identical_across_kv": task.get("code_identical_across_kv"),
        "token_count_matches_run": task.get("token_count_matches_run"),
        "kvcomm_natural_mode_by_kv": task.get("kvcomm_natural_mode_by_kv"),
    }
    return (
        {"blend": blend, "dense": dense, "dense_b": dense_b, "debug": debug},
        records,
    )


def self_test(llm, model, meta, layers, spec, args) -> int:
    """Harness checks that must pass before any result is believable.

    V2 single-chunk identity - with the entire prompt as ONE chunk, old_kv is
    bit-identical to what a full prefill computes, so the blend is an
    *arithmetic* identity. It is not a bitwise one: the blend path runs a
    different xformers kernel (a subset of queries under
    LowerTriangularFromBottomRightMask) in bf16, so reductions happen in a
    different order. A single flipped logit near a tie then cascades through
    greedy decoding.

    So the meaningful signal is *where* the two diverge, not whether. Sharing a
    long prefix and then splitting means the plumbing is right and bf16 is
    responsible; diverging at token 0 means the wiring is wrong. The gate is on
    mean common-prefix length, not on string equality.

    V3 r=1.0 near-dense - every non-suffix token is recomputed at layers >= 2,
    but layers 0-1 still read chunk-local KV. That residual is what CacheBlend
    never repairs at any ratio, so this is reported, never asserted.
    """
    tasks = spec["tasks"][: args.self_test_tasks]
    key = "matched" if args.suffix_config == "B" else "native"
    suffix_len_cfg = len(spec["invariant"][f"suffix_{key}_ids"])
    print(f"[self-test] {len(tasks)} tasks, config {args.suffix_config}, "
          f"suffix={suffix_len_cfg} tok, max_tokens={args.self_test_max_tokens}\n")

    def compare(label: str, chunks, suffix_ids, full_ids, ratio):
        collect_chunk_kvs(llm, model, meta, layers, chunks,
                          expected_total=len(full_ids))
        blend = generate_once(llm, meta, full_ids, blend=True,
                              suffix_len=len(suffix_ids), recomp_ratio=ratio,
                              max_tokens=args.self_test_max_tokens)
        dense = generate_once(llm, meta, full_ids, blend=False,
                              suffix_len=len(suffix_ids), recomp_ratio=ratio,
                              max_tokens=args.self_test_max_tokens)
        shared = common_prefix_len(blend["token_ids"], dense["token_ids"])
        n = max(len(dense["token_ids"]), 1)
        exact = blend["token_ids"] == dense["token_ids"]
        print(f"  {label}: {'IDENTICAL' if exact else 'diverges'}"
              f"  common_prefix={shared}/{len(dense['token_ids'])} tokens"
              f"  ({100 * shared / n:.0f}%)")
        return exact, shared, n

    print("V2 single-chunk identity (whole prompt as ONE chunk == full prefill)")
    v2_exact, v2_shared, v2_total = 0, 0, 0
    for task in tasks:
        _, _, full_ids = prompt_builder.build(spec, task, args.suffix_config)
        exact, shared, n = compare(
            f"task {task['task_index']:>2}", [full_ids], list(range(suffix_len_cfg)),
            full_ids, args.recomp_ratio,
        )
        v2_exact += exact
        v2_shared += shared
        v2_total += n
    v2_frac = v2_shared / max(v2_total, 1)
    print(f"  -> {v2_exact}/{len(tasks)} bitwise identical, "
          f"mean common prefix {100 * v2_frac:.1f}%\n")

    print("V3 r=1.0 near-dense (real chunking, all non-suffix tokens recomputed)")
    v3_exact, v3_shared, v3_total = 0, 0, 0
    for task in tasks:
        chunks, suffix_ids, full_ids = prompt_builder.build(spec, task, args.suffix_config)
        exact, shared, n = compare(
            f"task {task['task_index']:>2}", chunks + [suffix_ids], suffix_ids,
            full_ids, 1.0,
        )
        v3_exact += exact
        v3_shared += shared
        v3_total += n
    print(f"  -> {v3_exact}/{len(tasks)} bitwise identical, "
          f"mean common prefix {100 * v3_shared / max(v3_total, 1):.1f}%\n")

    # A plumbing bug shows up as divergence from the very first token, because
    # the blended KV would be wrong everywhere. bf16 kernel differences show up
    # as a long shared prefix that eventually splits.
    if v2_frac < 0.5:
        print("FAIL: single-chunk blending diverges from full prefill almost "
              "immediately. old_kv should be bit-identical to a fresh prefill "
              "here, so this is a wiring bug, not approximation loss. Do not "
              "run the sweep until it passes.")
        return 1
    print(f"PASS: single-chunk blending tracks full prefill for "
          f"{100 * v2_frac:.1f}% of generated tokens. Residual divergence is "
          f"bf16 reduction order in a different attention kernel, not missing "
          f"KV. V3 is reported, not asserted: layers 0-1 read chunk-local KV at "
          f"any recompute ratio.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--spec", required=True, type=Path, help="*_prompt_spec.json from KVCOMM")
    p.add_argument("--out-root", type=Path, default=Path("results/CacheBlendRun"))
    p.add_argument("--recomp-ratio", type=float, default=0.16)
    p.add_argument("--suffix-config", choices=["A", "B"], default="B",
                   help="B: rubric blended (fair vs KVCOMM). A: rubric forced exact (CacheBlend-native).")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--noise-floor-baseline", action="store_true",
                   help="run a second dense pass per task for a within-cell dense-vs-dense floor")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pass-name", default="warm")
    p.add_argument("--self-test", action="store_true", help="run V2/V3 harness checks and exit")
    p.add_argument("--self-test-tasks", type=int, default=3)
    p.add_argument("--self-test-max-tokens", type=int, default=64)
    return p.parse_args()


def main() -> int:
    _require_xformers()
    args = parse_args()

    spec = prompt_builder.load_spec(args.spec)
    is_primevul = "arm" in spec
    if is_primevul and (
        spec.get("schema_version") != 2
        or spec.get("handoff_schema") != "primevul_bare_code_kv_v2"
    ):
        raise SystemExit(
            "PrimeVul CacheBlend requires a freshly exported corrected v2 prompt spec."
        )
    if spec["model"] != args.model:
        raise SystemExit(
            f"spec is for {spec['model']!r} but --model is {args.model!r}. "
            "The token ids are tokenizer-specific; refusing to cross them."
        )

    llm = build_llm(args)
    from transformers import AutoTokenizer

    llm.set_tokenizer(AutoTokenizer.from_pretrained(args.model))
    model, meta, layers = model_handles(llm)
    print(f"[cacheblend] {type(model).__name__}, {len(layers)} layers, "
          f"backend={os.environ['VLLM_ATTENTION_BACKEND']}, "
          f"check_layers={meta['check_layers']}")

    if args.self_test:
        return self_test(llm, model, meta, layers, spec, args)

    tasks = spec["tasks"][: args.limit] if args.limit else spec["tasks"]
    run_dir = artifacts.cell_dir(
        args.out_root, spec["model_slug"], args.recomp_ratio, args.suffix_config, args.rep
    )
    if is_primevul and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix corrected PrimeVul CacheBlend results with {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    slug = artifacts.cell_slug(args.recomp_ratio, args.suffix_config)

    artifacts.write_env(run_dir, {
        "cli_args": {
            "llm_name": args.model,
            "max_tokens": args.max_tokens,
            "tasks_path": str(args.spec),
            "mode": "cacheblend",
            "recomp_ratio": args.recomp_ratio,
            "suffix_config": args.suffix_config,
            "noise_floor_baseline": args.noise_floor_baseline,
            "decode": "greedy",
            "determinism": "enforce_eager, prefix caching off, one request in flight",
            "force_exact_nist_prefill": args.suffix_config == "A",
            "approximate_nist_rules": args.suffix_config == "B",
        },
        "engine": {
            "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
        "spec": {
            "tokenizer_source": spec["tokenizer_source"],
            "chat_template_sha256": spec["chat_template_sha256"],
            "generator_agent_id": spec["generator_agent_id"],
        },
    })

    started = time.perf_counter()
    n_identical = 0
    n_truncated = 0
    for position, task in enumerate(tasks):
        result, records = run_task(
            llm, model, meta, layers, spec, task, args,
            request_uid=f"cb{args.rep}-{task['task_index']:03d}",
        )
        artifacts.write_task_artifacts(
            run_dir=run_dir,
            pass_name=args.pass_name,
            task_slug=task["task_slug"],
            task_text=task["seg_T"],
            generated_code=task["seg_C"],
            blend_report=result["blend"]["text"],
            dense_report=result["dense"]["text"],
            dense_report_b=result["dense_b"]["text"] if result["dense_b"] else None,
            debug=result["debug"],
        )
        artifacts.append_latency(run_dir, records)
        n_identical += result["debug"]["identical_to_dense"]
        n_truncated += result["blend"]["finish_reason"] == "length"
        print(
            f"[{position + 1}/{len(tasks)}] {task['task_slug']}  "
            f"exact={result['debug']['exact_fraction']:.3f}  "
            f"blend_ttft={result['debug']['blend_ttft']:.3f}s  "
            f"dense_ttft={result['debug']['dense_ttft']:.3f}s  "
            f"{'same-as-dense' if result['debug']['identical_to_dense'] else 'differs'}"
        )

    duration = time.perf_counter() - started
    peak_mem = 0
    try:
        import torch

        peak_mem = int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 - reporting only
        pass

    artifacts.write_metrics(run_dir, {
        "passes": [{
            "name": args.pass_name,
            "duration_s": duration,
            "gpu_peak_mem_bytes": peak_mem,
            "tasks_per_minute": (len(tasks) / duration * 60) if duration else None,
            "energy_joules": None,
            "avg_power_w": None,
        }],
        "n_tasks": len(tasks),
        "n_blend_identical_to_dense": n_identical,
        "n_blend_truncated": n_truncated,
    })

    cell_id = f"{spec['model_slug']}/cacheblend/{slug}/rep_{args.rep}"
    manifest_path = args.out_root / "manifest.json"
    cells: Dict[str, Any] = {}
    if manifest_path.exists():
        cells = json.loads(manifest_path.read_text(encoding="utf-8")).get("cells", {})
    cells[cell_id] = {
        "model": args.model,
        "mode": "cacheblend",
        "kv_point": {
            "recomp_ratio": args.recomp_ratio,
            "suffix_config": args.suffix_config,
        },
        "rep_index": args.rep,
        "seed": args.seed,
        "cell_dir": cell_id,
        "status": "done",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    artifacts.write_manifest(args.out_root, cells)

    print(
        f"\n[cacheblend] {len(tasks)} tasks in {duration / 60:.1f} min -> {run_dir}\n"
        f"  blend identical to dense: {n_identical}/{len(tasks)}\n"
        f"  blend hit the token cap : {n_truncated}/{len(tasks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
