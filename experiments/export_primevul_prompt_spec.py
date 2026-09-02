"""Export the PrimeVul validator's exact token sequence for the CacheBlend arm.

The CacheBlend adapter runs on a different serving stack and must feed the model
*exactly* the token sequence KVCOMM feeds it. This drives KVCOMM's own methods
against a tokenizer-only shim -- no model is loaded, but every string- and
token-level decision is made by the real engine code -- and emits the same
A/T/B/C/D spec ``cacheblend/bench/prompt_builder.py`` already consumes, so the
blend runner needs no changes at all:

    merged = A ++ B ++ C ++ D

    A  chat BOS + validator system prompt + code introduction
    T  empty compatibility field
    C  the PrimeVul function
    B  empty compatibility field
    D  CWE catalogue + output ask + eot + assistant header

One spec per (model, arm): the vulnerable and patched arms are separate runs,
exactly as on the KVCOMM side.

Unlike the VIBE exporter this needs no frozen-inputs step -- PrimeVul supplies
the code, so there is nothing a generator could have drifted on. Pass
``--kvcomm-root`` to cross-check the reconstructed token counts against a
completed KVCOMM run rather than trusting them.

Usage::

    python experiments/export_primevul_prompt_spec.py --arm vulnerable
    python experiments/export_primevul_prompt_spec.py --arm patched \
        --kvcomm-root results/PrimeVulPatchedRun
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from repo_paths import bootstrap  # noqa: E402
bootstrap(require_kvcomm=True)

from experiments.export_validator_prompt_spec import (  # noqa: E402
    _build_shim,
    _ids,
    _model_slug,
    _sha256,
    _sha256_ids,
    _split_ids_at_text,
)
from experiments.secure_code_artifacts import task_slug  # noqa: E402
from datasets.primevul_dataset import DATA_DIR, load_tasks  # noqa: E402

# The provider is node 0 of the two-agent chain, so the code placeholder is
# {agent_0_current} (see experiments/run_primevul.py::build_graph).
DEFAULT_PROVIDER_AGENT_ID = "0"
_VALIDATOR_ROLE = "Security Validator"
PRIMEVUL_SCHEMA_VERSION = 2


def _kvcomm_dense_input_tokens(root: Path, model_slug: str) -> Dict[str, int]:
    """message -> dense validator input length, from a completed KVCOMM run.

    Only dense records count: a kv_reuse record's input length is assembled from
    blended segments and is not the reconstruction target.

    A sweep root can contain several models whose tokenizers assign different
    lengths to the same task.  Restrict the scan to the requested model when
    that model directory is present; otherwise retain support for callers that
    already pass a single-model root.
    """
    model_root = root / model_slug
    scan_root = model_root if model_root.is_dir() else root
    lengths: Dict[str, List[int]] = defaultdict(list)
    for latency_path in scan_root.rglob("Latency.json"):
        try:
            records = json.loads(latency_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in records:
            if (record.get("agent_role") == _VALIDATOR_ROLE
                    and record.get("mode") == "dense_prefill"
                    and record.get("message") and record.get("input_tokens")):
                lengths[record["message"]].append(int(record["input_tokens"]))
    # Identical inputs must give identical lengths; if not, say so by dropping it.
    return {message: values[0] for message, values in lengths.items()
            if len(set(values)) == 1}


def build_spec(model: str, tokenizer_source: str, arm: str, tasks_path: Path,
               provider_agent_id: str, kvcomm_root: Optional[Path],
               trust_remote_code: bool = True) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.prompt.primevul_common import (
        CWE_CATALOGUE,
        CWE_HEADER,
        OUTPUT_ASK,
        build_validator_user_prompt,
        validate_validator_prompt_structure,
    )
    from KVCOMM.prompt.primevul_prompt_set import VALIDATOR_INSTRUCTION

    tasks_data = load_tasks(str(tasks_path))
    if not tasks_data:
        raise SystemExit(f"{tasks_path} is empty")
    arms = {t["arm"] for t in tasks_data}
    if arms != {arm}:
        raise SystemExit(f"{tasks_path} holds arm(s) {sorted(arms)}, expected {arm!r}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source,
                                              trust_remote_code=trust_remote_code)
    shim = _build_shim(tokenizer)

    system_prompt = VALIDATOR_INSTRUCTION
    user_template = build_validator_user_prompt(provider_agent_id=provider_agent_id)
    # Hard gate: refuse to export a prompt whose structure would let the CWE
    # catalogue stop being a trailing exact-text segment.
    validate_validator_prompt_structure(user_template, provider_agent_id=provider_agent_id)

    messages = LLMChat._render_base_messages(shim, system_prompt, user_template)
    _, prompt_text, _ = LLMChat._build_chat_inputs(shim, messages, add_generation_prompt=True)
    _, _, segments = LLMChat.locate_placeholder(shim, prompt_text, return_segments=True)

    text_segments = [seg for seg in segments if seg[0] == "text"]
    if len(text_segments) != 2:
        raise SystemExit(
            f"{model}: expected 2 text segments (A/D), got {len(text_segments)}. "
            "The validator prompt structure changed; the chunking must be revisited."
        )
    seg_a, seg_d = text_segments
    a_text, a_ids = seg_a[1], _ids(seg_a[2])
    b_text, b_ids = "", []
    d_text, d_ids = seg_d[1], _ids(seg_d[2])
    request_drop_num = 0

    # Config A/B boundary inside segment D: the catalogue is the blended chunk,
    # the ask is the forced-exact suffix. KVCOMM blends the trailing segment too
    # (only layer 0 stays exact), so Config B is the structurally fair analogue.
    trailing_instruction = OUTPUT_ASK.split(",")[0]
    split_at = d_text.find(trailing_instruction)
    if split_at < 0:
        raise SystemExit(
            f"{model}: trailing instruction not found in segment D; cannot separate "
            "the rubric chunk from the forced-exact suffix."
        )
    rubric_text, suffix_matched_text = d_text[:split_at], d_text[split_at:]
    d_split = _split_ids_at_text(d_ids, rubric_text, tokenizer)
    if d_split is None:
        raise SystemExit(
            f"{model}: segment D has no token boundary at the rubric/suffix split; "
            "Config B chunking is not expressible without changing the token ids."
        )
    rubric_ids, suffix_matched_ids = d_ids[:d_split], d_ids[d_split:]

    run_lengths = (
        _kvcomm_dense_input_tokens(kvcomm_root, _model_slug(model))
        if kvcomm_root else {}
    )
    tasks: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    drift: List[int] = []

    for index, record in enumerate(tasks_data):
        request_text, code_text = record["task"], record["code"]
        t_ids: List[int] = []
        c_ids = tokenizer(code_text, add_special_tokens=False)["input_ids"]

        full_ids = a_ids + b_ids + c_ids + d_ids
        whole_len = len(tokenizer(tokenizer.decode(full_ids), add_special_tokens=False)["input_ids"])
        if whole_len != len(full_ids):
            drift.append(index)
        expected = run_lengths.get(_sha256(code_text))
        if expected is not None and expected != len(full_ids):
            mismatches.append({"task_index": index, "sample_id": record["sample_id"],
                               "expected": expected, "reconstructed": len(full_ids),
                               "delta": len(full_ids) - expected})

        tasks.append({
            "task_index": index,
            "task_slug": task_slug(index, request_text),
            "sample_id": record["sample_id"],
            "pair_id": record["pair_id"],
            "cwe": record["cwe"],
            "cwe_rule_number": record["cwe_rule_number"],
            "arm": record["arm"],
            "label": record["label"],
            "audit_key": _sha256(code_text),
            "seg_T": "",
            "seg_T_ids": t_ids,
            "seg_C": code_text,
            "seg_C_ids": c_ids,
            "full_ids": full_ids,
            "full_ids_sha256": _sha256_ids(full_ids),
            "n_tokens": len(full_ids),
            "whole_string_ids_len": whole_len,
            "kvcomm_dense_input_tokens": expected,
            "token_count_matches_run": None if expected is None else expected == len(full_ids),
            # The dataset supplies the code, so it cannot drift across cells and
            # nothing was generated that could have been truncated.
            "code_identical_across_kv": True,
            "generator_truncated": False,
        })

    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return {
        "schema_version": PRIMEVUL_SCHEMA_VERSION,
        "handoff_schema": "primevul_bare_code_kv_v2",
        "legacy_results_comparable": False,
        "model": model,
        "model_slug": _model_slug(model),
        "arm": arm,
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": _sha256(chat_template),
        # Named generator_agent_id for compatibility with the CacheBlend adapter;
        # on this arm the upstream agent supplies code rather than writing it.
        "generator_agent_id": provider_agent_id,
        "chat_markers": dict(shim._chat_markers),
        "default_assistant_prompt": shim.default_assistant_prompt,
        "invariant": {
            "system_prompt": system_prompt,
            "user_template": user_template,
            "cwe_header": CWE_HEADER,
            "cwe_catalogue": CWE_CATALOGUE,
            "task_prefix_text": "",
            "task_drop_num": int(request_drop_num),
            "seg_A": a_text, "seg_A_ids": a_ids,
            "seg_B": b_text, "seg_B_ids": b_ids,
            "seg_D": d_text, "seg_D_ids": d_ids,
            "rubric_chunk_text": rubric_text,
            "rubric_chunk_ids": rubric_ids,
            "suffix_matched_text": suffix_matched_text,
            "suffix_matched_ids": suffix_matched_ids,
            "suffix_matched_len": len(suffix_matched_ids),
            "suffix_native_text": d_text,
            "suffix_native_ids": d_ids,
            "suffix_native_len": len(d_ids),
        },
        "tasks": tasks,
        "checks": {
            "n_tasks": len(tasks),
            "n_cross_checked_vs_run": sum(1 for t in tasks
                                          if t["kvcomm_dense_input_tokens"] is not None),
            "n_token_count_mismatch_vs_run": len(mismatches),
            "token_count_mismatches": mismatches,
            # The spliced sequence is not the same as tokenizing the joined
            # string; this is why ids are exported rather than prompt text.
            "n_tokenization_drift_vs_whole_string": len(drift),
            "whole_string_drift_deltas": {
                "min": min((t["whole_string_ids_len"] - t["n_tokens"] for t in tasks), default=0),
                "max": max((t["whole_string_ids_len"] - t["n_tokens"] for t in tasks), default=0),
            },
            "n_tokens": {
                "min": min((t["n_tokens"] for t in tasks), default=0),
                "median": sorted(t["n_tokens"] for t in tasks)[len(tasks) // 2] if tasks else 0,
                "max": max((t["n_tokens"] for t in tasks), default=0),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", dest="models",
                        help="HF model id; repeatable. Defaults to the two Qwen models "
                             "(the CacheBlend fork predates Llama-3.2).")
    parser.add_argument("--arm", choices=["vulnerable", "patched", "both"], default="both")
    parser.add_argument("--tokenizer-source", action="append", default=[],
                        metavar="MODEL=SOURCE",
                        help="Override where a model's tokenizer is loaded from. The "
                             "chat_template hash is recorded so any substitution is auditable.")
    parser.add_argument("--provider-agent-id", default=DEFAULT_PROVIDER_AGENT_ID)
    parser.add_argument("--kvcomm-root", type=Path, default=None,
                        help="Completed KVCOMM run for the same arm; cross-checks the "
                             "reconstructed token counts against what the engine ran.")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "validator_prompt_spec")
    args = parser.parse_args()

    models = args.models or ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]
    overrides: Dict[str, str] = {}
    for item in args.tokenizer_source:
        if "=" not in item:
            raise SystemExit(f"--tokenizer-source expects MODEL=SOURCE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value

    arms = ["vulnerable", "patched"] if args.arm == "both" else [args.arm]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for arm in arms:
        tasks_path = DATA_DIR / f"tasks_{arm}.jsonl"
        for model in models:
            source = overrides.get(model, model)
            try:
                spec = build_spec(model, source, arm, tasks_path,
                                  args.provider_agent_id, args.kvcomm_root)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"{model} [{arm}]: FAILED - {type(exc).__name__}: {exc}")
                failures += 1
                continue
            out_path = args.out_dir / f"{_model_slug(model)}__{arm}_prompt_spec.json"
            out_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            checks, inv = spec["checks"], spec["invariant"]
            print(
                f"{model} [{arm}]: {checks['n_tasks']} tasks -> {out_path.name}\n"
                f"    prompt tokens: min={checks['n_tokens']['min']} "
                f"median={checks['n_tokens']['median']} max={checks['n_tokens']['max']}\n"
                f"    cross-checked vs run: {checks['n_cross_checked_vs_run']}"
                f"/{checks['n_tasks']} (mismatches={checks['n_token_count_mismatch_vs_run']})\n"
                f"    suffix_native={inv['suffix_native_len']} tok  "
                f"suffix_matched={inv['suffix_matched_len']} tok  "
                f"rubric_chunk={len(inv['rubric_chunk_ids'])} tok"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
