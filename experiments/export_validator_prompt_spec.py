"""Export the Security Validator's exact token sequence for the CacheBlend arm.

The CacheBlend adapter lives in another repo, on another serving stack, and must
feed the model *exactly* the token sequence KVCOMM fed it. Re-deriving that by
hand is fragile, so this script drives KVCOMM's own methods
(``_build_chat_inputs``, ``locate_placeholder``, ``tokenize_segment``,
``format_chat_segment``) against a tokenizer-only shim: no model is loaded, but
every string- and token-level decision is made by the real engine code.

The sequence KVCOMM actually runs is assembled in ``agen_kvcomm``
(``KVCOMM/llm/gpt_chat.py``) as::

    merged = prefix[0] ++ (T ++ prefix[1]) ++ (C ++ prefix[2])
           =    A     ++ (T ++    B    ) ++ (C ++    D    )

where A/B/D are the literal text segments of the templated prompt and T/C are
the runtime contents of ``{user_question}`` and ``{agent_<id>_current}``. Note
this is *not* ``tokenizer(full_prompt_text)`` - it is a splice of separately
tokenized segments, which is exactly why the ids have to be exported rather
than reconstructed from a string downstream.

Two chunkings of the trailing segment D are emitted, because KVCOMM approximates
the NIST rule block's KV too (``kvcomm_engine.py:1015-1022`` adds a blended
delta to the trailing prefix segment, keeping only layer 0 exact):

* ``suffix_matched`` (Config B) - rules stay a blended chunk, mirroring KVCOMM.
* ``suffix_native``  (Config A) - rules fall inside CacheBlend's forced-exact
  suffix, which is its canonical query-in-suffix usage.

Every task carries ``kvcomm_dense_input_tokens`` from the recorded run, so the
reconstruction is checked against ground truth rather than trusted.

Usage::

    python experiments/export_validator_prompt_spec.py
    python experiments/export_validator_prompt_spec.py --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from repo_paths import bootstrap  # noqa: E402
bootstrap(require_kvcomm=True)

SCHEMA_VERSION = 1

_DEFAULT_FROZEN = _REPO_ROOT / "datasets" / "secure_code" / "frozen_validator_inputs"

# Model id -> slug used for result folders (mirrors experiments/sweep.py).
def _model_slug(model: str) -> str:
    return model.replace("/", "_")


# The trailing instruction that ends the user turn. Splitting segment D here
# separates "the rubric" from "the ask", which is the Config A/B boundary.
_TRAILING_INSTRUCTION = "Produce a security validation report"

_TASK_PREFIX = "The task is: "
_VALIDATOR_ROLE = "Security Validator"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_ids(ids: List[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_shim(tokenizer):
    """Instantiate LLMChat without loading a model.

    ``LLMChat.__init__`` loads weights, so we bypass it and populate only the
    attributes the string/token methods touch. The methods themselves are the
    real ones - nothing here reimplements engine behaviour.
    """
    from KVCOMM.llm.gpt_chat import LLMChat

    shim = LLMChat.__new__(LLMChat)
    shim.tokenizer = tokenizer
    # locate_placeholder / tokenize_segment call `.to(self.model.device)`.
    shim.model = types.SimpleNamespace(device="cpu")
    shim._chat_markers = LLMChat._extract_chat_markers(shim)
    shim.default_assistant_prompt = "A: "
    shim.base_messages_template = [
        {"role": "system", "content": "{system_prompt}"},
        {"role": "user", "content": "{user_prompt}"},
    ]
    # Agents construct the LLM with prefix="" (security_validator.py:37), which
    # routes through _prepare_prefix_template and clears the assistant prompt.
    LLMChat._prepare_prefix_template(shim, "")
    return shim


def _ids(encoding: Dict[str, Any]) -> List[int]:
    return [int(x) for x in encoding["input_ids"][0].tolist()]


def _split_ids_at_text(
    ids: List[int], prefix_text: str, tokenizer
) -> Optional[int]:
    """Return k such that decode(ids[:k]) == prefix_text, or None.

    Splitting in *token* space keeps ``full_ids`` byte-identical to what KVCOMM
    ran, which re-tokenizing the two halves separately would not guarantee.
    """
    for k in range(len(ids) + 1):
        if tokenizer.decode(ids[:k]) == prefix_text:
            return k
    return None


def _load_frozen(frozen_dir: Path, model_slug: str) -> List[Dict[str, Any]]:
    path = frozen_dir / f"{model_slug}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"frozen inputs not found: {path}\n"
            f"Run experiments/export_frozen_inputs.py first."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_spec(
    model: str,
    tokenizer_source: str,
    frozen_dir: Path,
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.prompt.secure_code_common import (
        NIST_HEADER,
        build_validator_user_prompt,
        validate_validator_prompt_structure,
    )
    from KVCOMM.prompt.secure_code_prompt_set import NIST_RULES, VALIDATOR_INSTRUCTION

    slug = _model_slug(model)
    frozen = _load_frozen(frozen_dir, slug)
    if not frozen:
        raise SystemExit(f"{model}: frozen input file is empty")

    generator_agent_id = frozen[0].get("generator_agent_id")
    if not generator_agent_id:
        raise SystemExit(f"{model}: generator_agent_id missing from frozen inputs")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=trust_remote_code
    )
    shim = _build_shim(tokenizer)

    system_prompt = VALIDATOR_INSTRUCTION
    user_template = build_validator_user_prompt(
        generator_agent_id=generator_agent_id, nist_rules=NIST_RULES
    )
    # Hard gate: refuse to export a prompt whose structure would let the rules
    # stop being a trailing segment.
    validate_validator_prompt_structure(
        user_template, generator_agent_id=generator_agent_id
    )

    messages = LLMChat._render_base_messages(shim, system_prompt, user_template)
    _, prompt_text, _ = LLMChat._build_chat_inputs(
        shim, messages, add_generation_prompt=True
    )
    _, _, segments = LLMChat.locate_placeholder(shim, prompt_text, return_segments=True)

    text_segments = [seg for seg in segments if seg[0] == "text"]
    if len(text_segments) != 3:
        raise SystemExit(
            f"{model}: expected 3 text segments (A/B/D), got {len(text_segments)}. "
            "The validator prompt structure changed; the chunking must be revisited."
        )
    seg_a, seg_b, seg_d = text_segments
    a_text, a_ids = seg_a[1], _ids(seg_a[2])
    b_text, b_ids = seg_b[1], _ids(seg_b[2])
    d_text, d_ids = seg_d[1], _ids(seg_d[2])

    # drop_num for {user_question}: update_input_anchor tokenizes
    # "The task is: " + task with the user header, then drops the header +
    # prefix so they are not duplicated after segment A.
    task_drop_num = LLMChat.tokenize_segment(
        shim,
        role="user",
        content=_TASK_PREFIX,
        include_begin=True,
        include_eot=False,
        add_special_tokens=False,
    )["input_ids"].shape[-1]

    # Config A/B boundary inside segment D.
    split_at = d_text.find(_TRAILING_INSTRUCTION)
    if split_at < 0:
        raise SystemExit(
            f"{model}: trailing instruction not found in segment D; cannot "
            "separate the rubric chunk from the forced-exact suffix."
        )
    rubric_text, suffix_matched_text = d_text[:split_at], d_text[split_at:]
    d_split = _split_ids_at_text(d_ids, rubric_text, tokenizer)
    if d_split is None:
        raise SystemExit(
            f"{model}: segment D has no token boundary at the rubric/suffix split; "
            "Config B chunking is not expressible without changing the token ids."
        )
    rubric_ids, suffix_matched_ids = d_ids[:d_split], d_ids[d_split:]

    tasks: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    drift: List[int] = []

    for record in frozen:
        task_text = record["task"]
        code_text = record["generated_code"]

        t_full = LLMChat.tokenize_segment(
            shim,
            role="user",
            content=_TASK_PREFIX + task_text,
            include_begin=True,
            include_eot=False,
            add_special_tokens=False,
        )
        t_ids = [int(x) for x in t_full["input_ids"][0].tolist()][task_drop_num:]
        c_ids = tokenizer(code_text, add_special_tokens=False)["input_ids"]

        full_ids = a_ids + t_ids + b_ids + c_ids + d_ids
        full_text = tokenizer.decode(full_ids)
        whole_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        expected = record.get("kvcomm_dense_input_tokens")

        truncated = bool(record.get("generator_truncated"))
        if whole_len != len(full_ids):
            drift.append(record["task_index"])
        if expected is not None and expected != len(full_ids):
            mismatches.append(
                {
                    "task_index": record["task_index"],
                    "expected": expected,
                    "reconstructed": len(full_ids),
                    "delta": len(full_ids) - expected,
                    "generator_truncated": truncated,
                }
            )

        tasks.append(
            {
                "task_index": record["task_index"],
                "task_slug": record["task_slug"],
                "seg_T": task_text,
                "seg_T_ids": t_ids,
                "seg_C": code_text,
                "seg_C_ids": c_ids,
                "full_ids": full_ids,
                "full_ids_sha256": _sha256_ids(full_ids),
                "n_tokens": len(full_ids),
                "whole_string_ids_len": whole_len,
                "kvcomm_dense_input_tokens": expected,
                "token_count_matches_run": expected is None or expected == len(full_ids),
                "code_identical_across_kv": record["code_identical_across_kv"],
                "generator_truncated": truncated,
                "kvcomm_natural_mode_by_kv": record["kvcomm_natural_mode_by_kv"],
            }
        )

    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "model_slug": slug,
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": _sha256(chat_template),
        "generator_agent_id": generator_agent_id,
        "chat_markers": dict(shim._chat_markers),
        "default_assistant_prompt": shim.default_assistant_prompt,
        "invariant": {
            "system_prompt": system_prompt,
            "user_template": user_template,
            "nist_header": NIST_HEADER,
            "nist_rules": NIST_RULES,
            "task_prefix_text": _TASK_PREFIX,
            "task_drop_num": int(task_drop_num),
            "seg_A": a_text,
            "seg_A_ids": a_ids,
            "seg_B": b_text,
            "seg_B_ids": b_ids,
            "seg_D": d_text,
            "seg_D_ids": d_ids,
            # Config B: rubric is a blended chunk, only the ask is forced exact.
            "rubric_chunk_text": rubric_text,
            "rubric_chunk_ids": rubric_ids,
            "suffix_matched_text": suffix_matched_text,
            "suffix_matched_ids": suffix_matched_ids,
            "suffix_matched_len": len(suffix_matched_ids),
            # Config A: the whole trailing segment is forced exact.
            "suffix_native_text": d_text,
            "suffix_native_ids": d_ids,
            "suffix_native_len": len(d_ids),
        },
        "tasks": tasks,
        "checks": {
            "n_tasks": len(tasks),
            # `generated_code.md` holds the DECODED generator output, so
            # re-encoding it need not reproduce the ids the generator emitted.
            # Two separate causes, kept apart because they mean different things:
            #   - truncated: the generator hit max_tokens, so the saved code is
            #     cut mid-output and the engine also drops the final token.
            #   - round-trip: plain decode/re-encode non-idempotence, usually +-1.
            "n_token_count_mismatch_vs_run": len(mismatches),
            "n_mismatch_generator_truncated": sum(
                1 for m in mismatches if m["generator_truncated"]
            ),
            "n_mismatch_roundtrip_only": sum(
                1 for m in mismatches if not m["generator_truncated"]
            ),
            "max_abs_delta_roundtrip_only": max(
                (abs(m["delta"]) for m in mismatches if not m["generator_truncated"]),
                default=0,
            ),
            "token_count_mismatches": mismatches,
            # The spliced sequence is not the same as tokenizing the joined
            # string; this is why ids are exported rather than prompt text.
            "n_tokenization_drift_vs_whole_string": len(drift),
            "whole_string_drift_deltas": {
                "min": min(
                    (t["whole_string_ids_len"] - t["n_tokens"] for t in tasks), default=0
                ),
                "max": max(
                    (t["whole_string_ids_len"] - t["n_tokens"] for t in tasks), default=0
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="HF model id; repeatable. Defaults to the three study models.",
    )
    parser.add_argument(
        "--tokenizer-source",
        action="append",
        default=[],
        metavar="MODEL=SOURCE",
        help=(
            "Override where a model's tokenizer is loaded from, e.g. for gated "
            "repos. The chat_template hash is recorded so any substitution is "
            "auditable."
        ),
    )
    parser.add_argument("--frozen-dir", type=Path, default=_DEFAULT_FROZEN)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "datasets" / "secure_code" / "validator_prompt_spec",
    )
    args = parser.parse_args()

    models = args.models or [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    overrides: Dict[str, str] = {}
    for item in args.tokenizer_source:
        if "=" not in item:
            raise SystemExit(f"--tokenizer-source expects MODEL=SOURCE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for model in models:
        source = overrides.get(model, model)
        try:
            spec = build_spec(model, source, args.frozen_dir)
        except Exception as exc:  # noqa: BLE001 - report and continue to next model
            print(f"{model}: FAILED - {type(exc).__name__}: {exc}")
            failures += 1
            continue

        out_path = args.out_dir / f"{_model_slug(model)}_prompt_spec.json"
        out_path.write_text(
            json.dumps(spec, ensure_ascii=False), encoding="utf-8"
        )
        checks = spec["checks"]
        inv = spec["invariant"]
        matched = checks["n_tasks"] - checks["n_token_count_mismatch_vs_run"]
        drift = checks["whole_string_drift_deltas"]
        print(
            f"{model}: {checks['n_tasks']} tasks -> {out_path.name}\n"
            f"    token_count_vs_run: {matched}/{checks['n_tasks']} exact"
            f"  (truncated={checks['n_mismatch_generator_truncated']},"
            f" roundtrip={checks['n_mismatch_roundtrip_only']},"
            f" max|delta|_roundtrip={checks['max_abs_delta_roundtrip_only']})\n"
            f"    whole_string_drift: {checks['n_tokenization_drift_vs_whole_string']}"
            f"/{checks['n_tasks']} tasks, delta in [{drift['min']}, {drift['max']}]\n"
            f"    suffix_native={inv['suffix_native_len']} tok  "
            f"suffix_matched={inv['suffix_matched_len']} tok  "
            f"rubric_chunk={len(inv['rubric_chunk_ids'])} tok"
        )
    if failures:
        return 1
    print(
        "\nNote: token-count mismatches are expected where the generator was "
        "truncated (the saved code is cut off and the engine drops the final "
        "token) or where decode/re-encode is not idempotent. The headline "
        "metric is within-system paired -- CacheBlend blend vs CacheBlend dense "
        "on the SAME ids -- so these affect only the cross-system engine floor, "
        "which must be reported separately."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
