"""Turn an exported validator prompt spec into CacheBlend chunks + suffix.

CacheBlend prefills each chunk independently, concatenates the per-chunk KV,
re-rotates K to the new absolute positions, and then recomputes only the
highest-deviation tokens plus a forced-exact suffix. So the adapter's job is to
partition the validator's token sequence into chunks whose concatenation is
*exactly* the sequence KVCOMM ran.

The validator prompt has five segments (see
experiments/export_validator_prompt_spec.py)::

    A  chat BOS + system prompt + "The task is: "
    T  the task text
    B  "A developer agent produced the following code:"
    C  the generated code
    D  NIST header + rules + trailing ask + eot + assistant header

Two suffix configurations, because KVCOMM approximates the rule block's KV too
(kvcomm_engine.py:1015-1022 adds a blended delta to the trailing prefix segment,
keeping only layer 0 exact):

    Config B (suffix_matched, primary)
        chunks = [A, T, B+C, rubric]      suffix = trailing ask
        The rubric is blended, mirroring KVCOMM. The structurally fair analogue.

    Config A (suffix_native, secondary)
        chunks = [A, T, B+C]              suffix = all of D
        The rubric falls inside the forced-exact suffix. CacheBlend's canonical
        query-in-suffix usage, and the charitable setting for it.

Both partition the same token sequence; only the chunk/suffix boundary moves.
That is asserted on every build, not assumed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONFIG_MATCHED = "B"
CONFIG_NATIVE = "A"


def load_spec(spec_path: Path) -> Dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    inv = spec["invariant"]
    # The exporter guarantees this, but the adapter runs in a different repo on
    # a different stack, so re-check rather than trust the file.
    if inv["rubric_chunk_ids"] + inv["suffix_matched_ids"] != inv["seg_D_ids"]:
        raise ValueError(
            f"{spec_path}: rubric chunk + matched suffix != segment D. "
            "The spec is inconsistent; re-export it."
        )
    if inv["suffix_native_ids"] != inv["seg_D_ids"]:
        raise ValueError(f"{spec_path}: native suffix != segment D.")
    return spec


def sha256_ids(ids: List[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build(
    spec: Dict[str, Any], task: Dict[str, Any], suffix_config: str
) -> Tuple[List[List[int]], List[int], List[int]]:
    """Return ``(chunks, suffix_ids, full_ids)`` for one task.

    Raises if the partition does not reproduce the exported sequence, so a
    mis-chunked prompt can never reach the model and silently produce a report.
    """
    inv = spec["invariant"]
    a_ids = inv["seg_A_ids"]
    b_ids = inv["seg_B_ids"]
    t_ids = task["seg_T_ids"]
    c_ids = task["seg_C_ids"]

    if suffix_config == CONFIG_MATCHED:
        chunks = [a_ids, t_ids, b_ids + c_ids, inv["rubric_chunk_ids"]]
        suffix_ids = inv["suffix_matched_ids"]
    elif suffix_config == CONFIG_NATIVE:
        chunks = [a_ids, t_ids, b_ids + c_ids]
        suffix_ids = inv["suffix_native_ids"]
    else:
        raise ValueError(
            f"unknown suffix config {suffix_config!r}; expected "
            f"{CONFIG_MATCHED!r} (rubric blended) or {CONFIG_NATIVE!r} (rubric exact)"
        )

    chunks = [c for c in chunks if c]
    full_ids: List[int] = [tok for chunk in chunks for tok in chunk] + list(suffix_ids)

    expected = task["full_ids"]
    if full_ids != expected:
        raise ValueError(
            f"task {task['task_index']}: chunk partition does not reproduce "
            f"full_ids (built {len(full_ids)} tokens, expected {len(expected)}). "
            "Refusing to run on a prompt that differs from the exported sequence."
        )
    if sha256_ids(full_ids) != task["full_ids_sha256"]:
        raise ValueError(
            f"task {task['task_index']}: full_ids hash mismatch against the spec."
        )
    if not suffix_ids:
        raise ValueError(
            f"task {task['task_index']}: empty suffix. CacheBlend always forces "
            "the suffix exact; an empty one means the split is wrong."
        )
    return chunks, list(suffix_ids), full_ids


def exact_fraction(
    chunks: List[List[int]], suffix_ids: List[int], recomp_ratio: float
) -> Dict[str, float]:
    """Fraction of tokens that end up exactly recomputed, for reporting.

    Mirrors the kernel's own arithmetic
    (vllm/attention/backends/xformers.py): ``topk_num`` is taken over the
    NON-suffix span only, and the suffix is appended unconditionally. This is
    the axis on which CacheBlend and KVCOMM are actually comparable, since
    recomp_ratio and KVCOMM's entropy gate are not commensurable quantities.
    """
    total = sum(len(c) for c in chunks) + len(suffix_ids)
    suffix_len = len(suffix_ids)
    topk_num = int((total - suffix_len) * recomp_ratio)
    return {
        "n_tokens": total,
        "suffix_len": suffix_len,
        "topk_num": topk_num,
        "n_exact": topk_num + suffix_len,
        "exact_fraction": (topk_num + suffix_len) / total if total else 0.0,
    }
