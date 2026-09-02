"""Diagnose the CacheBlend environment before spending GPU time on it.

Every check here corresponds to a failure mode that is silent or misleadingly
reported if you just run the sweep:

* **transformers too new** - transformers 5.x needs torch >= 2.4, so against
  torch 2.2.1 it disables PyTorch support and continues. The first symptom is an
  unrelated-looking assertion inside vllm/config.py.
* **numpy 2.x** - torch 2.2.1's compiled extensions were built against numpy
  1.x and fail at import with a wall of text that does not name the cause.
* **rope_scaling shape** - vLLM 0.4.1 does ``assert "factor" in rope_scaling``
  and reads ``rope_scaling["type"]``. Newer transformers emit ``rope_type``
  instead, so the assert fires on models that are otherwise fine.
* **unpatched model file** - a stock qwen2.py loads and runs happily; blending
  simply never happens and the "blend" arm is an ordinary full prefill.
* **wrong attention backend** - blending lives only in the XFormers backend.
  FlashAttention silently bypasses it, with the same consequence.

Run this after pin_env.sh and before blend_runner.py.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List, Tuple

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

# (module, minimum inclusive, maximum exclusive or None)
_EXPECTED: List[Tuple[str, str, str]] = [
    ("numpy", "1.20", "2.0"),
    ("torch", "2.2", "2.3"),
    ("transformers", "4.40", "4.45"),
    ("xformers", "0.0.25", "0.0.26"),
]


def _tuple(version: str):
    parts = []
    for chunk in version.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_versions() -> bool:
    print("== package versions ==")
    ok = True
    for name, low, high in _EXPECTED:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<14} IMPORT FAILED: {exc}")
            ok = False
            continue
        version = getattr(mod, "__version__", "?")
        good = _tuple(low) <= _tuple(version) < _tuple(high)
        ok = ok and good
        print(f"  {name:<14} {version:<12} {'ok' if good else f'EXPECTED >={low},<{high}'}")
    if not ok:
        print("\n  -> run: bash cacheblend/pin_env.sh")
    return ok


def check_torch_cuda() -> bool:
    print("\n== torch / cuda ==")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  torch import failed: {exc}")
        return False
    available = torch.cuda.is_available()
    print(f"  cuda available : {available}")
    if available:
        print(f"  device         : {torch.cuda.get_device_name(0)}")
        print(f"  torch cuda ver : {torch.version.cuda}")
    # torch compiled against numpy 1.x will fail here under numpy 2.x.
    try:
        torch.zeros(2).numpy()
        print("  numpy bridge   : ok")
    except Exception as exc:  # noqa: BLE001
        print(f"  numpy bridge   : BROKEN ({exc})")
        print("                   -> numpy 2.x against a torch built for 1.x")
        return False
    return available


def check_config(model: str) -> bool:
    print(f"\n== model config: {model} ==")
    try:
        from transformers import AutoConfig
    except Exception as exc:  # noqa: BLE001
        print(f"  transformers import failed: {exc}")
        return False
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  config load FAILED: {exc}")
        return False

    arch = getattr(config, "architectures", None)
    print(f"  architectures  : {arch}")
    print(f"  num layers     : {getattr(config, 'num_hidden_layers', '?')}")
    print(f"  tie embeddings : {getattr(config, 'tie_word_embeddings', '?')}")

    rope = getattr(config, "rope_scaling", None)
    print(f"  rope_scaling   : {rope}")
    if rope is not None:
        if "factor" not in rope:
            print("    -> vLLM 0.4.1 asserts \"factor\" in rope_scaling; this WILL fail.")
            print("       Usually means transformers is too new and rewrote the field.")
            return False
        if "type" not in rope:
            print("    -> vLLM 0.4.1 reads rope_scaling['type']; only 'rope_type' present.")
            return False
        if rope["type"] not in ("linear", "dynamic", "yarn"):
            print(f"    -> vLLM 0.4.1 supports linear|dynamic|yarn, not {rope['type']!r}.")
            return False

    if arch and "Qwen2ForCausalLM" not in arch and "LlamaForCausalLM" not in arch:
        print(f"    -> no CacheBlend plumbing exists for {arch}.")
        return False
    return True


def check_patch() -> bool:
    print("\n== CacheBlend patch ==")
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    print(f"  VLLM_ATTENTION_BACKEND : {backend}")
    if backend != "XFORMERS":
        print("    -> blending exists only in the XFormers backend; any other")
        print("       backend runs a plain full prefill and calls it a blend.")
        return False

    ok = True
    try:
        from vllm.model_executor.models import qwen2 as qwen2_mod

        source_ok = all(
            marker in open(qwen2_mod.__file__, encoding="utf-8").read()
            for marker in ("cache_fuse_metadata", "hack_kv", "imp_indices")
        )
        print(f"  qwen2.py patched       : {source_ok}")
        if not source_ok:
            print("    -> apply cacheblend/patches/qwen2_cacheblend.patch and reinstall")
        ok = ok and source_ok
    except Exception as exc:  # noqa: BLE001
        print(f"  qwen2.py import FAILED : {exc}")
        ok = False

    try:
        import inspect

        from vllm.attention.layer import Attention

        params = list(inspect.signature(Attention.forward).parameters)
        patched = "cache_fuse_metadata" in params
        print(f"  Attention.forward      : {'patched' if patched else 'STOCK'}")
        ok = ok and patched
    except Exception as exc:  # noqa: BLE001
        print(f"  Attention import FAILED: {exc}")
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--skip-config", action="store_true",
                        help="skip the config check (avoids a network fetch)")
    args = parser.parse_args()

    results = {
        "versions": check_versions(),
        "torch/cuda": check_torch_cuda(),
    }
    if not args.skip_config:
        results["model config"] = check_config(args.model)
    results["patch"] = check_patch()

    print("\n== summary ==")
    for name, ok in results.items():
        print(f"  {name:<14} {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\nEnvironment looks correct. Next: blend_runner.py --self-test")
        return 0
    print("\nFix the FAIL rows before running the sweep; each one silently "
          "changes what the 'blend' arm actually computes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
