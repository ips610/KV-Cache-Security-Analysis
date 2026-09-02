#!/usr/bin/env bash
# Pin the CacheBlend env to versions contemporaneous with vLLM 0.4.1.
#
# The fork's requirements are open-ended (`transformers >= 4.40.0`, numpy
# unpinned), so a fresh `pip install -e .` today resolves to a stack years newer
# than the code. Two failures follow, and neither is obvious from its traceback:
#
#   transformers 5.x  requires torch >= 2.4, so with torch 2.2.1 it disables
#                     PyTorch support entirely and keeps going. It also
#                     normalises configs to rope_scaling={"rope_type": ...},
#                     which has no "factor" key, tripping
#                     `assert "factor" in rope_scaling` in vllm/config.py.
#   numpy 2.x         torch 2.2.1's C extensions were compiled against numpy
#                     1.x: "A module that was compiled using NumPy 1.x cannot
#                     be run in NumPy 2.2.6".
#
# Safe to re-run. Does not rebuild the CUDA kernels.
#
#   bash cacheblend/pin_env.sh
set -euo pipefail

echo "==> pinning to the vLLM 0.4.1-era stack"
echo "    (torch 2.2.1 / xformers 0.0.25 stay as built)"

# transformers 4.40.2 is contemporaneous with vLLM 0.4.1 and supports the Qwen2
# architecture that Qwen2.5 uses. huggingface_hub must stay <1.0 for it, and
# tokenizers is bounded by transformers itself.
pip install --no-cache-dir \
  "numpy==1.26.4" \
  "transformers==4.40.2" \
  "tokenizers==0.19.1" \
  "huggingface_hub==0.23.0"

echo
echo "==> installed versions"
python - <<'PY'
import importlib
for name in ("numpy", "torch", "transformers", "tokenizers", "huggingface_hub", "xformers"):
    try:
        mod = importlib.import_module(name)
        print(f"    {name:<18} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"    {name:<18} IMPORT FAILED: {exc}")
PY

echo
echo "==> now verify:  python cacheblend/bench/check_env.py --model Qwen/Qwen2.5-Coder-7B-Instruct"
