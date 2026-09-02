#!/usr/bin/env bash
# Build the patched CacheBlend fork in its own conda env.
#
# CacheBlend vendors vLLM 0.4.1 pinned to torch 2.2.1 + xformers 0.0.25. That
# stack cannot coexist with the KVCOMM env (torch 2.1.0 / transformers 4.50.2),
# so this creates a separate env and never touches the existing one.
#
# The blend kernel (vllm/attention/backends/xformers.py) is left byte-identical
# and its hash is recorded. The only change is model-file plumbing: the ~80-line
# patch that the fork already applies to llama.py, ported to qwen2.py.
#
#   bash cacheblend/setup_cacheblend.sh [install_dir] [env_name]
#
#   install_dir  default: $CACHEBLEND_ROOT if set, else <repo>/external/CacheBlend
#                (gitignored; CacheBlend is not distributed with this repository,
#                see external/README.md)
#   env_name     conda env to create/use (default: cacheblend)
#
# The fork is pinned to CACHEBLEND_SHA. The runs reported in the paper used the
# fork's main branch as of their run date; this pin is that branch's head at the
# time this harness was packaged, recorded so later upstream commits cannot
# silently change the kernel under the sha256 check below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-${CACHEBLEND_ROOT:-$HERE/../external/CacheBlend}}"
ENV_NAME="${2:-cacheblend}"
REPO_URL="https://github.com/YaoJiayi/CacheBlend.git"
CACHEBLEND_SHA="55ad02675939f783a38d579393527d218a7fd581"

echo "==> install dir : $INSTALL_DIR"
echo "==> conda env   : $ENV_NAME"

if [ ! -d "$INSTALL_DIR/.git" ]; then
  git clone "$REPO_URL" "$INSTALL_DIR"
  git -C "$INSTALL_DIR" checkout --detach "$CACHEBLEND_SHA"
else
  echo "==> already cloned, reusing"
fi
cd "$INSTALL_DIR"
HEAD_SHA="$(git rev-parse HEAD)"
if [ "$HEAD_SHA" != "$CACHEBLEND_SHA" ]; then
  echo "FATAL: $INSTALL_DIR is at $HEAD_SHA, expected $CACHEBLEND_SHA." >&2
  echo "       git -C \"$INSTALL_DIR\" fetch origin && git -C \"$INSTALL_DIR\" checkout --detach $CACHEBLEND_SHA" >&2
  exit 1
fi
echo "==> CacheBlend commit: $CACHEBLEND_SHA"

KERNEL="vllm_blend/vllm/attention/backends/xformers.py"
KERNEL_SHA="$(sha256sum "$KERNEL" | cut -d' ' -f1)"
echo "==> blend kernel sha256: $KERNEL_SHA"

# Idempotent: skip if the plumbing is already present.
if grep -q "cache_fuse_metadata" vllm_blend/vllm/model_executor/models/qwen2.py; then
  echo "==> qwen2.py already patched"
else
  echo "==> applying qwen2 plumbing patch"
  git apply --verbose "$HERE/patches/qwen2_cacheblend.patch"
fi

python - <<'PY'
import pathlib, sys
p = pathlib.Path("vllm_blend/vllm/model_executor/models/qwen2.py")
src = p.read_text()
missing = [m for m in ("cache_fuse_metadata", "hack_kv", "old_kvs", "imp_indices")
           if m not in src]
if missing:
    sys.exit(f"qwen2.py is missing {missing}; the patch did not apply")
compile(src, str(p), "exec")
print("==> qwen2.py patched and parses")
PY

KERNEL_SHA_AFTER="$(sha256sum "$KERNEL" | cut -d' ' -f1)"
if [ "$KERNEL_SHA" != "$KERNEL_SHA_AFTER" ]; then
  echo "FATAL: the blend kernel changed. It must stay the authors' code." >&2
  exit 1
fi
echo "==> blend kernel unmodified (verified)"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "==> creating conda env $ENV_NAME (python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> installing vllm_blend from source (CUDA kernels; 20-40 min)"
cd "$INSTALL_DIR/vllm_blend"
MAX_JOBS="${MAX_JOBS:-4}" pip install -e .
cd "$INSTALL_DIR"
pip install -r requirements.txt

# The fork's requirements are open-ended (`transformers >= 4.40.0`, numpy
# unpinned), so the install above resolves to a stack years newer than the code:
# transformers 5.x (which needs torch >= 2.4 and therefore disables PyTorch
# support against torch 2.2.1) and numpy 2.x (which torch 2.2.1's compiled
# extensions cannot load). Both fail indirectly, via an assertion in
# vllm/config.py that names neither. Pin them back.
echo "==> pinning to the vLLM 0.4.1-era stack"
bash "$HERE/pin_env.sh"

cat > "$INSTALL_DIR/bench_env.json" <<EOF
{
  "cacheblend_commit": "$CACHEBLEND_SHA",
  "blend_kernel_sha256": "$KERNEL_SHA",
  "conda_env": "$ENV_NAME",
  "install_dir": "$INSTALL_DIR",
  "patches_applied": ["qwen2_cacheblend.patch"],
  "kernel_modified": false
}
EOF

echo
echo "==> done. Provenance written to $INSTALL_DIR/bench_env.json"
echo "    conda activate $ENV_NAME"
echo "    export VLLM_ATTENTION_BACKEND=XFORMERS"
