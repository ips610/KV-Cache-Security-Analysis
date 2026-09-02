#!/usr/bin/env bash
# Run the CacheBlend arm across models and recompute ratios.
#
# One fresh subprocess per cell, mirroring experiments/sweep.py, so no engine
# state can leak between configurations.
#
#   bash cacheblend/run_sweep.sh [--stage 1|2] [--limit N] [--config A|B]
#                                [--model SUBSTR] [--noise-floor]
#
# --noise-floor runs a second dense pass per task, giving a within-cell
# dense-vs-dense floor. Use it. KVCOMM's HF runs measured that floor at 0.0%,
# but vLLM in bf16 can flip a greedy token between two mathematically identical
# computations, so here it must be measured rather than assumed -- any silencing
# we report has to clear it.
#
# OUT_ROOT overrides the destination, e.g. for a throwaway pilot:
#   OUT_ROOT=/tmp/cbpilot bash cacheblend/run_sweep.sh --stage 1 --limit 3
#
# Stage 1 is the headline plus the harness controls: r in {0.00, 0.16, 1.00}.
#   r=0.00  only the suffix is exact -- the maximally lossy lower anchor. Without
#           it, low silencing could just mean a bigger exact-token budget.
#   r=0.16  CacheBlend's own default.
#   r=1.00  near-dense control. NOT dense: layers 0-1 still read chunk-local KV.
# Stage 2 fills in the curve: r in {0.04, 0.08, 0.32, 0.50}.
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_DIR="$HARNESS_ROOT/datasets/secure_code/validator_prompt_spec"
OUT_ROOT="${OUT_ROOT:-$HARNESS_ROOT/results/CacheBlendRun}"
RUNNER="$HARNESS_ROOT/cacheblend/bench/blend_runner.py"

STAGE=1
LIMIT=""
CONFIG="B"
NOISE=""
MODEL_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage)       STAGE="$2"; shift 2 ;;
    --limit)       LIMIT="--limit $2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --model)       MODEL_FILTER="$2"; shift 2 ;;
    --noise-floor) NOISE="--noise-floor-baseline"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$STAGE" in
  1) RATIOS=(0.00 0.16 1.00) ;;
  2) RATIOS=(0.04 0.08 0.32 0.50) ;;
  *) echo "stage must be 1 or 2" >&2; exit 1 ;;
esac

# Llama-3.2-3B is excluded deliberately. Its corpus does not support a replayed
# comparison: 37/90 generations were truncated at the token cap, 59/90 differ
# across kv points despite a dense greedy generator, and only 22/90 prompts
# rebuild token-exactly -- leaving 9-13 comparable cases per kv point. It also
# needs two upstream backports (llama3 RoPE, tied embeddings) that the fork
# lacks, one of which fails silently by loading uninitialised weights.
MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "stage=$STAGE  config=$CONFIG  ratios=${RATIOS[*]}"
echo "out=$OUT_ROOT"
echo

for model in "${MODELS[@]}"; do
  if [ -n "$MODEL_FILTER" ] && [[ "$model" != *"$MODEL_FILTER"* ]]; then
    continue
  fi
  slug="${model//\//_}"
  spec="$SPEC_DIR/${slug}_prompt_spec.json"
  if [ ! -f "$spec" ]; then
    echo "SKIP $model: no prompt spec at $spec" >&2
    continue
  fi
  for r in "${RATIOS[@]}"; do
    echo "=============================================================="
    echo "  $model   r=$r   config=$CONFIG"
    echo "=============================================================="
    # shellcheck disable=SC2086
    python "$RUNNER" \
      --model "$model" \
      --spec "$spec" \
      --out-root "$OUT_ROOT" \
      --recomp-ratio "$r" \
      --suffix-config "$CONFIG" \
      $NOISE $LIMIT
  done
done

echo
echo "Done. Score it with KVCOMM's own analysis (one cell at a time -- pooling"
echo "recompute ratios would reproduce the collapse make_kv_view.py prevents):"
echo
echo "  cd $HARNESS_ROOT"
echo "  python experiments/make_kv_view.py $OUT_ROOT --all"
echo "  for v in ${OUT_ROOT}__view_*; do python experiments/paper_metrics.py \"\$v\"; done"
