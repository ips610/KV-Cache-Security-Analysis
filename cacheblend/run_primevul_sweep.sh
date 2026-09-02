#!/usr/bin/env bash
# Run the CacheBlend arm over the PrimeVul artifacts, one arm at a time.
#
# The PrimeVul question is whether a *second* lossy cross-context reuse
# mechanism also suppresses detections that PrimeVul independently labels real.
# CacheBlend consumes a prompt spec, not a KVCOMM run, so nothing here differs
# from the VIBE arm except which spec is fed in -- blend_runner.py is unchanged.
#
#   bash cacheblend/run_primevul_sweep.sh [--arm vulnerable|patched|both]
#                                         [--stage 1|2] [--limit N]
#                                         [--config A|B] [--model SUBSTR]
#                                         [--noise-floor]
#
# Export the specs first (CPU only, on this machine or the run host):
#   python experiments/export_primevul_prompt_spec.py --arm both
#
# --noise-floor runs a second dense pass per sample, giving a within-cell
# dense-vs-dense floor. Use it. vLLM in bf16 can flip a greedy token between two
# mathematically identical computations, so the floor must be measured rather
# than assumed -- any silencing reported has to clear it.
#
# The two arms write to separate trees and keep separate anchor-free state, and
# are matched offline by pair_id, exactly as on the KVCOMM side.
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_DIR="$HARNESS_ROOT/datasets/primevul/validator_prompt_spec"
OUT_BASE="${OUT_BASE:-$HARNESS_ROOT/results}"
RUNNER="$HARNESS_ROOT/cacheblend/bench/blend_runner.py"

ARM="both"
STAGE=1
LIMIT=""
CONFIG="B"
NOISE=""
MODEL_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arm)         ARM="$2"; shift 2 ;;
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

case "$ARM" in
  both)               ARMS=(vulnerable patched) ;;
  vulnerable|patched) ARMS=("$ARM") ;;
  *) echo "arm must be vulnerable, patched or both" >&2; exit 1 ;;
esac

# Llama-3.2-3B is excluded for the same reason as the VIBE arm: the fork predates
# Llama-3.2 and lacks two upstream backports (llama3 RoPE scaling, tied
# embeddings), one of which fails silently by loading uninitialised weights.
MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

export VLLM_ATTENTION_BACKEND=XFORMERS

echo "arms=${ARMS[*]}  stage=$STAGE  config=$CONFIG  ratios=${RATIOS[*]}"
echo "specs=$SPEC_DIR"
echo

for arm in "${ARMS[@]}"; do
  out_root="$OUT_BASE/PrimeVul${arm^}CacheBlendRun"
  echo "### arm=$arm -> $out_root"
  for model in "${MODELS[@]}"; do
    if [ -n "$MODEL_FILTER" ] && [[ "$model" != *"$MODEL_FILTER"* ]]; then
      continue
    fi
    slug="${model//\//_}"
    spec="$SPEC_DIR/${slug}__${arm}_prompt_spec.json"
    if [ ! -f "$spec" ]; then
      echo "SKIP $model [$arm]: no prompt spec at $spec" >&2
      continue
    fi
    for r in "${RATIOS[@]}"; do
      echo "=============================================================="
      echo "  $model   arm=$arm   r=$r   config=$CONFIG"
      echo "=============================================================="
      # shellcheck disable=SC2086
      python "$RUNNER" \
        --model "$model" \
        --spec "$spec" \
        --out-root "$out_root" \
        --recomp-ratio "$r" \
        --suffix-config "$CONFIG" \
        --max-tokens 6000 \
        $NOISE $LIMIT
    done
  done
done

echo
echo "Score it (one cell at a time -- pooling recompute ratios would reproduce"
echo "the collapse make_kv_view.py prevents):"
echo
echo "  cd $HARNESS_ROOT"
echo "  python experiments/build_primevul_report.py \\"
echo "      --vuln-root $OUT_BASE/PrimeVulVulnerableCacheBlendRun \\"
echo "      --patched-root $OUT_BASE/PrimeVulPatchedCacheBlendRun \\"
echo "      --report docs/PRIMEVUL_CACHEBLEND_RESULTS.md"
