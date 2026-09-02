# Fast but Forgetful: Cross-Context KV Reuse Breaks Decision Fidelity in LLM Security Review

Experiment harness for the paper of the same title (under double-blind review;
attribution, citation and license will be added after the review period).

The paper asks whether cross-context KV-cache reuse, an inference optimisation
for multi-agent LLM pipelines, preserves the *decisions* a security-review agent
makes. The harness holds the model, code, prompt and greedy decoding fixed and
compares approximate KV reuse against identical-input exact dense inference:

- **KVCOMM arm (primary).** A two-agent pipeline, `CodeGenerator` then
  `SecurityValidator` reviewing the generated code against ten NIST-derived
  rules, on 90 MiniMaxAI/VIBE application tasks. On every natural anchor hit the
  validator is also run as a measurement-only dense pass on the same input, so
  each reuse has a paired exact reference. Threshold sweep gamma in {0.1, 0.3, 0.5}.
- **CacheBlend arm (replication).** The same validator token sequences replayed
  through CacheBlend's selective recomputation, r in {0.00, 0.16, 1.00}.
- **PrimeVul arm (externally labelled).** 100 vulnerable / 100 patched C
  functions reviewed against ten CWE rules, under both mechanisms, scored
  against ground truth.

## What is (and is not) in this repository

This repository contains only the harness: runners, sweep orchestration,
dataset builders and samples, analysis and reporting scripts, tests, and the
harness's own modifications to the two systems under study.

**Third-party code is not included.** Two directories must be present before
anything GPU-side runs, and the setup scripts create them:

| Expected path         | Contents | Created by |
|-----------------------|----------|------------|
| `external/KVCOMM`     | KVCOMM should be present here: upstream [FastMAS/KVCOMM](https://github.com/FastMAS/KVCOMM) at a pinned commit, with `kvcomm_patch/` applied | `bash kvcomm_patch/setup_kvcomm.sh` |
| `external/CacheBlend` | CacheBlend should be present here: the authors' [fork](https://github.com/YaoJiayi/CacheBlend) at a pinned commit, with `cacheblend/patches/` applied, built in its own conda env | `bash cacheblend/setup_cacheblend.sh` |

See [`external/README.md`](external/README.md) for why they are fetched rather
than vendored (KVCOMM has no license file; CacheBlend is a vLLM fork with
compiled kernels) and for the `KVCOMM_ROOT` / `CACHEBLEND_ROOT` overrides.

## Layout

| Path | Purpose |
|---|---|
| `kvcomm_patch/` | Patch for five upstream KVCOMM files + eight new agent/prompt files + `setup_kvcomm.sh`. The only KVCOMM-related code shipped here. |
| `cacheblend/` | CacheBlend arm: setup/pin scripts, the `qwen2.py` plumbing patch, the blend runner and sweep scripts. |
| `experiments/` | Runners (`run_secure_code.py`, `run_primevul.py`), sweep orchestration (`sweep.py`, `harness/`, `sweeps/*.yaml`), analysis and reporting. |
| `datasets/` | Task corpora and builders: `secure_code/` (VIBE tasks, frozen validator inputs, prompt specs) and `primevul/` (sampled pairs, prompt specs, `build_dataset.py`). |
| `generatedCodebases/` | Agent 1 outputs for the 90 VIBE tasks per model: the fixed code artifacts both validator executions review. |
| `tests/` | CPU test suite (GPU tests opt-in). |
| `docs/` | `PROMPTS.md` (every prompt verbatim), `SWEEP_CHEATSHEET.md` (runbook), `TASK_SOURCES.md` (corpus provenance), `PRIMEVUL_EXPERIMENT_PLAN.md`, `methodology.tex`. |
| `run_generated_codebase_validation.py` | Standalone dense replay with plain Transformers (no KVCOMM import), used as an independent cross-check. |
| `repo_paths.py` | Locates the repo and the two external checkouts; every entry point calls `bootstrap()`. |
| `external/` | Placeholder for the third-party checkouts (gitignored). |

`results/`, `runs/` and `docs/experiment_reports/` are output locations and are
gitignored; the paper's raw run trees are not distributed here.

## Setup

### Arm A: KVCOMM (Python 3.10, one environment)

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n kvfid python=3.10
pip install -r requirements.txt
bash kvcomm_patch/setup_kvcomm.sh                      # clone, pin, patch, overlay, import check
python -m pytest tests -q                              # CPU tests
```

`setup_kvcomm.sh` is idempotent, refuses to touch a checkout with unexpected
local changes, and accepts `--reset` to re-apply. On Windows run it under Git
Bash or WSL. `requirements.txt` pins the versions used for the paper's runs.

### Arm B: CacheBlend (separate conda env)

CacheBlend vendors vLLM 0.4.1 on torch 2.2.1, which cannot coexist with the
KVCOMM environment. The setup builds it from source (20-40 min):

```bash
bash cacheblend/setup_cacheblend.sh          # clones into external/CacheBlend, applies the qwen2 patch,
                                             # verifies the blend kernel sha256, builds, pins the stack
conda activate cacheblend
export VLLM_ATTENTION_BACKEND=XFORMERS
python cacheblend/bench/check_env.py --model Qwen/Qwen2.5-Coder-7B-Instruct
```

Details, the two suffix configurations and why Llama-3.2-3B is excluded from
this arm: [`cacheblend/README.md`](cacheblend/README.md).

## Reproducing the experiments

All GPU runs go through `experiments/sweep.py`. Each cell (model x kv point x
repetition) runs in a fresh subprocess; interrupted sweeps resume from
`manifest.json`; `--dry-run` lists cells without executing.

**VIBE secure-code experiment (KVCOMM):**

```bash
python experiments/sweep.py --spec experiments/sweeps/paper_main.yaml --dry-run
python experiments/sweep.py --spec experiments/sweeps/paper_main.yaml            # gamma = 0.3, 3 models x 3 seeds
python experiments/sweep.py --spec experiments/sweeps/margin_threshold_run.yaml  # gamma in {0.1, 0.3, 0.5}
python experiments/sweep.py --spec experiments/sweeps/paper_main.yaml --retry-failed
```

Within one natural KVCOMM pass every task runs once and the engine decides
hit-or-miss organically. On a hit the validator's `kv_reuse` generation is
followed by a measurement-only `dense_prefill` of the same input, tagged
`"measure_only": true` in `Latency.json`; it cannot write to the anchor pool, so
it does not bias later tasks. Natural misses run dense once and are their own
baseline. The ten security rules are always exact-prefilled: only the
communicated code span is approximated.

A single manual run without the sweep:

```bash
python experiments/run_secure_code.py --mode kvcomm \
  --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --run_name my_run --output_dir ./runs/secure_code
python experiments/analyze_secure_code.py --output_dir ./runs/secure_code/my_run
```

**PrimeVul experiment (KVCOMM):** the vulnerable and patched arms run as
separate sweeps with fresh anchor pools and are joined offline by `pair_id`.

```bash
python datasets/primevul/build_dataset.py            # downloads PrimeVul (sha256-pinned), samples 100 pairs
python experiments/sweep.py --spec experiments/sweeps/primevul_vuln.yaml    --runner experiments/run_primevul.py
python experiments/sweep.py --spec experiments/sweeps/primevul_patched.yaml --runner experiments/run_primevul.py
```

**CacheBlend arm:** export the exact validator token sequences from the KVCOMM
side, then replay them.

```bash
# KVCOMM env (CPU only: tokenizer, no model load)
python experiments/export_frozen_inputs.py        results/<vibe_sweep>
python experiments/export_validator_prompt_spec.py
python experiments/export_primevul_prompt_spec.py --arm both
# CacheBlend env
bash cacheblend/run_sweep.sh --stage 1 --noise-floor            # r in {0.00, 0.16, 1.00}
bash cacheblend/run_primevul_sweep.sh --arm both --stage 1 --noise-floor
```

**Standalone dense replay** (plain Transformers, independent of KVCOMM):

```bash
python run_generated_codebase_validation.py dense   --codebase-root generatedCodebases
python run_generated_codebase_validation.py compare --help
```

## Offline analysis (no GPU)

Reports are regenerated from a results tree, so `results/` can be synced to
another machine first. Blend runs are written in KVCOMM's folder layout, so both
arms are scored by identical code.

```bash
python experiments/report.py                    results/<sweep>   # timing, throughput, 95% CIs, figures, LaTeX
python experiments/make_kv_view.py --all        results/<sweep>   # one view per kv point (never pools distinct gamma)
python experiments/paper_metrics.py             results/<sweep>   # change rate, agreement, transitions, silencing rate
python experiments/timing_metrics.py            results/<sweep>   # TTFT vs end-to-end speedup
python experiments/build_paper_report.py        --help            # VIBE: KVCOMM + CacheBlend cross-system report
python experiments/build_primevul_report.py     --help            # PrimeVul: ground-truth scoring
python experiments/compare_arms_cross_system.py --help
```

`docs/SWEEP_CHEATSHEET.md` walks through the whole cycle and explains each
report file.

## Tests

```bash
python -m pytest tests -q                                   # CPU: harness, metrics, prompt structure, chunking
KVCOMM_RUN_GPU_TESTS=1 python -m pytest tests -q            # adds the GPU integration tests
```

The suite needs `external/KVCOMM` for the tests that import the patched package;
without it those tests fail with a message pointing at the setup script.

## Datasets and provenance

- VIBE tasks: 90 prompts sampled with a fixed seed from MiniMaxAI/VIBE, balanced
  over 5 domains x 3 difficulties, plus 10 authentication tasks. Sources and
  sampling in `docs/TASK_SOURCES.md`; regenerate with
  `python datasets/secure_code/build_tasks.py`.
- PrimeVul: 100 vulnerable/patched pairs over ten CWEs. The 62 MB raw dump is
  not tracked; `datasets/primevul/manifest.json` pins its sha256 and
  `build_dataset.py` re-downloads and re-derives the same sample
  deterministically.
- The validator sees none of the label, arm, CWE target or pair metadata; see
  `docs/PROMPTS.md`.

## License and third-party notice

A license for this repository's original code will be added after the review
period. Upstream KVCOMM and CacheBlend remain under their authors' terms and are
not redistributed here.
