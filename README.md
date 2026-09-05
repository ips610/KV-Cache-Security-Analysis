# KV-Cache Security Analysis

Harness for "Fast but Forgetful: Cross-Context KV Reuse Breaks Decision Fidelity in LLM Security Review".
Nothing regenerable is checked in; the commands below build the data, run the experiments and produce the tables.

Requirements: Python 3.10, CUDA GPU (runs used one H100 80 GB), Git + Bash (Git Bash or WSL on Windows).
Upstream KVCOMM and CacheBlend are not included; the setup scripts fetch them at pinned commits into `external/` (gitignored).
`meta-llama/Llama-3.2-3B-Instruct` is gated: run `huggingface-cli login` after accepting its licence.

## 1. Setup (KVCOMM arm + analysis)

```bash
pip install -r requirements.txt
bash kvcomm_patch/setup_kvcomm.sh          # clones FastMAS/KVCOMM @ 48ca0b37 into external/KVCOMM, applies kvcomm_patch/
```

## 2. Build the data

```bash
python datasets/secure_code/build_tasks.py     # 90 VIBE tasks      -> datasets/secure_code/coding_tasks.jsonl
python datasets/primevul/build_dataset.py      # 100 PrimeVul pairs -> datasets/primevul/*.jsonl
```

## 3. Run KVCOMM experiments

```bash
python experiments/sweep.py --spec experiments/sweeps/margin_threshold_run.yaml --dry-run
python experiments/sweep.py --spec experiments/sweeps/margin_threshold_run.yaml             # VIBE, gamma in {0.1,0.3,0.5}
python experiments/sweep.py --spec experiments/sweeps/final_run.yaml                        # VIBE, gamma 0.3, 3 seeds, noise floor
python experiments/sweep.py --spec experiments/sweeps/primevul_vuln.yaml    --runner experiments/run_primevul.py --no-analysis
python experiments/sweep.py --spec experiments/sweeps/primevul_patched.yaml --runner experiments/run_primevul.py --no-analysis
```

Interrupted sweeps resume with the same command. Results land in `results/<sweep>/`.

## 4. Export validator token sequences (needed by steps 5 and 7)

```bash
python experiments/export_frozen_inputs.py --sweep-root results/MarginThresholdRun --canonical-kv kv_t0.3_a20_w5
python experiments/export_validator_prompt_spec.py
python experiments/export_primevul_prompt_spec.py --arm both
```

## 5. CacheBlend arm (separate conda env)

```bash
bash cacheblend/setup_cacheblend.sh        # clones YaoJiayi/CacheBlend @ 55ad0267 into external/CacheBlend, builds vLLM (20-40 min)
conda activate cacheblend
export VLLM_ATTENTION_BACKEND=XFORMERS
python cacheblend/bench/check_env.py --model Qwen/Qwen2.5-Coder-7B-Instruct
bash cacheblend/run_sweep.sh --stage 1 --noise-floor
bash cacheblend/run_primevul_sweep.sh --arm both --stage 1 --noise-floor
```

## 6. Analysis (no GPU)

```bash
python experiments/make_kv_view.py   results/MarginThresholdRun --all
python experiments/paper_metrics.py  results/MarginThresholdRun__view_kv_t0.3_a20_w5      # fidelity, transitions, silencing
python experiments/timing_metrics.py results/MarginThresholdRun                           # TTFT vs end-to-end speedup
python experiments/report.py         results/MarginThresholdRun                           # tables, CIs, figures, LaTeX
python experiments/methodology.py    results/MarginThresholdRun
python experiments/build_paper_report.py   --kvcomm-root results/MarginThresholdRun --cacheblend-root results/CacheBlendRun
python experiments/build_primevul_report.py
```

Outputs: `results/<sweep>/report/` (Markdown, CSV, LaTeX, PDF figures).

## 7. Optional: dense cross-check with plain Transformers

```bash
python run_generated_codebase_validation.py dense   --codebase-root <folder of task folders> --prompt-spec-dir datasets/secure_code/validator_prompt_spec
python run_generated_codebase_validation.py compare --kvcomm-results results/MarginThresholdRun
```

Every script documents its options with `--help`. See `kvcomm_patch/README.md` and `cacheblend/README.md` for what the patches change.
License: MIT (`LICENSE`); upstream KVCOMM and CacheBlend under their own terms.
