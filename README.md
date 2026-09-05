# Fast but Forgetful: Cross-Context KV Reuse Breaks Decision Fidelity in LLM Security Review

Experiment harness and analysis code for the paper of the same title. The
submission is under double-blind review: this repository carries no author
names, and the citation and de-anonymised links will be added after the review
period.

Nothing that the code can regenerate is checked in: no dataset files, no model
outputs, no result trees. This README gives the commands that produce all of
them. Folder-level notes live next to the code: `kvcomm_patch/README.md` (the
harness's modifications to KVCOMM and how upstream is fetched) and
`cacheblend/README.md` (the CacheBlend arm).

## 1. What the paper measures

Multi-agent LLM pipelines for software engineering pass the same program
between agents (a generator writes code, a reviewer audits it). Cross-context
KV-cache reuse avoids re-prefilling that shared text in the receiver's context
and is reported to cut time-to-first-token (TTFT) several-fold. The paper asks
whether the receiver's *decisions* survive the approximation.

The harness holds everything fixed except the receiver-side KV computation:
same model, same tokenizer, same code artifact, byte-identical prompt token
sequence, greedy decoding. It then compares

- **R**, the report produced under approximate KV reuse, against
- **D**, the report produced by exact dense prefill of the identical input,

and scores rule-level agreement, the direction of disagreements, and whether
findings that dense inference produced disappear under reuse ("silencing").

Three experimental arms:

| Arm | Reuse mechanism | Code under review | Reference | Configurations |
|---|---|---|---|---|
| VIBE (primary) | KVCOMM anchor-based offset approximation | 90 application tasks from MiniMaxAI/VIBE, code written by Agent 1 (`CodeGenerator`), reviewed by Agent 2 (`SecurityValidator`) against ten NIST SP 800-63B-style rules | measurement-only dense pass of the same validation on every anchor hit | entropy threshold gamma in {0.1, 0.3, 0.5}, 20 anchors, window 5 |
| CacheBlend (replication) | chunked prefill with selective recomputation | the same validator token sequences, replayed | full dense prefill on the same stack, plus a dense-vs-dense noise floor | recompute ratio r in {0.00, 0.16, 1.00}, two suffix configurations |
| PrimeVul (labelled) | both of the above | 100 vulnerable / 100 patched C functions, reviewed by `CweValidator` against ten CWE rules | as above, joined offline by pair id | as above |

Models: Qwen2.5-3B-Instruct, Qwen2.5-Coder-7B-Instruct, Llama-3.2-3B-Instruct
(the CacheBlend arm uses the two Qwen models only; see `cacheblend/README.md`
for why). The exact agent prompts are in the four
`kvcomm_patch/overlay/KVCOMM/prompt/*.py` files.

## 2. Requirements

**Hardware.** All GPU experiments were run on a single NVIDIA H100 80 GB. Any
CUDA GPU with enough memory for a 7B model in bf16 plus a 6000-token generation
works; a 24 GB card suffices for the two 3B models. The offline analysis needs
no GPU.

**Software.** Python 3.10 for the KVCOMM arm (`requirements.txt` pins the
versions the paper used: torch 2.1.0, transformers 4.50.2). The CacheBlend arm
needs a separate conda environment because its vLLM 0.4.1 fork pins torch
2.2.1; `cacheblend/setup_cacheblend.sh` builds it. Git and Bash (Git Bash or
WSL on Windows) are required by the setup scripts.

**Models.** Weights are downloaded from the Hugging Face Hub on first use.
`meta-llama/Llama-3.2-3B-Instruct` is gated and needs an accepted licence and
`huggingface-cli login`; the two Qwen models are ungated.

**Third-party code that must be present.** Neither system under study is
distributed here (KVCOMM's repository carries no licence file; CacheBlend is a
vLLM fork with compiled kernels). The setup scripts fetch them at pinned
commits into `external/`, which is gitignored and created on demand:

| Expected path | Project | Created by |
|---|---|---|
| `external/KVCOMM` | KVCOMM should be present here: [FastMAS/KVCOMM](https://github.com/FastMAS/KVCOMM) at commit `48ca0b37`, with `kvcomm_patch/` applied | `bash kvcomm_patch/setup_kvcomm.sh` |
| `external/CacheBlend` | CacheBlend should be present here: [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) at commit `55ad0267`, with `cacheblend/patches/` applied, built in its own conda env | `bash cacheblend/setup_cacheblend.sh` |

Set `KVCOMM_ROOT` or `CACHEBLEND_ROOT` to use checkouts that live elsewhere;
`repo_paths.py` reads the same variables when the Python entry points start.

## 3. Setup

### 3.1 KVCOMM arm and analysis environment

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n kvfid python=3.10
pip install -r requirements.txt
bash kvcomm_patch/setup_kvcomm.sh                      # clone, pin, patch, overlay, import check
```

`setup_kvcomm.sh` is idempotent, verifies the checkout by tree hash and
sha256, refuses to touch a checkout with unexpected local changes, and accepts
`--reset` to re-apply. On Windows, clone into a short path or set
`git config core.longpaths true`.

### 3.2 CacheBlend arm

```bash
bash cacheblend/setup_cacheblend.sh          # clone at pinned commit, apply qwen2 patch, verify the
                                             # blend kernel sha256, build vLLM (20-40 min), pin the stack
conda activate cacheblend
export VLLM_ATTENTION_BACKEND=XFORMERS
python cacheblend/bench/check_env.py --model Qwen/Qwen2.5-Coder-7B-Instruct
```

## 4. Build the data

The two corpora are deterministic samples of public datasets:

```bash
python datasets/secure_code/build_tasks.py     # 90 VIBE tasks -> datasets/secure_code/coding_tasks.jsonl
python datasets/primevul/build_dataset.py      # 100 PrimeVul pairs -> datasets/primevul/{primevul_pairs,tasks_vulnerable,tasks_patched}.jsonl
```

- **VIBE tasks**: 90 prompts sampled with a fixed seed from
  [MiniMaxAI/VIBE](https://huggingface.co/datasets/MiniMaxAI/VIBE) (MIT),
  6 per cell of the 5 domains x 3 difficulties grid. Byte-identical on every
  rerun.
- **PrimeVul pairs**: 100 vulnerable/patched pairs over ten CWEs from the
  [`colin/PrimeVul`](https://huggingface.co/datasets/colin/PrimeVul) mirror
  (MIT) of PrimeVul (Ding et al., 2024). The 62 MB source splits are
  downloaded and sha256-checked; selection is deterministic (length cap,
  dedup by function hash, ten CWEs, 10 pairs per CWE, at most 3 per project).
  No answer-revealing field reaches the validator prompt.

## 5. Run the experiments

There is no training: every experiment is inference under two KV-computation
regimes. All GPU runs go through `experiments/sweep.py`, which expands a YAML
spec into cells (model x kv point x repetition), runs each cell in a fresh
subprocess so no anchor state or CUDA context leaks between configurations,
records progress in `results/<sweep>/manifest.json` so an interrupted sweep
resumes, and runs the offline analysis when the last cell finishes.

**Step 1: VIBE arm under KVCOMM** (the efficiency and fidelity tables, transition
matrices, and the generated code artifacts that every later step reuses):

```bash
python experiments/sweep.py --spec experiments/sweeps/margin_threshold_run.yaml --dry-run   # list the 9 cells
python experiments/sweep.py --spec experiments/sweeps/margin_threshold_run.yaml             # gamma in {0.1, 0.3, 0.5}
python experiments/sweep.py --spec experiments/sweeps/final_run.yaml                        # gamma 0.3, 3 seeds, dense-vs-dense noise floor
```

Within one natural KVCOMM pass every task runs once and the engine decides
hit-or-miss organically. On a hit the validator's `kv_reuse` generation is
followed by a measurement-only `dense_prefill` of the same input, tagged
`"measure_only": true` in `Latency.json`; it cannot write to the anchor pool, so
it does not bias later tasks. Natural misses run dense once and are their own
baseline. The security rules are always exact-prefilled: only the communicated
code span is approximated. Budget on one H100: several hours per 3B cell, up to
a day per Coder-7B cell.

**Step 2: PrimeVul arm under KVCOMM** (vulnerable and patched versions are
separate sweeps with fresh anchor pools, joined offline by pair id):

```bash
python experiments/sweep.py --spec experiments/sweeps/primevul_vuln.yaml    --runner experiments/run_primevul.py --no-analysis
python experiments/sweep.py --spec experiments/sweeps/primevul_patched.yaml --runner experiments/run_primevul.py --no-analysis
python experiments/build_primevul_report.py
```

**Step 3: export the validator's exact token sequences** from the finished
KVCOMM runs. The CacheBlend arm and the stand-alone dense cross-check replay
these, so both arms judge byte-identical prompts:

```bash
python experiments/export_frozen_inputs.py --sweep-root results/MarginThresholdRun --canonical-kv kv_t0.3_a20_w5
python experiments/export_validator_prompt_spec.py                 # -> datasets/secure_code/validator_prompt_spec/
python experiments/export_primevul_prompt_spec.py --arm both       # -> datasets/primevul/validator_prompt_spec/
python experiments/corpus_health.py                                # truncated / drifted / not-token-exact accounting
```

**Step 4: CacheBlend arm** (conda env `cacheblend`):

```bash
conda activate cacheblend && export VLLM_ATTENTION_BACKEND=XFORMERS
bash cacheblend/run_sweep.sh --stage 1 --noise-floor                 # r in {0.00, 0.16, 1.00}, config B
bash cacheblend/run_primevul_sweep.sh --arm both --stage 1 --noise-floor
```

**Step 5: offline analysis** (any machine, no GPU, over a finished results tree):

```bash
python experiments/make_kv_view.py   results/MarginThresholdRun --all   # one view per gamma; never pool distinct gammas
python experiments/paper_metrics.py  results/MarginThresholdRun__view_kv_t0.3_a20_w5   # change rate, agreement, transitions, silencing
python experiments/timing_metrics.py results/MarginThresholdRun                        # TTFT speedup vs end-to-end speedup
python experiments/report.py         results/MarginThresholdRun                        # throughput, 95% CIs, figures, LaTeX
python experiments/methodology.py    results/MarginThresholdRun                        # METHODOLOGY.md harvested from env.json
python experiments/build_paper_report.py --kvcomm-root results/MarginThresholdRun --cacheblend-root results/CacheBlendRun
python experiments/primevul_metrics.py --vuln-root results/PrimeVulVulnRun --patched-root results/PrimeVulPatchedRun
```

Every script prints its inputs and outputs with `--help`. The results tree
layout is `results/<sweep>/<model_slug>/<mode>/kv_t<gamma>_a<anchors>_w<window>/rep_<n>/`
with `Latency.json`, `metrics.json`, `env.json`, `meta.json` and one folder per
task holding the task, the code under review, the natural report and the
dense-baseline report. `report/` under each sweep holds the tables (Markdown,
CSV, LaTeX) and figures (PDF).

**Optional: independent dense cross-check** with plain Transformers, outside
the KVCOMM engine, over the Agent 1 outputs collected from a sweep:

```bash
python run_generated_codebase_validation.py dense   --codebase-root <folder of task folders> --prompt-spec-dir datasets/secure_code/validator_prompt_spec
python run_generated_codebase_validation.py compare --kvcomm-results results/MarginThresholdRun
```

## 6. Pre-trained models

No model is trained or fine-tuned. The three open-weight instruction models
are used as released:

| Hugging Face id | Role in the paper | Access |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | all arms | open |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | all arms | open |
| `meta-llama/Llama-3.2-3B-Instruct` | KVCOMM and PrimeVul arms | gated (accept the licence on the Hub) |

## 7. Results

Headline numbers reported in the paper (pooled over the three models unless
stated). All are produced by the steps above.

| Quantity | Value |
|---|---|
| Validator calls admitted to KVCOMM reuse (533 paired gate-hit executions) | 65.8 % |
| Mean TTFT speedup on those calls | 2.52x |
| End-to-end speedup on the same calls | 1.02x to 1.10x |
| Security reports changed by approximate execution | 40.6 % to 85.7 % per model and gamma |
| Dense FAIL findings silenced under reuse | 21.7 % to 89.5 % (53.4 % pooled at gamma = 0.3) |
| New findings created under reuse | at most 5.3 % |
| Dominant transition | FAIL to NOT APPLICABLE |
| CacheBlend, r = 0.16, findings silenced | 38.8 % pooled, falling to 0.0 % at r = 1.00 |
| PrimeVul, true-positive detections removed by reuse | 41.1 % (KVCOMM) and 37.9 % (CacheBlend) |

Exact numeric equality needs the same stack the paper used (H100, pinned
torch and transformers). Greedy decoding plus the determinism flags in the
runners make runs reproducible on that setup; a different GPU or library
version can flip individual tokens in bf16, which the dense-vs-dense noise
floor in `final_run.yaml` is there to quantify.

## 8. Is the code self-contained and executable?

The analysis, the dataset builders and the exporters run on CPU from a fresh
clone. The GPU experiments are executable but not fully self-contained, for
three reasons the guidelines ask us to state:

1. **Specialised hardware.** A CUDA GPU is required; the reported runs used an
   H100 80 GB.
2. **Third-party code we cannot redistribute.** The KVCOMM reference
   implementation carries no licence file, so it is fetched from its public
   repository at a pinned commit and this harness's modifications to it are
   shipped as a patch and an overlay (`kvcomm_patch/`). CacheBlend is a vLLM
   fork with compiled CUDA kernels; it is likewise fetched and built by a
   script. Neither contains the scientific contribution of the paper, which is
   the measurement harness in this repository.
3. **Gated weights and external datasets.** Llama-3.2-3B requires accepting
   Meta's licence on the Hugging Face Hub. The corpora are rebuilt from their
   public sources by the scripts in section 4.

## 9. Repository layout

| Path | Purpose |
|---|---|
| `experiments/` | Runners (`run_secure_code.py`, `run_primevul.py`), sweep orchestration (`sweep.py`, `harness/`, `sweeps/*.yaml`), exporters, analysis and reporting |
| `kvcomm_patch/` | Patch for five upstream KVCOMM files, eight new agent and prompt-set files, and the setup script. The only KVCOMM-related code shipped |
| `cacheblend/` | CacheBlend arm: setup and pinning scripts, the `qwen2.py` plumbing patch, the blend runner and sweep scripts |
| `datasets/` | Dataset builders and loaders; the corpora themselves are rebuilt, not shipped |
| `run_generated_codebase_validation.py` | Stand-alone dense replay of the validator with plain Transformers, independent of KVCOMM |
| `repo_paths.py` | Path bootstrap shared by every entry point; reads `KVCOMM_ROOT` / `CACHEBLEND_ROOT` |
| `external/` | Created by the setup scripts for the two third-party checkouts (gitignored) |

`results/`, `runs/`, `logs/` and every built or exported data file are
gitignored.

## 10. Licence

The code in this repository is released under the MIT License (`LICENSE`).
Upstream KVCOMM and CacheBlend remain under their authors' terms and are not
redistributed here. MiniMaxAI/VIBE and the `colin/PrimeVul` mirror are MIT;
Qwen2.5 weights are Apache-2.0; Llama 3.2 weights are under the Llama 3.2
Community License.
