# Fast but Forgetful: Cross-Context KV Reuse Breaks Decision Fidelity in LLM Security Review

Experiment harness, datasets and analysis code for the paper of the same title.
The submission is under double-blind review: this repository carries no author
names, and the citation and de-anonymised links will be added after the review
period.

**Documentation index**

| Document | Contents |
|---|---|
| this README | What the paper measures, requirements, setup, how to run every experiment, results, what is and is not self-contained |
| [`docs/CODE_DOCUMENTATION.md`](docs/CODE_DOCUMENTATION.md) | Detailed code reference: pipeline, every module, configuration format, output tree and file schemas, metric definitions, environment variables |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Step-by-step commands that regenerate each table and figure of the paper, with hardware and time budgets |
| [`docs/DATA.md`](docs/DATA.md) | Data card: every shipped dataset and artifact, its fields, provenance, licence and rebuild procedure |
| [`paper_results/README.md`](paper_results/README.md) | Summaries of the runs the paper reports, tree by tree |
| [`docs/PROMPTS.md`](docs/PROMPTS.md) | Every system and user prompt used by the agents, verbatim, and what is held fixed between arms |
| [`docs/SWEEP_CHEATSHEET.md`](docs/SWEEP_CHEATSHEET.md) | Operational runbook for the GPU host |
| [`docs/TASK_SOURCES.md`](docs/TASK_SOURCES.md) | Survey of the task corpora considered and why MiniMaxAI/VIBE was chosen |
| [`docs/PRIMEVUL_EXPERIMENT_PLAN.md`](docs/PRIMEVUL_EXPERIMENT_PLAN.md) | Design document for the externally-labelled PrimeVul arm |
| [`docs/methodology.tex`](docs/methodology.tex) | The paper's methodology section |
| [`kvcomm_patch/README.md`](kvcomm_patch/README.md), [`cacheblend/README.md`](cacheblend/README.md), [`external/README.md`](external/README.md) | The two systems under study: what is patched, how they are fetched, why they are not vendored |

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
for why).

## 2. Requirements

**Hardware.** All GPU experiments were run on a single NVIDIA H100 80 GB. Any
CUDA GPU with enough memory for a 7B model in bf16 plus a 6000-token generation
works; a 24 GB card suffices for the two 3B models. The offline analysis and the
CPU test suite need no GPU.

**Software.** Python 3.10 for the KVCOMM arm (`requirements.txt` pins the
versions the paper used: torch 2.1.0, transformers 4.50.2). The CacheBlend arm
needs a separate conda environment because its vLLM 0.4.1 fork pins torch
2.2.1; `cacheblend/setup_cacheblend.sh` builds it. Git and Bash (Git Bash or
WSL on Windows) are required by the setup scripts.

**Models.** Weights are downloaded from the Hugging Face Hub on first use.
`meta-llama/Llama-3.2-3B-Instruct` is gated and needs an accepted licence and
`huggingface-cli login`; the two Qwen models are ungated.

**Third-party code that must be present** (fetched by scripts, not vendored;
see section 7):

| Expected path | Project | Created by |
|---|---|---|
| `external/KVCOMM` | KVCOMM should be present here: [FastMAS/KVCOMM](https://github.com/FastMAS/KVCOMM) at commit `48ca0b37`, with `kvcomm_patch/` applied | `bash kvcomm_patch/setup_kvcomm.sh` |
| `external/CacheBlend` | CacheBlend should be present here: [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) at commit `55ad0267`, with `cacheblend/patches/` applied, built in its own conda env | `bash cacheblend/setup_cacheblend.sh` |

## 3. Setup

### 3.1 KVCOMM arm and analysis environment

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n kvfid python=3.10
pip install -r requirements.txt
bash kvcomm_patch/setup_kvcomm.sh                      # clone, pin, patch, overlay, import check
python -m pytest tests -q                              # CPU test suite (about 30 s)
```

`setup_kvcomm.sh` is idempotent, verifies the checkout by tree hash and
sha256, refuses to touch a checkout with unexpected local changes, and accepts
`--reset` to re-apply. Point `KVCOMM_ROOT` at an existing checkout to install
elsewhere. Expected test result: all tests pass, with the GPU tests skipped.
On Windows, clone into a short path or set `git config core.longpaths true`;
the deepest path in the repository is about 130 characters.

### 3.2 CacheBlend arm

```bash
bash cacheblend/setup_cacheblend.sh          # clone at pinned commit, apply qwen2 patch, verify the
                                             # blend kernel sha256, build vLLM (20-40 min), pin the stack
conda activate cacheblend
export VLLM_ATTENTION_BACKEND=XFORMERS
python cacheblend/bench/check_env.py --model Qwen/Qwen2.5-Coder-7B-Instruct
python cacheblend/bench/blend_runner.py --self-test --model Qwen/Qwen2.5-3B-Instruct \
    --spec datasets/secure_code/validator_prompt_spec/Qwen_Qwen2.5-3B-Instruct_prompt_spec.json
```

## 4. Running the experiments

There is no training: every experiment is inference under two KV-computation
regimes. All GPU runs go through `experiments/sweep.py`, which expands a YAML
spec into cells (model x kv point x repetition), runs each cell in a fresh
subprocess so no anchor state or CUDA context leaks between configurations,
records progress in `results/<sweep>/manifest.json` so an interrupted sweep
resumes, and runs the offline analysis when the last cell finishes.

```bash
# VIBE arm, gamma = 0.3, 3 models x 3 repetitions, with the dense-vs-dense noise floor
python experiments/sweep.py --spec experiments/sweeps/final_run.yaml --dry-run    # list the 9 cells
python experiments/sweep.py --spec experiments/sweeps/final_run.yaml

# VIBE arm, threshold sweep gamma in {0.1, 0.3, 0.5} (Tables on hit rate, speedup and fidelity per gamma)
python experiments/sweep.py --spec experiments/sweeps/margin_threshold_run.yaml

# PrimeVul arm: vulnerable and patched versions are separate sweeps with fresh anchor pools
python experiments/sweep.py --spec experiments/sweeps/primevul_vuln.yaml    --runner experiments/run_primevul.py --no-analysis
python experiments/sweep.py --spec experiments/sweeps/primevul_patched.yaml --runner experiments/run_primevul.py --no-analysis
python experiments/build_primevul_report.py

# CacheBlend arm (conda env "cacheblend"); prompt specs are shipped, so this needs no KVCOMM run first
bash cacheblend/run_sweep.sh --stage 1 --noise-floor                 # r in {0.00, 0.16, 1.00}, config B
bash cacheblend/run_primevul_sweep.sh --arm both --stage 1 --noise-floor
```

Every analysis step also runs stand-alone on a finished results tree and needs
no GPU, so `results/` can be copied to another machine first:

```bash
python experiments/report.py         results/FinalRun     # timing, throughput, 95% CIs, figures, LaTeX
python experiments/paper_metrics.py  results/FinalRun     # change rate, agreement, transitions, silencing rate
python experiments/timing_metrics.py results/FinalRun     # TTFT speedup vs end-to-end speedup
python experiments/methodology.py    results/FinalRun     # METHODOLOGY.md harvested from the run's env.json files
python experiments/build_paper_report.py --kvcomm-root results/MarginThresholdRun --cacheblend-root results/CacheBlendRun
```

`docs/REPRODUCTION.md` maps each paper table and figure to the exact commands,
and `docs/CODE_DOCUMENTATION.md` describes what every script does and writes.

## 5. Pre-trained models

No model is trained or fine-tuned. The three open-weight instruction models
are used as released:

| Hugging Face id | Role in the paper | Access |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | all arms | open |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | all arms | open |
| `meta-llama/Llama-3.2-3B-Instruct` | KVCOMM and PrimeVul arms | gated (accept the licence on the Hub) |

The agents' generated code for the 90 VIBE tasks, per model, is shipped in
`generatedCodebases/` and as frozen validator inputs under
`datasets/secure_code/frozen_validator_inputs/`, so the review stage can be
reproduced without re-running the generator.

## 6. Results

Headline numbers reported in the paper (pooled over the three models unless
stated). All are produced by the analysis scripts above from the raw run trees.

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

Summaries of the recorded runs behind these numbers are shipped in
`paper_results/` (about 20 MB: per-cell timing records, metrics, environment
snapshots, and the generated report tables and figures for every run the paper
cites; see `paper_results/README.md`). The per-task validator reports are
regenerated by the sweep commands in `docs/REPRODUCTION.md`.

## 7. Is the code self-contained and executable?

The analysis, the dataset builders and the test suite are self-contained and
run on CPU from a fresh clone. The GPU experiments are executable but not fully
self-contained, for three reasons the guidelines ask us to state:

1. **Specialised hardware.** A CUDA GPU is required; the reported runs used an
   H100 80 GB and a full VIBE cell on the 7B model takes on the order of a day
   because every task generates up to 6000 tokens twice.
2. **Third-party code we cannot redistribute.** The KVCOMM reference
   implementation, whose behaviour the paper studies, carries no licence file,
   so it is fetched from its public repository at a pinned commit and this
   harness's modifications to it are shipped as a patch and an overlay
   (`kvcomm_patch/`). CacheBlend is a vLLM fork with compiled CUDA kernels; it
   is likewise fetched and built by a script. Neither contains the scientific
   contribution of the paper, which is the measurement harness in this
   repository.
3. **Gated weights and external datasets.** Llama-3.2-3B requires accepting
   Meta's licence on the Hugging Face Hub. The PrimeVul raw dump (62 MB) is
   re-downloaded by `datasets/primevul/build_dataset.py` and is sha256-pinned in
   `datasets/primevul/manifest.json`; the derived 100-pair sample is shipped.

## 8. Repository layout

| Path | Purpose |
|---|---|
| `experiments/` | Runners (`run_secure_code.py`, `run_primevul.py`), sweep orchestration (`sweep.py`, `harness/`, `sweeps/*.yaml`), export of frozen inputs and prompt specs, analysis and reporting |
| `kvcomm_patch/` | Patch for five upstream KVCOMM files, eight new agent and prompt-set files, and the setup script. The only KVCOMM-related code shipped |
| `cacheblend/` | CacheBlend arm: setup and pinning scripts, the `qwen2.py` plumbing patch, the blend runner and sweep scripts |
| `datasets/` | Task corpora, builders and loaders (`secure_code/`, `primevul/`), frozen validator inputs and exported prompt token specs |
| `generatedCodebases/` | Agent 1 outputs for the 90 VIBE tasks, per model |
| `paper_results/` | Summaries of the runs the paper reports: timing records, metrics, environment snapshots and report tables/figures per run |
| `tests/` | CPU test suite; GPU integration tests are opt-in via `KVCOMM_RUN_GPU_TESTS=1` |
| `docs/` | The documents listed at the top of this file |
| `run_generated_codebase_validation.py` | Stand-alone dense replay of the validator with plain Transformers, independent of KVCOMM, used as a cross-check |
| `repo_paths.py`, `conftest.py`, `pytest.ini` | Path bootstrap shared by every entry point and the test configuration |
| `external/` | Placeholder for the two third-party checkouts (contents gitignored) |

`results/`, `runs/`, `logs/` and `docs/experiment_reports/` are output
locations and are gitignored.

## 9. Licence

The code in this repository is released under the MIT License (`LICENSE`).
Upstream KVCOMM and CacheBlend remain under their authors' terms and are not
redistributed here. Dataset licences are listed in `docs/DATA.md`.
