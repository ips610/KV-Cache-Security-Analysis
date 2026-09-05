# KVCOMM arm: patch and overlay

Upstream KVCOMM is not included in this repository (no license file upstream).
`setup_kvcomm.sh` clones it into `external/KVCOMM` at a pinned commit and applies
the harness's changes, which are the only KVCOMM-related code shipped here.

| Item | Value |
|---|---|
| Upstream | https://github.com/FastMAS/KVCOMM |
| Pinned commit | `48ca0b376c7f4fbf1c24042c1709a6fe4148c959` |
| `KVCOMM/` tree hash at that commit | `a4d838a28707815602f8e9fad0badd1b4c4032ae` |

```bash
bash kvcomm_patch/setup_kvcomm.sh            # clone + patch + overlay + import check
bash kvcomm_patch/setup_kvcomm.sh --reset    # re-apply after editing the patch/overlay
KVCOMM_ROOT=/some/checkout bash kvcomm_patch/setup_kvcomm.sh   # elsewhere
```

On Windows run it under Git Bash or WSL.

## `patches/0001-security-review-hooks.patch` (five upstream files)

| File | What the patch does |
|---|---|
| `KVCOMM/agents/__init__.py` | Registers the four harness agents (`CodeGenerator`, `CodeProvider`, `SecurityValidator`, `CweValidator`). |
| `KVCOMM/prompt/__init__.py` | Registers `SecureCodePromptSet` and `PrimeVulPromptSet`. |
| `KVCOMM/llm/llm.py` | Generation defaults used by every run: `DEFAULT_MAX_TOKENS` 512 -> 6000, `DEFAULT_TEMPERATURE` 1.0 -> 0.3 (runs use greedy decoding). |
| `KVCOMM/llm/gpt_chat.py` | Measurement-only dense pass on anchor hits (the paired dense reference that cannot write to the anchor pool), per-generation TTFT / total-latency records, `[ANCHOR CREATED]` / `[ANCHOR USED]` / `[GEN:<mode>]` observability lines, the forced-response "bare code" handoff used by the PrimeVul `CodeProvider`, and sampling controls read from `KVCOMM_SAMPLE_ROLES` / `KVCOMM_SAMPLE_TEMPERATURE`. |
| `KVCOMM/llm/kvcomm_engine.py` | `BareCodeKVPackage` (opaque KV handoff between provider and validator), anchor-pool write suppression for measurement-only passes, anchor provenance (which agent created / consumed an anchor), and the anchor logging hooks. |

The anchor matching, offset approximation and anchor prediction code paths of
the engine are not changed.

## `overlay/KVCOMM/` (eight new files)

| File | Purpose |
|---|---|
| `agents/code_generator.py` | Agent 1 of the VIBE experiment: dense-prefilled code generation. |
| `agents/security_validator.py` | Agent 2: ten-rule NIST-style review; the generated-code span is the only approximated region. |
| `agents/code_provider.py` | PrimeVul stage 1: promptless dense encoding of the dataset function. |
| `agents/cwe_validator.py` | PrimeVul stage 2: ten-CWE review. |
| `prompt/secure_code_common.py`, `prompt/secure_code_prompt_set.py` | Prompts, rule catalogue and prompt-structure guards for the VIBE arm. |
| `prompt/primevul_common.py`, `prompt/primevul_prompt_set.py` | The same for the PrimeVul arm. |

The exact prompt text lives in the four `prompt/*.py` overlay files.

## Regenerating the patch

The patch is a plain `git diff` of the five files between pristine upstream and
the harness's version; the overlay files are copied byte-for-byte. Keep both LF
(`.gitattributes` enforces it) and regenerate the patch with `git diff` rather
than editing it by hand.
