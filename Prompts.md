# Prompts and agents used in the paper experiments

This document covers only the experiments reported in [`paper/paper.tex`](paper/paper.tex). It does not describe the repository's older MMLU, GSM8K, HumanEval, COPY, or general-purpose KVCOMM prompts.

The paper evaluates one security-review pipeline in three settings:

1. The primary behavioural-fidelity experiment on 90 MINIMAXAI/VIBE application-development tasks.
2. A CacheBlend replication using the same validator inputs.
3. A labelled PrimeVul experiment using 100 vulnerable functions and their 100 patched counterparts.

## Experimental pipeline

The primary experiment uses two agents:

| Stage | Agent | Purpose | Inference |
|---|---|---|---|
| 1 | Code Generator (`CodeGenerator`) | Generate an implementation from a VIBE task | Exact dense prefill |
| 2 | Security Validator (`SecurityValidator`) | Review the generated code against ten fixed security rules | KVCOMM natural execution, paired dense reference on gate hits |

For every paired comparison, the Security Validator receives the same model, system instruction, task, code, rule set, report request, and greedy decoding configuration. The only intervention is how the validator-side KV states are computed.

## Primary VIBE experiment

### Agent 1: Code Generator

#### When it is used

The Code Generator is used only for the 90 unlabelled application-development tasks sampled from MINIMAXAI/VIBE. It always runs densely so that the Security Validator compares the same generated artifact in the approximate and exact executions.

#### System prompt

~~~text
You are a senior backend engineer. Write secure, production-ready code for the requested task. Do not include explanations.
~~~

#### User prompt

~~~text
The task is: <VIBE_TASK>
~~~

`<VIBE_TASK>` is replaced with one complete application-development request from `datasets/secure_code/coding_tasks.jsonl`.

#### Exact example

Using the first task in the paper's VIBE corpus:

~~~text
SYSTEM
You are a senior backend engineer. Write secure, production-ready code for the requested task. Do not include explanations.

USER
The task is: Reimagine our boutique hotelsâ€™ booking experience so it feels premium and trustworthy. Guests should browse immersive, highâ€‘resolution photo galleries, check live room availability on a calendar, pick a room, and complete a smooth, stepâ€‘byâ€‘step reservation that ends with a secure Stripe checkout. Use Next.js with serverâ€‘side rendering for fast first loads and strong SEO, and use React Query so availability and pricing stay fresh without blocking the UI. The design should be corporateâ€‘professional with an elegant, highâ€‘end feel, fully responsive and accessible. Include clear validation and error states (e.g., soldâ€‘out nights, minimumâ€‘stay rules) and provide a polished confirmation screen after payment.
~~~

The generated response, whether complete or truncated by the generation limit, becomes the fixed code artifact reviewed by both executions of Agent 2.

### Agent 2: Security Validator

#### When it is used

The Security Validator reviews every generated VIBE artifact. It returns one verdict for each of ten rules and then gives additional recommendations and general code-review comments.

#### System prompt

~~~text
You are a security validator and code reviewer. You are given code written by a developer agent and a set of NIST-derived security rules. Review the code strictly against each rule and produce a report in EXACTLY the following format, with no other sections:

### Security Validation Report

#### Rule <n>: <full rule text>
**PASS**, **FAIL**, or **NOT APPLICABLE**
- <evidence from the code justifying the verdict>

(Repeat the '#### Rule' block above for every rule, in order. The line directly under each rule heading must contain only the single verdict **PASS**, **FAIL**, or **NOT APPLICABLE** in bold. Use **NOT APPLICABLE** only when the rule's subject matter â€” e.g. passwords, sessions, database queries â€” does not exist anywhere in the task or the code.)

### Additional Recommendations
- <security improvements beyond the listed rules>

### Code Review Comments
- <general code-quality observations (structure, readability, error handling)>
~~~

#### User prompt

~~~text
The task is: <VIBE_TASK>

A developer agent produced the following code:

<GENERATED_CODE>

Validate the code above against the following security rules (NIST SP 800-63B style):
1. Passwords must be hashed with a memory-hard algorithm (bcrypt, scrypt, or argon2); never store or compare plaintext passwords.
2. Secret comparisons (tokens, MACs, passwords) must be constant-time.
3. Session tokens must be generated with a cryptographically secure RNG (>=128 bits).
4. Never log secrets, passwords, or full tokens.
5. Enforce account lockout or rate limiting on repeated failed authentication attempts.
6. No hardcoded secrets, API keys, or credentials in source code; load them from configuration or environment.
7. User-supplied input used in database queries must be parameterized; never build SQL by string concatenation or interpolation.
8. Verification, password-reset, and magic-link tokens must be single-use and expire.
9. Authentication failure messages must be generic; do not reveal whether a username or email exists.
10. Session cookies must set the Secure and HttpOnly flags, and credentials must only be transmitted over TLS.

Produce a security validation report citing each rule as PASS, FAIL, or NOT APPLICABLE with reasons.
~~~

`<VIBE_TASK>` is the original task given to Agent 1. `<GENERATED_CODE>` is Agent 1's fixed response.

#### Required output shape

The report must contain all ten rule blocks in order. A valid abbreviated example is:

~~~text
### Security Validation Report

#### Rule 1: Passwords must be hashed with a memory-hard algorithm (bcrypt, scrypt, or argon2); never store or compare plaintext passwords.
**PASS**
- Passwords are hashed with Argon2 before persistence.

#### Rule 2: Secret comparisons (tokens, MACs, passwords) must be constant-time.
**FAIL**
- The reset token is compared with the normal equality operator.

...one block for every remaining rule, in order...

### Additional Recommendations
- Add security-focused integration tests.

### Code Review Comments
- Separate authentication logic from route handlers.
~~~

The example is abbreviated only to demonstrate formatting. Actual experiment reports must include Rules 1 through 10.

### Meaning of NOT APPLICABLE

The paper treats `NOT APPLICABLE` as a narrow applicability verdict, not as uncertainty or abstention. It is allowed only when the rule's subject matter—such as passwords, sessions, or database queries—is absent from both the task and the code.

## Execution cases for the same VIBE prompt

The text of the Security Validator prompt does not change across these cases:

| Case | What happens | Included in paired fidelity analysis? |
|---|---|---|
| KVCOMM gate hit | Agent 2 uses approximate cross-context KV reuse for the communicated code | Yes; followed by an identical-input dense reference |
| KVCOMM gate miss | Agent 2 performs ordinary dense prefill | No; no approximation occurred |
| Measurement-only dense reference | Agent 2 densely processes the same complete input after a gate hit | Yes; this is reference `D` and cannot update the anchor pool |
| CacheBlend approximate run | The same validator token sequence is replayed with selective recomputation | Yes; paired with dense execution on the same CacheBlend stack |
| CacheBlend all-recompute control | The same token sequence is run with `r=1.0` | Control condition |

The approximate result is denoted `R`; the exact dense result is denoted `D`. A pair differs only in receiver-side KV computation.

## CacheBlend replication

CacheBlend does not introduce a new agent role or a new natural-language prompt. It replays the same Security Validator system and user prompts shown above for the two supported Qwen models.

The validator prompt is divided into these semantic regions:

| Region | Content |
|---|---|
| A | Chat/system prefix and the beginning of the user request |
| T | Original VIBE task |
| B | “A developer agent produced the following code” connector |
| C | Generated code |
| D | Ten-rule security policy and final report request |

CacheBlend evaluates recomputation ratios `r ∈ {0.0, 0.16, 1.0}`. These values alter KV recomputation, not the prompt wording. The replication is limited to Qwen2.5-3B-Instruct and Qwen2.5-Coder-7B-Instruct; it does not use Llama-3.2-3B-Instruct.

## PrimeVul labelled experiment

PrimeVul supplies existing source functions instead of application-generation prompts. The paper therefore uses a different first stage:

| Stage | Agent | Purpose | Model-facing prompt |
|---|---|---|---|
| 1 | Code Provider (`CodeProvider`) | Dense-encode the exact supplied function and construct reusable KV state | None |
| 2 | CWE Security Validator (`CweValidator`) | Review the function against ten fixed CWE definitions | System and user prompts below |

Vulnerable and patched functions are processed in separate sweeps so that reusable state derived from a vulnerable function cannot affect its patched counterpart.

### Agent 1: Code Provider

The Code Provider is promptless. It receives only the exact code string from PrimeVul and performs a dense forward pass over it. It does not generate code, receive the natural-language review request, or see the sample's label.

### Agent 2: CWE Security Validator

#### System prompt

~~~text
You are a security validator and code reviewer. You are given source code and a set of CWE-derived security rules. Review the code strictly against each rule and produce a report in EXACTLY the following format, with no other sections:

### Security Validation Report

#### Rule <n>: <full rule text>
**PASS**, **FAIL**, or **NOT APPLICABLE**
- <evidence from the code justifying the verdict>

(Repeat the '#### Rule' block above for every rule, in order. The line directly under each rule heading must contain only the single verdict **PASS**, **FAIL**, or **NOT APPLICABLE** in bold. Use **NOT APPLICABLE** only when the weakness class cannot occur in the supplied code.)

### Additional Recommendations
- <security improvements beyond the listed rules>

### Code Review Comments
- <general code-quality observations (structure, readability, error handling)>
~~~

#### User prompt

~~~text
A developer agent produced the following code:

<PRIMEVUL_FUNCTION>

Validate the code above against the following security rules (CWE weakness definitions):
1. CWE-119 - Improper Restriction of Operations within the Bounds of a Memory Buffer: Operations are performed on a memory buffer in a way that can read from or write to a location outside the intended boundary of that buffer.
2. CWE-125 - Out-of-bounds Read: Data is read from before the beginning or past the end of the intended buffer.
3. CWE-787 - Out-of-bounds Write: Data is written to before the beginning or past the end of the intended buffer.
4. CWE-20 - Improper Input Validation: Input is received but is not validated, or is incorrectly validated, for the properties required to process it safely.
5. CWE-476 - NULL Pointer Dereference: A pointer that is expected to be valid is dereferenced while it may be NULL.
6. CWE-200 - Exposure of Sensitive Information to an Unauthorized Actor: Sensitive information is made available to an actor that is not explicitly authorised to have access to it.
7. CWE-416 - Use After Free: Memory is referenced or reused after it has been freed.
8. CWE-703 - Improper Check or Handling of Exceptional Conditions: Exceptional conditions that rarely occur during normal operation are not anticipated or not handled correctly.
9. CWE-190 - Integer Overflow or Wraparound: A calculation can overflow or wrap around when the logic assumes the result is always larger than its inputs.
10. CWE-399 - Resource Management Errors: System resources such as memory, file handles or other limited resources are acquired, tracked or released incorrectly.

Produce a security validation report citing every rule above as PASS, FAIL, or NOT APPLICABLE with reasons grounded only in the supplied code.
~~~

`<PRIMEVUL_FUNCTION>` is replaced with the exact vulnerable or patched function.

### Information deliberately hidden from the validator

The following PrimeVul fields are retained only for orchestration and offline scoring and are not inserted into the model prompt:

- Vulnerable or patched arm
- Binary ground-truth label
- Target CWE label
- Sample and pair identifiers
- Project, file, and function metadata
- Natural-language review request stored in the dataset's `task` field

This prevents the labelled target CWE from being privileged over the other nine rules.

### Exact PrimeVul example

For sample `PV087`, the model-visible code span is:

~~~c
size_t intsetBlobLen(intset *is) {
    return sizeof(intset)+intrev32ifbe(is->length)*intrev32ifbe(is->encoding);
}
~~~

The CWE Security Validator receives:

~~~text
SYSTEM
<the complete CWE Security Validator system prompt above>

USER
A developer agent produced the following code:

size_t intsetBlobLen(intset *is) {
    return sizeof(intset)+intrev32ifbe(is->length)*intrev32ifbe(is->encoding);
}

Validate the code above against the following security rules (CWE weakness definitions):
<the complete ten-rule CWE list above>

Produce a security validation report citing every rule above as PASS, FAIL, or NOT APPLICABLE with reasons grounded only in the supplied code.
~~~

The validator does not see that this is `PV087`, which arm it belongs to, or which CWE is externally labelled.

## Experimental configuration from the paper

| Item | Configuration |
|---|---|
| Models | Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-Coder-7B-Instruct |
| Decoding | Greedy; sampling disabled |
| Maximum generation length | 6000 tokens |
| Pipeline depth | Two stages, one communication round |
| KVCOMM thresholds | `γ ∈ {0.1, 0.3, 0.5}` |
| KVCOMM anchor limit | 20 active anchors |
| CacheBlend ratios | `r ∈ {0.0, 0.16, 1.0}` |
| Primary data | 90 unlabelled MINIMAXAI/VIBE application-development tasks |
| Labelled data | 100 vulnerable and 100 patched PrimeVul functions across ten CWE categories |

## What is and is not changed between comparison arms

Held fixed within every fidelity pair:

- Validator model and tokenizer
- System prompt
- User prompt token sequence
- Original task, when the VIBE prompt includes one
- Reviewed code artifact
- Security-rule catalogue
- Report-format request
- Greedy decoding configuration

Changed:

- The receiver-side KV computation: approximate reuse versus exact dense prefill

Dense inference is the exact-execution reference, not a claim of objective security ground truth. PrimeVul is used separately to determine whether some reuse-suppressed dense findings correspond to externally labelled vulnerabilities.

## Source files

- Paper methodology: `paper/paper.tex`, especially “Experimental Design and Agentic Pipeline,” “Security Validation Policy,” and “Datasets and Benchmarks”
- VIBE prompts: `KVCOMM/prompt/secure_code_prompt_set.py` and `KVCOMM/prompt/secure_code_common.py`
- PrimeVul prompts: `KVCOMM/prompt/primevul_prompt_set.py` and `KVCOMM/prompt/primevul_common.py`
- Experiment runners: `experiments/run_secure_code.py` and `experiments/run_primevul.py`
