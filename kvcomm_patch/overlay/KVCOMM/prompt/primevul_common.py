"""String-only helpers for the PrimeVul ground-truth experiment.

These helpers build and validate the Security Validator's metadata-free user
prompt and contain NO KV-cache logic. They guarantee that the code under
review appears as a single trailing placeholder
followed by the CWE catalogue as exact text, which is what makes the existing
``agen_kvcomm`` pipeline keep the catalogue exactly prefilled (text segments
are never offset-approximated).

The catalogue below is the structural analogue of the ten NIST rules used by
the VIBE arm: ten numbered weakness definitions, fixed for the whole run. Every
sample is assessed against all ten rules, just as every VIBE sample is assessed
against all ten NIST rules.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Same pattern the engine uses in `LLMChat.locate_placeholder`.
PLACEHOLDER_PATTERN = r"\{((?:agent|condition)_\w+_(?:current|history)|user_question)\}"

# Deprecated compatibility field for old artifact readers.  The corrected
# PrimeVul prompt has no task/query segment at all.
REVIEW_PREFIX = ""

CWE_HEADER = (
    "Validate the code above against the following security rules "
    "(CWE weakness definitions):"
)

# Canonical order == descending support in PrimeVul after the length filter.
# The rule number is the stable 1-based position in this list.
CWE_INFO: Tuple[Tuple[str, str, str], ...] = (
    ("CWE-119", "Improper Restriction of Operations within the Bounds of a Memory Buffer",
     "Operations are performed on a memory buffer in a way that can read from or "
     "write to a location outside the intended boundary of that buffer."),
    ("CWE-125", "Out-of-bounds Read",
     "Data is read from before the beginning or past the end of the intended buffer."),
    ("CWE-787", "Out-of-bounds Write",
     "Data is written to before the beginning or past the end of the intended buffer."),
    ("CWE-20", "Improper Input Validation",
     "Input is received but is not validated, or is incorrectly validated, for the "
     "properties required to process it safely."),
    ("CWE-476", "NULL Pointer Dereference",
     "A pointer that is expected to be valid is dereferenced while it may be NULL."),
    ("CWE-200", "Exposure of Sensitive Information to an Unauthorized Actor",
     "Sensitive information is made available to an actor that is not explicitly "
     "authorised to have access to it."),
    ("CWE-416", "Use After Free",
     "Memory is referenced or reused after it has been freed."),
    ("CWE-703", "Improper Check or Handling of Exceptional Conditions",
     "Exceptional conditions that rarely occur during normal operation are not "
     "anticipated or not handled correctly."),
    ("CWE-190", "Integer Overflow or Wraparound",
     "A calculation can overflow or wrap around when the logic assumes the result "
     "is always larger than its inputs."),
    ("CWE-399", "Resource Management Errors",
     "System resources such as memory, file handles or other limited resources are "
     "acquired, tracked or released incorrectly."),
)

CWE_IDS: Tuple[str, ...] = tuple(cid for cid, _name, _defn in CWE_INFO)
CWE_NAME: Dict[str, str] = {cid: name for cid, name, _defn in CWE_INFO}
# 1-based catalogue position, quoted in the review request and in the report.
CWE_RULE_NUMBER: Dict[str, int] = {cid: i + 1 for i, (cid, _n, _d) in enumerate(CWE_INFO)}

CWE_CATALOGUE = "\n".join(
    f"{i + 1}. {cid} - {name}: {defn}"
    for i, (cid, name, defn) in enumerate(CWE_INFO)
)

OUTPUT_ASK = (
    "Produce a security validation report citing every rule above as PASS, FAIL, "
    "or NOT APPLICABLE with reasons grounded only in the supplied code."
)


def build_review_request(*, sample_id: str, function_name: str, file_name: str,
                         project: str) -> str:
    """The neutral variable query segment; identical for a pair's twins.

    Carries enough surface variation (function, file, project) for the online
    anchor pool to behave as it does on the VIBE arm. It does not name the
    labelled CWE, because doing so would privilege one of the ten rules. The
    opaque sample id keeps the request unique for KV joins and artifact paths.
    """
    return (
        f"Sample {sample_id}: review function `{function_name}` in `{file_name}` "
        f"from project `{project}` for security weaknesses."
    )


def _placeholder_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, inner_name) for every placeholder, in order."""
    return [(m.start(), m.end(), m.group(1)) for m in re.finditer(PLACEHOLDER_PATTERN, text)]


def build_validator_user_prompt(*, provider_agent_id: str, cwe_catalogue: str = CWE_CATALOGUE) -> str:
    """Render the validator prompt: code placeholder, then exact CWE text.

    The code under review is referenced as `{agent_<id>_current}` (a response
    placeholder) and the catalogue follows as plain trailing text.
    """
    code_placeholder = "{agent_" + str(provider_agent_id) + "_current}"
    return (
        "A developer agent produced the following code:\n\n"
        f"{code_placeholder}\n\n"
        f"{CWE_HEADER}\n{cwe_catalogue}\n\n"
        f"{OUTPUT_ASK}"
    )


def validate_validator_prompt_structure(prompt: str, *, provider_agent_id: str) -> str:
    """Assert the CWE catalogue is exact-prefilled after the single code placeholder.

    Returns the trailing exact-text segment (everything after the last
    placeholder). Raises ValueError if the structure would let the catalogue be
    approximated.
    """
    code_inner = "agent_" + str(provider_agent_id) + "_current"
    spans = _placeholder_spans(prompt)

    if "{user_question}" in prompt:
        raise ValueError("PrimeVul validator prompt must not contain user_question.")
    if len(spans) != 1:
        raise ValueError(
            f"PrimeVul validator prompt must contain exactly one placeholder, found {len(spans)}."
        )

    all_code_spans = [s for s in spans if re.match(r"agent_\w+_current", s[2])]
    if len(all_code_spans) != 1:
        raise ValueError(
            f"Prompt must contain exactly one code placeholder (agent_*_current), "
            f"found {len(all_code_spans)}."
        )

    code_spans = [s for s in spans if s[2] == code_inner]
    if len(code_spans) != 1:
        raise ValueError(
            f"Prompt must contain exactly one code placeholder "
            f"'{{{code_inner}}}', found {len(code_spans)}."
        )

    code_start, code_end, _ = code_spans[0]
    last_start = max(s[0] for s in spans)
    if code_start != last_start:
        raise ValueError(
            "The code placeholder must be the LAST placeholder so the CWE "
            "catalogue forms a trailing exact-text segment."
        )

    trailing = prompt[code_end:]
    if CWE_HEADER not in trailing:
        raise ValueError(
            "The CWE catalogue must be exact-prefilled AFTER the code placeholder; "
            "the catalogue header was not found in the trailing text segment."
        )
    if not trailing.strip():
        raise ValueError("Trailing CWE text segment is empty.")
    return trailing
