"""String-only helpers for the secure-code experiment.

These helpers build and validate the Security Validator's user prompt. They
contain NO KV-cache logic — they only guarantee that the generated code appears
as a single trailing placeholder followed by the NIST rules as exact text, which
is what makes the existing `agen_kvcomm` pipeline keep the NIST rules exactly
prefilled (text segments are never offset-approximated).
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Same pattern the engine uses in `LLMChat.locate_placeholder`.
PLACEHOLDER_PATTERN = r"\{((?:agent|condition)_\w+_(?:current|history)|user_question)\}"

NIST_HEADER = "Validate the code above against the following security rules (NIST SP 800-63B style):"


def _placeholder_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, inner_name) for every placeholder, in order."""
    return [(m.start(), m.end(), m.group(1)) for m in re.finditer(PLACEHOLDER_PATTERN, text)]


def build_validator_user_prompt(*, generator_agent_id: str, nist_rules: str) -> str:
    """Render the validator user prompt: task, then code placeholder, then NIST text.

    The generated code is referenced as `{agent_<id>_current}` (a response
    placeholder) and the NIST rules follow as plain trailing text.
    """
    code_placeholder = "{agent_" + str(generator_agent_id) + "_current}"
    return (
        "The task is: {user_question}\n\n"
        "A developer agent produced the following code:\n\n"
        f"{code_placeholder}\n\n"
        f"{NIST_HEADER}\n{nist_rules}\n\n"
        "Produce a security validation report citing each rule as PASS, FAIL, or "
        "NOT APPLICABLE with reasons."
    )


def validate_validator_prompt_structure(prompt: str, *, generator_agent_id: str) -> str:
    """Assert NIST rules are exact-prefilled after the single code placeholder.

    Returns the trailing exact-text segment (everything after the last
    placeholder). Raises ValueError if the structure would let NIST rules be
    approximated.
    """
    code_inner = "agent_" + str(generator_agent_id) + "_current"
    spans = _placeholder_spans(prompt)

    # All agent_*_current placeholders count as "generated-code" placeholders.
    all_code_spans = [s for s in spans if re.match(r"agent_\w+_current", s[2])]
    if len(all_code_spans) != 1:
        raise ValueError(
            f"Prompt must contain exactly one generated-code placeholder "
            f"(agent_*_current), found {len(all_code_spans)}."
        )

    code_spans = [s for s in spans if s[2] == code_inner]
    if len(code_spans) != 1:
        raise ValueError(
            f"Prompt must contain exactly one generated-code placeholder "
            f"'{{{code_inner}}}', found {len(code_spans)}."
        )

    code_start, code_end, _ = code_spans[0]
    last_start = max(s[0] for s in spans)
    if code_start != last_start:
        raise ValueError(
            "The generated-code placeholder must be the LAST placeholder so the "
            "NIST rules form a trailing exact-text segment."
        )

    trailing = prompt[code_end:]
    if NIST_HEADER not in trailing:
        raise ValueError(
            "NIST rules must be exact-prefilled AFTER the generated-code placeholder; "
            "NIST header was not found in the trailing text segment."
        )
    if not trailing.strip():
        raise ValueError("Trailing NIST text segment is empty.")
    return trailing
