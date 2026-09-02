"""Prompt set for the two-agent secure-code experiment (domain 'SecureCode')."""
from typing import Any, Dict

from KVCOMM.prompt.prompt_set import PromptSet
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.prompt.common import get_combine_materials

GENERATOR_INSTRUCTION = (
    "You are a senior backend engineer. Write secure, production-ready "
    "code for the requested task. Do not include explanations."
)

VALIDATOR_INSTRUCTION = (
    "You are a security validator and code reviewer. You are given code "
    "written by a developer agent and a set of NIST-derived security rules. "
    "Review the code strictly against each rule and produce a report in EXACTLY the "
    "following format, with no other sections:\n\n"
    "### Security Validation Report\n\n"
    "#### Rule <n>: <full rule text>\n"
    "**PASS**, **FAIL**, or **NOT APPLICABLE**\n"
    "- <evidence from the code justifying the verdict>\n\n"
    "(Repeat the '#### Rule' block above for every rule, in order. The line "
    "directly under each rule heading must contain only the single verdict "
    "**PASS**, **FAIL**, or **NOT APPLICABLE** in bold. Use **NOT APPLICABLE** "
    "only when the rule's subject matter — e.g. passwords, sessions, database "
    "queries — does not exist anywhere in the task or the code.)\n\n"
    "### Additional Recommendations\n"
    "- <security improvements beyond the listed rules>\n\n"
    "### Code Review Comments\n"
    "- <general code-quality observations (structure, readability, error handling)>"
)

NIST_RULES = (
    "1. Passwords must be hashed with a memory-hard algorithm (bcrypt, scrypt, or argon2); "
    "never store or compare plaintext passwords.\n"
    "2. Secret comparisons (tokens, MACs, passwords) must be constant-time.\n"
    "3. Session tokens must be generated with a cryptographically secure RNG (>=128 bits).\n"
    "4. Never log secrets, passwords, or full tokens.\n"
    "5. Enforce account lockout or rate limiting on repeated failed authentication attempts.\n"
    "6. No hardcoded secrets, API keys, or credentials in source code; load them from "
    "configuration or environment.\n"
    "7. User-supplied input used in database queries must be parameterized; never build "
    "SQL by string concatenation or interpolation.\n"
    "8. Verification, password-reset, and magic-link tokens must be single-use and expire.\n"
    "9. Authentication failure messages must be generic; do not reveal whether a username "
    "or email exists.\n"
    "10. Session cookies must set the Secure and HttpOnly flags, and credentials must only "
    "be transmitted over TLS."
)

_ROLE_INSTRUCTIONS = {
    "Code Generator": GENERATOR_INSTRUCTION,
    "Security Validator": VALIDATOR_INSTRUCTION,
}


@PromptSetRegistry.register("SecureCode")
class SecureCodePromptSet(PromptSet):

    @staticmethod
    def get_role() -> str:
        return "Code Generator"

    @staticmethod
    def get_analyze_constraint(role: str) -> str:
        return _ROLE_INSTRUCTIONS.get(role, GENERATOR_INSTRUCTION)

    @staticmethod
    def get_nist_rules() -> str:
        return NIST_RULES

    @staticmethod
    def get_constraint() -> str:
        return ""

    @staticmethod
    def get_decision_role() -> str:
        return ""

    @staticmethod
    def get_decision_constraint() -> str:
        return ""

    @staticmethod
    def get_decision_few_shot() -> str:
        return ""

    @staticmethod
    def get_format():
        raise NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"{question}"

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
