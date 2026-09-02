"""Prompt set for the PrimeVul ground-truth experiment (domain 'PrimeVul')."""
from typing import Any, Dict

from KVCOMM.prompt.prompt_set import PromptSet
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.prompt.common import get_combine_materials
from KVCOMM.prompt.primevul_common import CWE_CATALOGUE

# The provider never generates: its response is teacher-forced to the dataset's
# function so the code under review is byte-identical everywhere. The
# instruction exists only to give the provider a stable prompt prefix (and
# therefore a stable KV prefix) across tasks.
PROVIDER_INSTRUCTION = (
    "You are a code repository agent. You return the requested source file "
    "region verbatim. Do not include explanations."
)

VALIDATOR_INSTRUCTION = (
    "You are a security validator and code reviewer. You are given source code "
    "and a set of CWE-derived security rules. Review the code strictly against "
    "each rule and produce a report "
    "in EXACTLY the following format, with no other sections:\n\n"
    "### Security Validation Report\n\n"
    "#### Rule <n>: <full rule text>\n"
    "**PASS**, **FAIL**, or **NOT APPLICABLE**\n"
    "- <evidence from the code justifying the verdict>\n\n"
    "(Repeat the '#### Rule' block above for every rule, in order. The line "
    "directly under each rule heading must contain only the single verdict "
    "**PASS**, **FAIL**, or **NOT APPLICABLE** in bold. Use **NOT APPLICABLE** "
    "only when the weakness class cannot occur in the supplied code.)\n\n"
    "### Additional Recommendations\n"
    "- <security improvements beyond the listed rules>\n\n"
    "### Code Review Comments\n"
    "- <general code-quality observations (structure, readability, error handling)>"
)

_ROLE_INSTRUCTIONS = {
    "Code Provider": PROVIDER_INSTRUCTION,
    "Security Validator": VALIDATOR_INSTRUCTION,
}


@PromptSetRegistry.register("PrimeVul")
class PrimeVulPromptSet(PromptSet):

    @staticmethod
    def get_role() -> str:
        return "Code Provider"

    @staticmethod
    def get_analyze_constraint(role: str) -> str:
        return _ROLE_INSTRUCTIONS.get(role, PROVIDER_INSTRUCTION)

    @staticmethod
    def get_cwe_catalogue() -> str:
        return CWE_CATALOGUE

    # The validator agent is shared with the SecureCode domain, which asks its
    # prompt set for `get_nist_rules()`. Alias it so the same agent class can
    # serve both domains without a branch.
    @staticmethod
    def get_nist_rules() -> str:
        return CWE_CATALOGUE

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
