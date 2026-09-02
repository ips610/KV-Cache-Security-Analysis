"""Agent 2: validates Agent 1's generated code against NIST-style rules.

The generated code is consumed as a single `{agent_<gen>_current}` response
placeholder, immediately followed by the NIST rules as EXACT trailing text.
This is what lets the existing `agen_kvcomm` pipeline approximate only the code
KV (via `offset_kv_cache_pair`) while keeping the NIST rules exactly prefilled.
"""
import asyncio
from typing import Any, Dict

from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.prompt.secure_code_common import (
    NIST_HEADER,
    build_validator_user_prompt,
    validate_validator_prompt_structure,
)
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import logger


@AgentRegistry.register('SecurityValidator')
class SecurityValidator(Node):
    """Reviews a code generator's output against NIST-style security rules."""

    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "SecurityValidator", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, prefix="", llm_config=llm_config)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Security Validator" if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.nist_rules = self.prompt_set.get_nist_rules()
        # Task text -> full report text of the on-hit measurement-only dense
        # baseline, so runners can persist it next to the natural kv_reuse report
        # for a same-task side-by-side comparison. Only populated on anchor hits.
        self.dense_baseline_reports: Dict[str, str] = {}
        # Second, independent measurement-only dense baseline for the SAME task.
        # Diffing it against `dense_baseline_reports` yields the dense-vs-dense
        # noise floor on exactly the paired cases fidelity is reported over, so
        # any KV-reuse divergence can be attributed rather than assumed.
        # Populated only when `noise_floor_baseline` is enabled (it costs one
        # extra full dense generation per anchor hit).
        self.dense_baseline_reports_b: Dict[str, str] = {}
        self.noise_floor_baseline: bool = False
        # When True, run a dense reference for EVERY task before the natural pass
        # (dense -> [dense again] -> kv_reuse-when-applicable), instead of only
        # recording a baseline on anchor hits. Gives every prompt a paired dense
        # reference and keeps the two arms symmetric.
        self.dense_reference_always: bool = False

    def _build_validator_prompts(self, raw_inputs, spatial_info):
        """Return (system_prompt, user_prompt) with code placeholder + exact NIST text.

        Uses the first spatial predecessor (the Code Generator) as the source of
        the `{agent_<id>_current}` code placeholder.
        """
        system_prompt = f"{self.constraint}"
        generator_agent_id = None
        for agent_id, info in spatial_info.items():
            generator_agent_id = agent_id
            break
        if generator_agent_id is None:
            raise ValueError("SecurityValidator requires a spatial predecessor (CodeGenerator).")
        user_prompt = build_validator_user_prompt(
            generator_agent_id=generator_agent_id,
            nist_rules=self.nist_rules,
        )
        # Enforce --force-exact-nist-prefill: NIST must be a trailing exact-text segment.
        validate_validator_prompt_structure(user_prompt, generator_agent_id=generator_agent_id)
        return system_prompt, user_prompt

    async def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        mode: str = "default",
        **kwargs,
    ) -> Dict[str, Any]:
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)

            if self.llm.has_prefix_initialized(self.id) and "placeholder_info" in agent_memory:
                prefix_text = kwargs.get('prefix', "The task is: ")
                user_content = prefix_text + raw_inputs['task']
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs['task'],
                    user_content=user_content,
                    prefix_text=prefix_text,
                )
                logger.opt(colors=True).info(
                    "<green>[MODE]</green> Task: {} Agent {} ({}) mode: {}",
                    raw_inputs["task"], self.id, self.role, preferred_mode,
                )
                return {"preferred_mode": preferred_mode, "early_response": None}

            system_prompt, user_prompt = self._build_validator_prompts(raw_inputs, spatial_info)
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}

        # default (baseline-by-dense) path: inline the actual generated code as text.
        return self._build_default_prompts(raw_inputs, spatial_info)

    def _build_default_prompts(self, raw_inputs, spatial_info) -> Dict[str, Any]:
        """Flat, placeholder-free prompts for the non-KV ``default`` path.

        Split out of ``_process_inputs`` so a domain subclass can swap the rubric
        without duplicating the KV-reuse branch.
        """
        system_prompt = f"{self.constraint}"
        code_text = ""
        for _agent_id, info in spatial_info.items():
            code_text = info['output']
            break
        user_prompt = (
            f"The task is: {raw_inputs['task']}\n\n"
            "A developer agent produced the following code:\n\n"
            f"{code_text}\n\n"
            f"{NIST_HEADER}\n{self.nist_rules}\n\n"
            "Produce a security validation report citing each rule as PASS, FAIL, or "
            "NOT APPLICABLE with reasons."
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        inputs = asyncio.run(
            self._process_inputs(input, spatial_info, temporal_info, mode="default", **kwargs)
        )
        message = [
            {'role': 'system', 'content': inputs["system_prompt"]},
            {'role': 'user', 'content': inputs["user_prompt"]},
        ]
        return self.llm.gen(message)

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], mode: str = "default", **kwargs):
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(input, spatial_info, temporal_info, mode=mode, **kwargs)
            message = [
                {'role': 'system', 'content': inputs["system_prompt"]},
                {'role': 'user', 'content': inputs["user_prompt"]},
            ]
            return await self.llm.agen(
                message,
                max_tokens=self.llm.DEFAULT_MAX_TOKENS,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )

        assert mode == "allow_kv_reuse", f"Unsupported async execution mode: {mode}"
        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")
        mode_data = await self._process_inputs(input, spatial_info, temporal_info, mode=mode, **kwargs)

        # Per-task order when `dense_reference_always` is set:
        #   1. the natural pass -> KV reuse when the entropy gate fires, else dense.
        #      Attempted for every task regardless of what the verdicts turn out
        #      to be; this is the run that updates the anchor pool.
        #   2. dense prefill    -> the reference report, recorded for EVERY task
        #      (not only anchor hits, as in the legacy path below).
        #   3. (optional) a second dense prefill -> a within-cell noise floor.
        # The reference runs use measure_only=True, which suppresses the
        # anchor-pool write, so they cannot bias later tasks' natural hit rate.
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input['task'],
            max_tokens=kwargs.get("max_tokens"),
            preferred_mode=mode_data["preferred_mode"],
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )

        if self.dense_reference_always:
            logger.opt(colors=True).info(
                "<green>[REFERENCE]</green> dense prefill after {} for: {}",
                getattr(result, "mode", "?"), input["task"][:60],
            )
            reference = await self.llm.generate_for_agent(
                request_uid=request_uid,
                message=input['task'],
                max_tokens=kwargs.get("max_tokens"),
                preferred_mode="dense_prefill",
                measure_only=True,
                output_dir=kwargs.get("output_dir"),
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            self.dense_baseline_reports[input['task']] = getattr(reference, "text", "") or ""
            if self.noise_floor_baseline:
                logger.opt(colors=True).info(
                    "<green>[NOISE-FLOOR]</green> second dense prefill for: {}",
                    input["task"][:60],
                )
                reference_b = await self.llm.generate_for_agent(
                    request_uid=request_uid,
                    message=input['task'],
                    max_tokens=kwargs.get("max_tokens"),
                    preferred_mode="dense_prefill",
                    measure_only=True,
                    output_dir=kwargs.get("output_dir"),
                    agent_id=self.id,
                    agent_name=self.agent_name,
                    agent_role=self.role,
                )
                self.dense_baseline_reports_b[input['task']] = (
                    getattr(reference_b, "text", "") or ""
                )
            return input['task'], result

        # Legacy path (FinalRun and earlier): the dense baseline is recorded only
        # on a real anchor hit, since a natural miss already ran dense and is its
        # own reference. Retained so existing sweeps stay reproducible.
        if getattr(result, "mode", None) == "kv_reuse":
            logger.opt(colors=True).info(
                "<green>[BASELINE]</green> Anchor hit -> recording dense baseline for: {}",
                input["task"][:60],
            )
            baseline_result = await self.llm.generate_for_agent(
                request_uid=request_uid,
                message=input['task'],
                max_tokens=kwargs.get("max_tokens"),
                preferred_mode="dense_prefill",
                measure_only=True,
                output_dir=kwargs.get("output_dir"),
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            self.dense_baseline_reports[input['task']] = getattr(baseline_result, "text", "") or ""

            # Optional second dense baseline of the same validation. Identical
            # inputs and identical prefill path as the first, so any difference
            # between the two is pipeline noise, not KV reuse — this is the
            # measured noise floor (see docs spec).
            if self.noise_floor_baseline:
                logger.opt(colors=True).info(
                    "<green>[NOISE-FLOOR]</green> Recording second dense baseline for: {}",
                    input["task"][:60],
                )
                baseline_result_b = await self.llm.generate_for_agent(
                    request_uid=request_uid,
                    message=input['task'],
                    max_tokens=kwargs.get("max_tokens"),
                    preferred_mode="dense_prefill",
                    measure_only=True,
                    output_dir=kwargs.get("output_dir"),
                    agent_id=self.id,
                    agent_name=self.agent_name,
                    agent_role=self.role,
                )
                self.dense_baseline_reports_b[input['task']] = (
                    getattr(baseline_result_b, "text", "") or ""
                )
        return input['task'], result
