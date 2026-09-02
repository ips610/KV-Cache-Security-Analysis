"""Agent 1: generates code from a task (domain 'SecureCode')."""
import asyncio
from typing import Any, Dict

from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import logger


@AgentRegistry.register('CodeGenerator')
class CodeGenerator(Node):
    """Generates secure, production-ready code; has no spatial predecessors."""

    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "CodeGenerator", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, prefix="", llm_config=llm_config)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = "Code Generator" if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        # When True the generator never uses KV reuse, so the code under review is
        # always produced by an exact forward pass. Confines the optimization under
        # study to the validator.
        self.force_dense: bool = False

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

            system_prompt = f"{self.constraint}"
            user_prompt = "The task is: {user_question}\n"
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}

        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
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
        # KV reuse is the object of study and belongs to the validator alone. When
        # force_dense is set the generator always dense-prefills, so the code under
        # review is produced by an unapproximated forward pass and is identical
        # across every arm, threshold and repetition. Anchor bookkeeping in
        # _process_inputs still runs, so the validator's pool is unaffected.
        preferred_mode = "dense_prefill" if self.force_dense else mode_data["preferred_mode"]
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input['task'],
            preferred_mode=preferred_mode,
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return input['task'], result
