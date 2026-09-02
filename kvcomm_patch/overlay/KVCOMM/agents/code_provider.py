"""PrimeVul Agent 1: a promptless, dense-only encoder of dataset code."""
from __future__ import annotations

from typing import Any, Dict

from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.kvcomm_engine import BareCodeKVPackage


@AgentRegistry.register("CodeProvider")
class CodeProvider(Node):
    """Encode one exact code string; never prompt, generate, gate, or reuse."""

    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "CodeProvider", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, prefix=None, llm_config=llm_config)
        self.role = "Code Provider" if role is None else role
        self.llm.set_id(self.id, self.role)
        self.force_dense = True
        self.last_package_metadata: Dict[str, Any] = {}

    async def encode_code(
        self,
        code: str,
        *,
        request_uid: str,
        output_dir: str | None = None,
    ) -> BareCodeKVPackage:
        """Pass only ``code`` to the model-facing dense encoder."""
        if not isinstance(code, str) or not code:
            raise ValueError("CodeProvider requires the exact non-empty code string.")
        package = await self.llm.encode_bare_code_kv(
            code,
            request_uid=request_uid,
            source_agent_id=self.id,
            output_dir=output_dir,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        self.last_package_metadata = {
            **package.audit_metadata(),
            "execution_mode": "dense_code_encode",
        }
        return package

    async def _process_inputs(self, *args, **kwargs):
        raise RuntimeError("CodeProvider is promptless; prompt processing is forbidden.")

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        raise RuntimeError("CodeProvider supports only its asynchronous dense encoder.")

    async def _async_execute(
        self,
        input,
        spatial_info,
        temporal_info,
        mode: str = "default",
        **kwargs,
    ):
        if mode not in {"default", "allow_kv_reuse"}:
            raise ValueError(f"Unsupported CodeProvider mode: {mode}")
        if not isinstance(input, str):
            raise TypeError("CodeProvider input must be the exact code string, not a record.")
        request_uid = kwargs.get("request_uid")
        if not request_uid:
            raise ValueError("request_uid is required for the opaque code package.")
        return await self.encode_code(
            input, request_uid=request_uid, output_dir=kwargs.get("output_dir")
        )
