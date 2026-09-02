"""PrimeVul Agent 2: validate an opaque code handoff against ten CWE rules."""
from __future__ import annotations

from typing import Any, Dict, Optional

from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.agents.security_validator import SecurityValidator
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.prompt.primevul_common import (
    CWE_HEADER,
    OUTPUT_ASK,
    build_validator_user_prompt,
    validate_validator_prompt_structure,
)


@AgentRegistry.register("CweValidator")
class CweValidator(SecurityValidator):
    """PrimeVul validator with a metadata-free, single-code-placeholder prompt."""

    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(
            id=id,
            role=role,
            domain=domain,
            llm_name=llm_name,
            llm_config=llm_config,
        )
        self.cwe_catalogue = self.nist_rules

    def _build_validator_prompts(self, raw_inputs, spatial_info):
        """Keep the existing validator system message; remove all record metadata."""
        provider_agent_id = next(iter(spatial_info), None)
        if provider_agent_id is None:
            raise ValueError("CweValidator requires a CodeProvider predecessor.")
        user_prompt = build_validator_user_prompt(
            provider_agent_id=provider_agent_id,
            cwe_catalogue=self.cwe_catalogue,
        )
        validate_validator_prompt_structure(
            user_prompt, provider_agent_id=provider_agent_id
        )
        return str(self.constraint), user_prompt

    def _build_default_prompts(self, raw_inputs, spatial_info) -> Dict[str, Any]:
        """Metadata-free flat form retained for isolated dense prompt checks."""
        code_text = next(
            (info["output"] for info in spatial_info.values()), ""
        )
        return {
            "system_prompt": str(self.constraint),
            "user_prompt": (
                "A developer agent produced the following code:\n\n"
                f"{code_text}\n\n{CWE_HEADER}\n{self.cwe_catalogue}\n\n{OUTPUT_ASK}"
            ),
        }

    async def ensure_primevul_prefix(self, *, source_agent_id: str) -> None:
        """Prepare Agent 2's one-placeholder template once per model/node."""
        memory = self.llm._ensure_agent_memory(self.id)
        if self.llm.has_prefix_initialized(self.id) and "placeholder_info" in memory:
            placeholders = set(memory["placeholder_info"])
            expected = {f"agent_{source_agent_id}_current"}
            if placeholders != expected:
                raise RuntimeError(
                    f"PrimeVul validator prefix mismatch: {placeholders} != {expected}."
                )
            return
        system_prompt, user_prompt = self._build_validator_prompts(
            {}, {str(source_agent_id): {"role": "Code Provider"}}
        )
        await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)

    async def run_primevul_stage(
        self,
        *,
        request_uid: str,
        source_agent_id: str,
        audit_key: str,
        preferred_mode: str,
        payload: Dict[str, Any],
        max_tokens: Optional[int],
        output_dir: Optional[str],
        measure_only: bool = False,
    ):
        """Execute one strict dense or opaque-KV validator pass.

        The hit payload has exactly one opaque package field. The dense payload
        has exactly one JSON code field, used solely to re-tokenize/assert the
        code span before the complete prompt is densely evaluated.
        """
        if preferred_mode not in {"dense_prefill", "kv_reuse"}:
            raise ValueError(f"Invalid PrimeVul Agent 2 mode: {preferred_mode}")
        await self.ensure_primevul_prefix(source_agent_id=source_agent_id)
        package = self.llm.get_bare_code_package(
            request_uid=request_uid, source_agent_id=source_agent_id
        )
        if preferred_mode == "kv_reuse":
            if set(payload) != {"opaque_package"}:
                raise RuntimeError(
                    "PrimeVul KV-hit payload must contain only the opaque package."
                )
            if payload["opaque_package"] is not package:
                raise RuntimeError("PrimeVul KV-hit payload package identity mismatch.")
        else:
            if set(payload) != {"code"} or not isinstance(payload["code"], str):
                raise RuntimeError(
                    "PrimeVul dense payload must contain only JSON code from orchestration."
                )
            self.llm.assert_dense_code_matches_package(
                request_uid=request_uid,
                source_agent_id=source_agent_id,
                code=payload["code"],
            )

        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=audit_key,
            max_tokens=max_tokens,
            preferred_mode=preferred_mode,
            measure_only=measure_only,
            output_dir=output_dir,
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
            strict_anchor_reuse=(preferred_mode == "kv_reuse"),
        )
        if result.mode != preferred_mode:
            raise RuntimeError(
                f"PrimeVul Agent 2 mode drift: selected {preferred_mode}, ran {result.mode}."
            )
        if preferred_mode == "kv_reuse":
            if result.metadata.get("offset_fallback") is not False:
                raise RuntimeError("PrimeVul KV hit attempted an offset fallback.")
            if result.metadata.get("anchor_hit") is not True:
                raise RuntimeError("PrimeVul KV hit did not apply a genuine anchor delta.")
        result.metadata.update({
            "handoff_schema": "primevul_bare_code_kv_v2",
            "code_sha256": package.code_sha256,
            "code_token_count": package.token_count,
            "genuine_anchor": result.metadata.get("anchor_hit") is True,
            "fallback_status": bool(result.metadata.get("offset_fallback")),
        })
        return result

    async def _async_execute(self, input, spatial_info, temporal_info, mode="default", **kwargs):
        raise RuntimeError(
            "CweValidator must be run through the staged PrimeVul orchestration path."
        )
