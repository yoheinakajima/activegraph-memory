"""Optional reasoning backend contracts used by the staged memory runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .query_ir import QueryAnalysis


ReasoningStage = Literal[
    "query_classification",
    "retrieval_strategy",
    "retrieval_analysis",
    "context_packaging",
]


@dataclass(frozen=True)
class ReasoningRequest:
    """A provider-neutral structured reasoning request."""

    stage: ReasoningStage
    payload: dict[str, Any]
    output_contract: dict[str, Any]
    instructions: str = ""


@dataclass(frozen=True)
class ReasoningResponse:
    """Structured reasoning output plus provider-reported usage."""

    data: dict[str, Any]
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ReasoningBackend(Protocol):
    """Small seam for LLM, local-model, or recorded reasoning providers."""

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        """Return a structured response satisfying ``output_contract``."""

        ...


class CallableReasoningBackend:
    """Adapter for applications that already have a structured-call function."""

    def __init__(self, fn):
        self._fn = fn

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        response = self._fn(request)
        if isinstance(response, ReasoningResponse):
            return response
        if not isinstance(response, dict):
            raise TypeError("Reasoning callable must return dict or ReasoningResponse")
        return ReasoningResponse(data=response)


class RetrievalStrategyReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_variants: list[str] = Field(default_factory=list)
    token_budget: int | None = None
    max_retrieval_rounds: int | None = None


class RetrievalAnalysisReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sufficient: bool = False
    additional_queries: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class ContextPackagingReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority_claim_ids: list[str] = Field(default_factory=list)
    priority_source_ids: list[str] = Field(default_factory=list)
    drop_claim_ids: list[str] = Field(default_factory=list)
    drop_source_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class ActiveGraphLLMReasoningBackend:
    """Use an ActiveGraph ``LLMProvider`` for optional memory reasoning."""

    def __init__(
        self,
        provider,
        *,
        model: str,
        max_tokens: int = 1_500,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        from activegraph.llm import LLMMessage

        schema = {
            "query_classification": QueryAnalysis,
            "retrieval_strategy": RetrievalStrategyReasoning,
            "retrieval_analysis": RetrievalAnalysisReasoning,
            "context_packaging": ContextPackagingReasoning,
        }[request.stage]
        system = (
            "You are a memory retrieval control-plane component. Return only the "
            "requested structured output. Preserve source ids, distinguish event time "
            "from observation time, and never invent evidence."
        )
        user = json.dumps(
            {
                "stage": request.stage,
                "instructions": request.instructions,
                "payload": request.payload,
                "output_contract": request.output_contract,
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        response = self.provider.complete(
            system=system,
            messages=[LLMMessage(role="user", content=user)],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            output_schema=schema,
            timeout_seconds=self.timeout_seconds,
            tools=None,
            structured_output_mode="prompt",
        )
        parsed = response.parsed
        if isinstance(parsed, BaseModel):
            data = parsed.model_dump()
        elif isinstance(parsed, dict):
            data = parsed
        else:
            raise TypeError(f"ActiveGraph reasoning provider returned unsupported parsed output: {type(parsed)!r}")
        return ReasoningResponse(
            data=data,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=float(response.cost_usd),
            latency_ms=response.latency_seconds * 1000.0,
            cached=response.cache_hit,
            metadata=dict(response.provider_meta),
        )
