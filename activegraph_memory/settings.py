"""Settings for the ActiveGraph Memory pack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ActiveGraphMemorySettings(BaseModel):
    """Configuration for semantic and epistemic memory behavior."""

    enable_claim_extraction: bool = Field(
        default=False,
        description="Reserved for a future graph-visible claim extraction behavior.",
    )
    enable_temporal_resolution: bool = Field(
        default=False,
        description="Reserved for a future graph-visible temporal normalization behavior.",
    )
    enable_conflict_detection: bool = Field(
        default=False,
        description="Reserved for a future graph-visible conflict detection behavior.",
    )
    enable_gateway_integration: bool = Field(
        default=True,
        description=(
            "Plan retrieval through memory_gateway-compatible request data "
            "instead of hidden backend calls."
        ),
    )
    enable_query_planning_behavior: bool = Field(
        default=True,
        description=(
            "Create retrieval_plan objects when memory_query objects are created."
        ),
    )
    default_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default maximum number of memory candidates to retrieve.",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for answer helper utilities.",
    )
    strict_coverage_for_latest: bool = Field(
        default=True,
        description=(
            "Require coverage-sensitive status for latest/current/final questions."
        ),
    )
    default_target_sources: list[str] = Field(
        default_factory=lambda: [
            "memory_claims",
            "memory_items",
            "sources",
            "observations",
        ],
        description="Default source families used by deterministic plans.",
    )
    runtime_profile: str = Field(
        default="balanced",
        pattern=r"^(fast|balanced|quality|max_quality)$",
        description="Default retrieval quality, latency, and cost profile.",
    )
    enable_query_analysis_behavior: bool = Field(
        default=True,
        description="Create graph-visible multi-operator query analyses.",
    )
    query_classification_reasoning: str | None = Field(
        default=None,
        pattern=r"^(off|fallback|always)$",
        description="Optional override for the profile's classification reasoning policy.",
    )
    retrieval_strategy_reasoning: str | None = Field(
        default=None,
        pattern=r"^(off|fallback|always)$",
        description="Optional override for strategy reasoning.",
    )
    retrieval_analysis_reasoning: str | None = Field(
        default=None,
        pattern=r"^(off|fallback|always)$",
        description="Optional override for evidence-sufficiency reasoning.",
    )
    context_packaging_reasoning: str | None = Field(
        default=None,
        pattern=r"^(off|fallback|always)$",
        description="Optional override for context-packaging reasoning.",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model passed to the runtime embedding-provider seam.",
    )
    embedding_cost_per_million_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional caller-supplied price used only for telemetry estimates.",
    )
