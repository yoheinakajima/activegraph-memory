"""Settings for the ActiveGraph Memory pack."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ActiveGraphMemorySettings(BaseModel):
    """Configuration for semantic and epistemic memory behavior."""

    enable_claim_extraction: bool = Field(
        default=False,
        description="Enable LLM-backed claim extraction. Disabled in v0.1.",
    )
    enable_temporal_resolution: bool = Field(
        default=False,
        description="Enable temporal normalization behavior. Disabled in v0.1.",
    )
    enable_conflict_detection: bool = Field(
        default=False,
        description="Enable conflict-detection behavior. Disabled in v0.1.",
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
