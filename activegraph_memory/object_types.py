"""Object and relation types for activegraph-memory."""

from __future__ import annotations

from typing import Any

from activegraph.packs import ObjectType, RelationType
from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    AuthorityLevel,
    ClaimKind,
    ClaimStatus,
    EpistemicStatus,
    EvaluationJudgment,
    QuantityExactness,
    QueryType,
    TemporalResolutionMethod,
)


class StrictMemoryModel(BaseModel):
    """Base schema with explicit fields only."""

    model_config = ConfigDict(extra="forbid")


class MemoryClaim(StrictMemoryModel):
    """A source-grounded claim or belief managed by the semantic memory layer."""

    text: str = Field(description="Human-readable claim text.")
    claim_kind: ClaimKind = Field(
        default="unknown",
        description="Kind of claim represented by the text.",
    )
    subject_ref: str | None = Field(
        default=None,
        description="Opaque entity or principal reference the claim is about.",
    )
    scope: list[str] = Field(
        default_factory=list,
        description="Contexts where this claim should be applied.",
    )
    status: ClaimStatus = Field(
        default="active",
        description="Belief lifecycle state.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Extraction or belief confidence.",
    )
    authority: AuthorityLevel = Field(
        default="unknown",
        description="Contextual authority of the supporting source.",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Core source ids that ground the claim.",
    )
    observation_ids: list[str] = Field(
        default_factory=list,
        description="Core observation ids that led to this claim.",
    )
    valid_from: str | None = Field(
        default=None,
        description="ISO 8601 date/datetime when the claim became valid.",
    )
    valid_until: str | None = Field(
        default=None,
        description="ISO 8601 date/datetime when the claim stopped being valid.",
    )
    observed_at: str | None = Field(
        default=None,
        description="ISO 8601 date/datetime when the system observed the claim.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEpisode(StrictMemoryModel):
    """A coherent bundle of events, claims, and sources."""

    summary: str = Field(description="Short episode summary.")
    event_start: str | None = Field(default=None)
    event_end: str | None = Field(default=None)
    source_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryQuery(StrictMemoryModel):
    """A graph-visible memory question or recall request."""

    query: str = Field(description="Natural language memory query.")
    query_type: QueryType = Field(
        default="unknown",
        description="Question class. Use unknown to let deterministic planning infer it.",
    )
    subject_ref: str | None = Field(default=None)
    time_anchor: str | None = Field(
        default=None,
        description="Date/datetime that should anchor temporal interpretation.",
    )
    required_guarantees: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalPlan(StrictMemoryModel):
    """A deterministic strategy for answering a memory query."""

    query_id: str = Field(description="ID of the memory_query this plan serves.")
    strategies: list[str] = Field(default_factory=list)
    target_sources: list[str] = Field(default_factory=list)
    requires_coverage: bool = Field(default=False)
    requires_freshness: bool = Field(default=False)
    risk_flags: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageReport(StrictMemoryModel):
    """What was searched, what was not, and what the answer can claim."""

    query_id: str = Field(description="ID of the memory_query being covered.")
    searched_scopes: list[str] = Field(default_factory=list)
    not_searched_scopes: list[str] = Field(default_factory=list)
    bounded: bool = Field(default=False)
    adequate_for: list[str] = Field(default_factory=list)
    not_adequate_for: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    coverage_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(StrictMemoryModel):
    """Candidate evidence gathered for a query."""

    query_id: str = Field(description="ID of the memory_query this evidence serves.")
    claim_ids: list[str] = Field(default_factory=list)
    memory_item_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    coverage_report_id: str | None = Field(default=None)
    conflict_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryAnswer(StrictMemoryModel):
    """An answer with evidence, caveats, and epistemic status."""

    query_id: str = Field(description="ID of the memory_query this answer addresses.")
    answer: str = Field(description="Natural language answer.")
    epistemic_status: EpistemicStatus = Field(default="unanswerable_from_available_memory")
    confidence: dict[str, float] = Field(default_factory=dict)
    evidence_bundle_id: str | None = Field(default=None)
    caveats: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalRef(StrictMemoryModel):
    """A temporal expression and its normalized interpretation."""

    text: str = Field(description="Temporal expression from source or query text.")
    resolved_start: str | None = Field(default=None)
    resolved_end: str | None = Field(default=None)
    anchor_time: str | None = Field(default=None)
    resolution_method: TemporalResolutionMethod = Field(default="unresolved")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantityClaim(StrictMemoryModel):
    """A numeric claim with owner, unit, and exactness metadata."""

    owner_ref: str | None = Field(default=None)
    property_name: str = Field(description="The measured property.")
    value: float | None = Field(default=None)
    unit: str | None = Field(default=None)
    exactness: QuantityExactness = Field(default="unknown")
    source_text: str = Field(description="Original text supporting the quantity.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryPolicy(StrictMemoryModel):
    """Policy knobs governing memory planning, retrieval, or answering."""

    name: str = Field(description="Stable policy name.")
    description: str = Field(default="")
    settings: dict[str, Any] = Field(default_factory=dict)
    active: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEvaluation(StrictMemoryModel):
    """Evaluation of a memory query, answer, or retrieval process."""

    answer_id: str | None = Field(default=None)
    query_id: str | None = Field(default=None)
    judgment: EvaluationJudgment = Field(default="unknown")
    rationale: str = Field(default="")
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


OBJECT_TYPES = [
    ObjectType(
        name="memory_claim",
        schema=MemoryClaim,
        description=(
            "A source-grounded semantic memory claim with scope, confidence, "
            "authority, temporal validity, and lifecycle status."
        ),
    ),
    ObjectType(
        name="memory_episode",
        schema=MemoryEpisode,
        description="A coherent bundle of sources, claims, entities, and events.",
    ),
    ObjectType(
        name="memory_query",
        schema=MemoryQuery,
        description="A graph-visible memory question or recall request.",
    ),
    ObjectType(
        name="retrieval_plan",
        schema=RetrievalPlan,
        description="A planned retrieval strategy for a memory query.",
    ),
    ObjectType(
        name="coverage_report",
        schema=CoverageReport,
        description="A report of searched and unsearched memory scopes.",
    ),
    ObjectType(
        name="evidence_bundle",
        schema=EvidenceBundle,
        description="A bundle of claims, memory items, and sources gathered for a query.",
    ),
    ObjectType(
        name="memory_answer",
        schema=MemoryAnswer,
        description="An answer with evidence, confidence dimensions, and caveats.",
    ),
    ObjectType(
        name="temporal_ref",
        schema=TemporalRef,
        description="A temporal expression and its normalized interpretation.",
    ),
    ObjectType(
        name="quantity_claim",
        schema=QuantityClaim,
        description="A numeric claim with owner, value, unit, and exactness.",
    ),
    ObjectType(
        name="memory_policy",
        schema=MemoryPolicy,
        description="A policy governing memory planning, retrieval, or answering.",
    ),
    ObjectType(
        name="memory_evaluation",
        schema=MemoryEvaluation,
        description="A judgment about a memory query, answer, or retrieval process.",
    ),
]


RELATION_TYPES = [
    RelationType(
        name="memory_supports",
        source_types=("memory_claim", "evidence_bundle", "memory_item", "source", "observation"),
        target_types=("memory_claim", "memory_answer"),
        description="Evidence or one claim supports another claim or answer.",
    ),
    RelationType(
        name="memory_contradicts",
        source_types=("memory_claim", "evidence_bundle", "memory_item", "source", "observation"),
        target_types=("memory_claim", "memory_answer"),
        description="Evidence or one claim contradicts another claim or answer.",
    ),
    RelationType(
        name="memory_supersedes",
        source_types=("memory_claim", "memory_episode", "memory_item"),
        target_types=("memory_claim", "memory_episode", "memory_item"),
        description="A newer or more authoritative memory replaces an older one.",
    ),
    RelationType(
        name="memory_has_temporal_ref",
        source_types=("memory_claim", "memory_query", "retrieval_plan", "memory_answer"),
        target_types=("temporal_ref",),
        description="A memory object has a temporal reference.",
    ),
    RelationType(
        name="memory_has_quantity",
        source_types=("memory_claim", "memory_answer"),
        target_types=("quantity_claim",),
        description="A memory claim or answer contains a numeric claim.",
    ),
    RelationType(
        name="memory_retrieved_for",
        source_types=(
            "memory_claim",
            "memory_episode",
            "memory_item",
            "retrieval_plan",
            "evidence_bundle",
        ),
        target_types=("memory_query",),
        description="A memory object was retrieved for a memory query.",
    ),
    RelationType(
        name="memory_used_as_evidence",
        source_types=("memory_claim", "memory_episode", "memory_item", "source", "observation"),
        target_types=("evidence_bundle", "memory_answer"),
        description="A source, claim, or item was used as evidence.",
    ),
    RelationType(
        name="memory_has_coverage",
        source_types=("memory_query", "retrieval_plan", "evidence_bundle", "memory_answer"),
        target_types=("coverage_report",),
        description="A query, plan, evidence bundle, or answer has a coverage report.",
    ),
    RelationType(
        name="memory_governed_by_policy",
        source_types=("memory_query", "retrieval_plan", "memory_answer", "memory_claim"),
        target_types=("memory_policy",),
        description="A memory object is governed by a policy.",
    ),
]


OBJECT_TYPE_NAMES = tuple(object_type.name for object_type in OBJECT_TYPES)
RELATION_TYPE_NAMES = tuple(relation_type.name for relation_type in RELATION_TYPES)
