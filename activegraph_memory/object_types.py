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
    extraction_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    belief_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
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


class MemoryEntity(StrictMemoryModel):
    """Canonical entity resolved from one or more source mentions."""

    entity_key: str
    canonical_name: str
    kind: str = "unknown"
    aliases: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEvent(StrictMemoryModel):
    """Canonical event with source mentions, modality, and event time."""

    event_key: str
    predicate: str
    summary: str
    entity_refs: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    modality: str = "actual"
    polarity: str = "affirmative"
    event_start: str | None = None
    event_end: str | None = None
    source_claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryState(StrictMemoryModel):
    """One version in a subject-property state history."""

    state_key: str
    value: str
    subject_ref: str
    predicate: str
    status: str = "active"
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str | None = None
    source_claim_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryPreference(StrictMemoryModel):
    """Scoped positive or negative preference evidence."""

    preference_key: str
    subject_ref: str
    text: str
    polarity: str
    scope_terms: list[str] = Field(default_factory=list)
    explicit: bool = False
    observed_at: str | None = None
    source_claim_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryListItem(StrictMemoryModel):
    """Position-preserving item extracted from a source list."""

    item_key: str
    list_key: str
    position: int = Field(ge=1)
    text: str
    role: str
    source_id: str
    observed_at: str | None = None


class MemoryQueryAnalysis(StrictMemoryModel):
    """Graph-visible executable query interpretation."""

    query_id: str
    query: str
    query_type: str
    operators: list[str] = Field(default_factory=list)
    operands: list[str] = Field(default_factory=list)
    entity_terms: list[str] = Field(default_factory=list)
    proof_requirements: list[str] = Field(default_factory=list)
    deterministic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalStage(StrictMemoryModel):
    """One measured stage in a memory retrieval run."""

    query_id: str
    stage: str
    implementation: str
    duration_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    candidates_in: int = Field(default=0, ge=0)
    candidates_out: int = Field(default=0, ge=0)
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryIngestionStage(StrictMemoryModel):
    """One replayable extraction/compilation run over immutable source turns."""

    stage_key: str
    operation: str = "fact_extraction"
    extractor: str
    model: str = ""
    source_ids: list[str] = Field(default_factory=list)
    fact_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryProof(StrictMemoryModel):
    """Proof obligations and execution result for a memory query."""

    query_id: str
    operation: str
    complete: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requirements: list[str] = Field(default_factory=list)
    satisfied: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    candidate_answer: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryBenchmark(StrictMemoryModel):
    """Aggregated profile latency, usage, cost, and optional quality metrics."""

    name: str
    profile: str
    cases: int = Field(default=0, ge=0)
    repetitions: int = Field(default=1, ge=1)
    latency_mean_ms: float = Field(default=0.0, ge=0.0)
    latency_p50_ms: float = Field(default=0.0, ge=0.0)
    latency_p95_ms: float = Field(default=0.0, ge=0.0)
    cold_latency_ms: float = Field(default=0.0, ge=0.0)
    warm_latency_mean_ms: float | None = Field(default=None, ge=0.0)
    mean_context_tokens: float = Field(default=0.0, ge=0.0)
    proof_complete_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficiency_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_retrieval_rounds: float = Field(default=0.0, ge=0.0)
    candidate_answer_render_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_calls: int = Field(default=0, ge=0)
    reasoning_cost_usd: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEmbedding(StrictMemoryModel):
    """Optional persistent vector for a compiled-memory field."""

    embedding_key: str
    subject_kind: str
    subject_key: str
    model: str
    text_sha256: str
    dimensions: int = Field(ge=1)
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySourceTurn(StrictMemoryModel):
    """Replayable projection of one immutable source turn."""

    turn_key: str
    session_id: str
    session_date: str
    session_idx: int = Field(ge=0)
    turn_idx: int = Field(ge=0)
    role: str
    content: str
    rendered_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryConflict(StrictMemoryModel):
    """An unresolved or resolved incompatibility between grounded claims."""

    conflict_key: str
    claim_ids: list[str] = Field(min_length=2)
    state_key: str | None = None
    reason: str
    status: str = "unresolved"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalAssessment(StrictMemoryModel):
    """A graph-visible sufficiency decision for one retrieval round."""

    query_id: str
    round_index: int = Field(ge=1)
    sufficient: bool = False
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: dict[str, float] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
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
    ObjectType(name="memory_entity", schema=MemoryEntity, description="A canonical memory entity."),
    ObjectType(name="memory_event", schema=MemoryEvent, description="A canonical source-grounded event."),
    ObjectType(name="memory_state", schema=MemoryState, description="A versioned memory state value."),
    ObjectType(name="memory_preference", schema=MemoryPreference, description="Scoped preference evidence."),
    ObjectType(name="memory_list_item", schema=MemoryListItem, description="A position-preserving source list item."),
    ObjectType(name="memory_query_analysis", schema=MemoryQueryAnalysis, description="An executable memory query analysis."),
    ObjectType(name="memory_ingestion_stage", schema=MemoryIngestionStage, description="A measured, replayable memory ingestion run."),
    ObjectType(name="memory_retrieval_stage", schema=MemoryRetrievalStage, description="A measured retrieval stage."),
    ObjectType(name="memory_proof", schema=MemoryProof, description="Proof obligations and evidence status."),
    ObjectType(name="memory_benchmark", schema=MemoryBenchmark, description="A memory profile benchmark report."),
    ObjectType(name="memory_embedding", schema=MemoryEmbedding, description="An optional persistent compiled-memory vector."),
    ObjectType(name="memory_source_turn", schema=MemorySourceTurn, description="A replayable immutable source-turn projection."),
    ObjectType(name="memory_conflict", schema=MemoryConflict, description="A source-grounded claim conflict and resolution state."),
    ObjectType(name="memory_retrieval_assessment", schema=MemoryRetrievalAssessment, description="A measured retrieval sufficiency decision."),
]


RELATION_TYPES = [
    RelationType(
        name="memory_supports",
        source_types=("memory_claim", "memory_conflict", "evidence_bundle", "memory_item", "source", "observation"),
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
    RelationType(
        name="memory_about",
        source_types=("memory_claim", "memory_event", "memory_state", "memory_preference", "memory_episode"),
        target_types=("memory_entity",),
        description="A memory object concerns a canonical entity.",
    ),
    RelationType(
        name="memory_grounded_in",
        source_types=("memory_source_turn", "memory_claim", "memory_entity", "memory_event", "memory_state", "memory_preference", "memory_list_item", "memory_conflict", "memory_ingestion_stage", "memory_proof"),
        target_types=("memory_source_turn", "memory_claim", "source", "observation", "memory_event"),
        description="A compiled memory object is grounded in an authoritative source object.",
    ),
    RelationType(
        name="memory_version_of",
        source_types=("memory_state", "memory_event", "memory_episode"),
        target_types=("memory_state", "memory_event", "memory_episode"),
        description="A memory object is another version of the same state or event.",
    ),
    RelationType(
        name="memory_stage_for",
        source_types=("memory_query_analysis", "retrieval_plan", "memory_retrieval_stage", "memory_retrieval_assessment", "memory_proof", "memory_benchmark"),
        target_types=("memory_query", "memory_evaluation"),
        description="A query artifact or benchmark stage belongs to a memory query or evaluation.",
    ),
    RelationType(
        name="memory_proves",
        source_types=("memory_proof",),
        target_types=("memory_answer", "evidence_bundle", "memory_query"),
        description="A proof supports a memory answer, evidence bundle, or query result.",
    ),
]


OBJECT_TYPE_NAMES = tuple(object_type.name for object_type in OBJECT_TYPES)
RELATION_TYPE_NAMES = tuple(relation_type.name for relation_type in RELATION_TYPES)
