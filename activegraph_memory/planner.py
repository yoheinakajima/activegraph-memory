"""Deterministic memory query classification and retrieval planning."""

from __future__ import annotations

import re
from typing import Iterable, cast

from .constants import QueryType
from .object_types import MemoryQuery, RetrievalPlan
from .settings import ActiveGraphMemorySettings


_LATEST_PATTERNS = (
    r"\blatest\b",
    r"\bmost recent\b",
    r"\bnewest\b",
    r"\bcurrent\b",
    r"\bcurrently\b",
    r"\bas of now\b",
    r"\bup[- ]?to[- ]?date\b",
)
_FINAL_PATTERNS = (
    r"\bfinal\b",
    r"\bapproved\b",
    r"\bsigned\b",
    r"\bagreed\b",
    r"\bdecided\b",
)
_NEGATIVE_PATTERNS = (
    r"\bdid (we|i|you|they) ever\b",
    r"\bhave (we|i|you|they) ever\b",
    r"\bany evidence\b",
    r"\bno evidence\b",
    r"\bnever\b",
    r"\bdid (we|i|you|they) already\b",
    r"\bhave (we|i|you|they) already\b",
)
_AGGREGATE_PATTERNS = (
    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\bsum\b",
    r"\baverage\b",
    r"\blist all\b",
    r"\ball of\b",
)
_TEMPORAL_QUANTITY_PATTERNS = (
    r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+ago\b",
    r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+(?:had\s+)?passed\b",
    r"\bhow many\s+(?:days?|weeks?|months?|years?)\s+did it take\b",
    r"\bhow long\b.*\b(?:ago|since|after|before)\b",
)
_TEMPORAL_SEQUENCE_PATTERNS = (
    r"\border(?:ed|ing)?\b.*\b(?:earliest|latest|first|last|chronological)\b",
    r"\b(?:earliest|oldest)\s+to\s+(?:latest|newest)\b",
    r"\b(?:latest|newest)\s+to\s+(?:earliest|oldest)\b",
    r"\bchronological(?:ly)?\b",
    r"\btimeline\b",
)
_TEMPORAL_PATTERNS = (
    r"\bas of\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bsince\b",
    r"\btimeline\b",
    r"\bhistory\b",
    r"\bwhen\b",
    r"\bwhat changed\b",
    r"\bfor \d+ (day|days|week|weeks|month|months|year|years)\b",
)
_PREFERENCE_PATTERNS = (
    r"\badvice\b",
    r"\bany tips\b",
    r"\btips?\b.*\b(better|improve|improving|results)\b",
    r"\bcan you suggest\b",
    r"\bsuggest (?:a|an|some|new|better)\b",
    r"\brecommend (?:a|an|some|new|better)\b",
    r"\brecommendation\b",
    r"\bprefer\b",
    r"\bprefers\b",
    r"\bpreference\b",
    r"\blikes\b",
    r"\bstyle\b",
    r"\btone\b",
    r"\binstruction\b",
)
_DECISION_RECONSTRUCTION_PATTERNS = (
    r"\bwhy did\b",
    r"\bwhy was\b",
    r"\bhow did (we|i|you|they) decide\b",
    r"\bwhat led to\b",
    r"\brationale\b",
    r"\bdecision\b",
)
_SEMANTIC_LOOKUP_PATTERNS = (
    r"\bfind\b",
    r"\bsearch\b",
    r"\bwhere did\b",
    r"\bwhere is\b",
    r"\bshow me\b",
    r"\bremind me\b",
)
_LOOKUP_PATTERNS = (
    r"\bwhat is\b",
    r"\bwhat was\b",
    r"\bwho is\b",
    r"\bwho was\b",
    r"\bwhere is\b",
    r"\bwhere was\b",
)


_GUARANTEES_BY_QUERY_TYPE: dict[str, tuple[str, ...]] = {
    "lookup": ("local_evidence", "source_match"),
    "semantic_lookup": ("local_evidence", "entity_disambiguation"),
    "latest": (
        "freshness",
        "chronological_ordering",
        "supersession_check",
        "bounded_coverage",
        "source_authority",
    ),
    "current": (
        "freshness",
        "chronological_ordering",
        "supersession_check",
        "bounded_coverage",
        "source_authority",
    ),
    "final": (
        "chronological_ordering",
        "supersession_check",
        "source_authority",
        "conflict_check",
    ),
    "negative_existence": (
        "bounded_coverage",
        "exact_term_search",
        "semantic_equivalent_search",
        "source_authority",
    ),
    "aggregate": ("exhaustive_candidate_set", "bounded_coverage"),
    "multi_hop": ("reasoning_chain", "local_evidence", "chronological_ordering"),
    "temporal": ("temporal_resolution", "chronological_ordering"),
    "preference": ("scope", "freshness", "source_authority"),
    "decision_reconstruction": (
        "chronological_ordering",
        "source_authority",
        "evidence_chain",
        "explicit_vs_inferred",
    ),
    "unknown": ("local_evidence",),
}

_STRATEGIES_BY_QUERY_TYPE: dict[str, tuple[str, ...]] = {
    "lookup": ("exact_source_lookup", "claim_graph_search"),
    "semantic_lookup": ("semantic_search", "keyword_search", "entity_filter"),
    "latest": (
        "semantic_search",
        "version_scan",
        "temporal_ordering",
        "supersession_scan",
        "coverage_check",
        "authority_rank",
    ),
    "current": (
        "semantic_search",
        "temporal_ordering",
        "supersession_scan",
        "conflict_scan",
        "coverage_check",
    ),
    "final": (
        "candidate_gathering",
        "status_classification",
        "authority_rank",
        "supersession_scan",
        "conflict_scan",
    ),
    "negative_existence": (
        "exact_keyword_search",
        "semantic_search",
        "authoritative_source_scan",
        "coverage_check",
    ),
    "aggregate": (
        "structured_filter",
        "exhaustive_candidate_gathering",
        "dedupe",
        "coverage_check",
    ),
    "multi_hop": (
        "semantic_search",
        "graph_expansion",
        "chronological_reconstruction",
        "evidence_chain",
    ),
    "temporal": (
        "temporal_resolution",
        "time_index_search",
        "chronological_reconstruction",
    ),
    "preference": (
        "subject_scoped_search",
        "recency_weighting",
        "scope_check",
        "supersession_scan",
    ),
    "decision_reconstruction": (
        "candidate_gathering",
        "chronological_reconstruction",
        "authority_rank",
        "evidence_chain",
        "explicit_vs_inferred_check",
    ),
    "unknown": ("semantic_search", "claim_graph_search"),
}

_RISK_FLAGS_BY_QUERY_TYPE: dict[str, tuple[str, ...]] = {
    "lookup": ("source_mismatch",),
    "semantic_lookup": ("predicate_mismatch", "entity_confusion"),
    "latest": (
        "stale_answer",
        "incomplete_coverage",
        "superseded_evidence",
        "draft_vs_final_confusion",
    ),
    "current": (
        "stale_answer",
        "incomplete_coverage",
        "superseded_evidence",
        "conflicting_evidence",
    ),
    "final": ("draft_vs_final_confusion", "conflicting_evidence"),
    "negative_existence": (
        "false_negative",
        "unsearched_corpus",
        "terminology_mismatch",
    ),
    "aggregate": ("partial_count", "duplicate_entities", "unsearched_corpus"),
    "multi_hop": ("inferred_not_stated", "missing_link"),
    "temporal": ("event_time_vs_observed_time", "relative_date_ambiguity"),
    "preference": ("overgeneralized_preference", "stale_preference"),
    "decision_reconstruction": (
        "inferred_not_stated",
        "draft_vs_final_confusion",
        "missing_decision_source",
    ),
    "unknown": ("underspecified_query",),
}


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def infer_query_type(query: str) -> QueryType:
    """Infer a memory query class from natural language."""

    text = f" {query.lower().strip()} "
    if not text.strip():
        return "unknown"
    if _matches_any(text, _NEGATIVE_PATTERNS):
        return "negative_existence"
    if _matches_any(text, _TEMPORAL_QUANTITY_PATTERNS):
        return "temporal"
    if _matches_any(text, _TEMPORAL_SEQUENCE_PATTERNS):
        return "temporal"
    if _matches_any(text, _PREFERENCE_PATTERNS):
        return "preference"
    if _matches_any(text, _AGGREGATE_PATTERNS):
        return "aggregate"
    if _matches_any(text, _LATEST_PATTERNS):
        if re.search(r"\bcurrent\b|\bcurrently\b|\bas of now\b", text):
            return "current"
        return "latest"
    if _matches_any(text, _SEMANTIC_LOOKUP_PATTERNS):
        return "semantic_lookup"
    if _matches_any(text, _FINAL_PATTERNS):
        return "final"
    if _matches_any(text, _DECISION_RECONSTRUCTION_PATTERNS):
        return "decision_reconstruction"
    if _matches_any(text, _TEMPORAL_PATTERNS):
        return "temporal"
    if re.search(r"\bwhy\b|\bhow\b|\bcompare\b|\brelationship between\b", text):
        return "multi_hop"
    if _matches_any(text, _LOOKUP_PATTERNS):
        return "lookup"
    return "semantic_lookup"


def required_guarantees_for(query_type: QueryType) -> list[str]:
    """Return default guarantees for a query class."""

    return list(_GUARANTEES_BY_QUERY_TYPE.get(query_type, _GUARANTEES_BY_QUERY_TYPE["unknown"]))


def strategies_for(query_type: QueryType, *, gateway_enabled: bool = True) -> list[str]:
    """Return deterministic retrieval strategies for a query class."""

    base = list(_STRATEGIES_BY_QUERY_TYPE.get(query_type, _STRATEGIES_BY_QUERY_TYPE["unknown"]))
    if gateway_enabled:
        base.insert(0, "memory_gateway_request")
    return _dedupe(base)


def risk_flags_for(query_type: QueryType) -> list[str]:
    """Return default risk flags for a query class."""

    return list(_RISK_FLAGS_BY_QUERY_TYPE.get(query_type, _RISK_FLAGS_BY_QUERY_TYPE["unknown"]))


def requires_coverage(query_type: QueryType, guarantees: Iterable[str]) -> bool:
    """Whether the plan needs a coverage report before a confident answer."""

    guarantee_set = set(guarantees)
    return query_type in {
        "latest",
        "current",
        "negative_existence",
        "aggregate",
        "final",
    } or bool({"bounded_coverage", "exhaustive_candidate_set"} & guarantee_set)


def requires_freshness(query_type: QueryType, guarantees: Iterable[str]) -> bool:
    """Whether stale evidence can break the answer."""

    return query_type in {"latest", "current", "preference"} or "freshness" in set(guarantees)


def plan_steps_for(query_type: QueryType) -> list[str]:
    """Human-readable steps for the plan metadata."""

    steps_by_type: dict[str, tuple[str, ...]] = {
        "lookup": (
            "Identify the likely source or claim.",
            "Retrieve exact evidence and cite it.",
            "Answer only what the evidence directly supports.",
        ),
        "semantic_lookup": (
            "Retrieve semantically related candidates.",
            "Filter by entity and predicate match.",
            "Return evidence-ranked candidates.",
        ),
        "latest": (
            "Gather all candidate artifacts or claims.",
            "Resolve aliases, versions, and duplicates.",
            "Order candidates by meaningful event, modified, approved, or effective time.",
            "Check supersession and conflict edges.",
            "Report coverage before claiming latest.",
        ),
        "current": (
            "Gather current and historical candidates.",
            "Filter out superseded or expired claims.",
            "Check for later contradictions.",
            "Report stale or incomplete coverage risk.",
        ),
        "final": (
            "Gather candidate decisions or artifacts.",
            "Classify draft, reviewed, approved, signed, or final status.",
            "Rank authoritative sources above drafts and summaries.",
            "Check later supersession or contradiction.",
        ),
        "negative_existence": (
            "Define the bounded corpus to search.",
            "Search exact terms and semantic equivalents.",
            "Inspect authoritative sources first.",
            "Answer as a bounded not-found claim unless coverage is complete.",
        ),
        "aggregate": (
            "Define the population and filters.",
            "Gather an exhaustive candidate set when possible.",
            "Deduplicate by stable ids or entity refs.",
            "Report whether the count is bounded.",
        ),
        "multi_hop": (
            "Retrieve first-hop evidence.",
            "Expand through graph neighbors.",
            "Build an evidence chain and mark inferred links.",
            "Separate direct statements from reasoning.",
        ),
        "temporal": (
            "Resolve explicit and relative temporal references.",
            "Separate event time from observed time.",
            "Reconstruct the relevant chronology.",
        ),
        "preference": (
            "Search subject-scoped guidance and behavior evidence.",
            "Prefer recent explicit statements over old inferred preferences.",
            "Check scope so the preference is not overgeneralized.",
        ),
        "decision_reconstruction": (
            "Find decision candidates and related discussions.",
            "Order evidence chronologically.",
            "Rank explicit approvals and decision logs above drafts.",
            "Separate stated rationale from inferred rationale.",
        ),
        "unknown": (
            "Retrieve candidate evidence.",
            "Assess whether more specific planning is required.",
        ),
    }
    return list(steps_by_type.get(query_type, steps_by_type["unknown"]))


def plan_query(
    query: MemoryQuery | str,
    *,
    query_id: str = "query",
    settings: ActiveGraphMemorySettings | None = None,
) -> RetrievalPlan:
    """Build a deterministic retrieval plan for a memory query."""

    settings = settings or ActiveGraphMemorySettings()
    memory_query = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query)
    query_type = memory_query.query_type
    if query_type == "unknown":
        query_type = infer_query_type(memory_query.query)
    query_type = cast(QueryType, query_type)

    guarantees = _dedupe(
        [
            *required_guarantees_for(query_type),
            *memory_query.required_guarantees,
        ]
    )
    target_sources = _dedupe(
        [
            *settings.default_target_sources,
            *memory_query.metadata.get("target_sources", []),
        ]
    )

    metadata = {
        "query": memory_query.query,
        "query_type": query_type,
        "subject_ref": memory_query.subject_ref,
        "time_anchor": memory_query.time_anchor,
        "required_guarantees": guarantees,
        "top_k": memory_query.top_k or settings.default_top_k,
        "planner": "activegraph_memory.deterministic_v1",
    }
    if settings.enable_gateway_integration:
        metadata["gateway_request"] = {
            "query": memory_query.query,
            "top_k": memory_query.top_k or settings.default_top_k,
            "category": query_type,
            "behavior_name": "activegraph_memory.memory_query_planner",
            "metadata": {
                "query_id": query_id,
                "query_type": query_type,
                "required_guarantees": guarantees,
                "subject_ref": memory_query.subject_ref,
            },
        }

    return RetrievalPlan(
        query_id=query_id,
        strategies=strategies_for(
            query_type,
            gateway_enabled=settings.enable_gateway_integration,
        ),
        target_sources=target_sources,
        requires_coverage=requires_coverage(query_type, guarantees),
        requires_freshness=requires_freshness(query_type, guarantees),
        risk_flags=risk_flags_for(query_type),
        steps=plan_steps_for(query_type),
        metadata=metadata,
    )
