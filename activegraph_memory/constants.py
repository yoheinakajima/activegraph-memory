"""Shared constants for activegraph-memory."""

from __future__ import annotations

from typing import Literal


PACK_NAME = "activegraph_memory"
PACK_VERSION = "0.2.0"

ClaimKind = Literal[
    "fact",
    "preference",
    "instruction",
    "decision",
    "procedure",
    "summary",
    "unknown",
]

ClaimStatus = Literal[
    "active",
    "superseded",
    "contradicted",
    "uncertain",
    "archived",
    "deleted",
]

AuthorityLevel = Literal["low", "medium", "high", "unknown"]

QueryType = Literal[
    "lookup",
    "semantic_lookup",
    "latest",
    "current",
    "final",
    "negative_existence",
    "aggregate",
    "multi_hop",
    "temporal",
    "preference",
    "decision_reconstruction",
    "unknown",
]

EpistemicStatus = Literal[
    "directly_supported",
    "inferred",
    "likely_but_not_exhaustive",
    "conflicting_evidence",
    "stale_risk",
    "insufficient_coverage",
    "not_found_in_bounded_search",
    "unanswerable_from_available_memory",
]

TemporalResolutionMethod = Literal[
    "explicit",
    "relative_to_query",
    "relative_to_source",
    "duration_start",
    "unresolved",
]

QuantityExactness = Literal["exact", "approximate", "range", "unknown"]

EvaluationJudgment = Literal[
    "helpful",
    "unhelpful",
    "incorrect",
    "unsupported",
    "partially_helpful",
    "unknown",
]


QUERY_TYPES: tuple[str, ...] = (
    "lookup",
    "semantic_lookup",
    "latest",
    "current",
    "final",
    "negative_existence",
    "aggregate",
    "multi_hop",
    "temporal",
    "preference",
    "decision_reconstruction",
    "unknown",
)

CONFIDENCE_DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "entity_match",
    "authority",
    "freshness",
    "coverage",
    "consistency",
    "extraction",
    "reasoning",
)
