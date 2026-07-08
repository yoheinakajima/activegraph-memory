"""Confidence scoring helpers for semantic memory answers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from .constants import CONFIDENCE_DIMENSIONS, EpistemicStatus
from .object_types import CoverageReport


class MemoryConfidence(BaseModel):
    """Confidence vector used by memory answers."""

    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_match: float = Field(default=0.0, ge=0.0, le=1.0)
    authority: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: float = Field(default=0.0, ge=0.0, le=1.0)

    def as_answer_confidence(self) -> dict[str, float]:
        """Return a plain dict suitable for MemoryAnswer.confidence."""

        return self.model_dump()


def confidence_vector(**values: float) -> MemoryConfidence:
    """Build a confidence vector, defaulting omitted dimensions to 0."""

    allowed = {key: values.get(key, 0.0) for key in CONFIDENCE_DIMENSIONS}
    return MemoryConfidence(**allowed)


def overall_confidence(
    confidence: MemoryConfidence | Mapping[str, float],
    *,
    requires_freshness: bool = False,
    requires_coverage: bool = False,
    requires_reasoning: bool = False,
) -> float:
    """Conservative aggregate confidence.

    Uses the minimum of required dimensions rather than an average, because a
    single weak dimension can break a memory answer.
    """

    data = (
        confidence.model_dump()
        if isinstance(confidence, MemoryConfidence)
        else dict(confidence)
    )
    required = [
        "relevance",
        "entity_match",
        "authority",
        "consistency",
        "extraction",
    ]
    if requires_freshness:
        required.append("freshness")
    if requires_coverage:
        required.append("coverage")
    if requires_reasoning:
        required.append("reasoning")
    values = [max(0.0, min(1.0, float(data.get(key, 0.0)))) for key in required]
    return round(min(values) if values else 0.0, 3)


def confidence_label(value: float) -> str:
    """Map a numeric confidence to a compact label."""

    if value >= 0.85:
        return "high"
    if value >= 0.65:
        return "moderate_high"
    if value >= 0.45:
        return "moderate"
    if value > 0.0:
        return "low"
    return "unknown"


def select_epistemic_status(
    confidence: MemoryConfidence | Mapping[str, float],
    *,
    found_evidence: bool = True,
    direct_support: bool = True,
    has_conflicts: bool = False,
    stale_risk: bool = False,
    coverage_report: CoverageReport | None = None,
    requires_freshness: bool = False,
    requires_coverage: bool = False,
    requires_reasoning: bool = False,
) -> EpistemicStatus:
    """Choose a conservative answer status from evidence quality signals."""

    if not found_evidence:
        if coverage_report and coverage_report.bounded:
            return "not_found_in_bounded_search"
        return "unanswerable_from_available_memory"
    if has_conflicts:
        return "conflicting_evidence"
    if stale_risk:
        return "stale_risk"
    if coverage_report and requires_coverage and not coverage_report.bounded:
        return "insufficient_coverage"

    value = overall_confidence(
        confidence,
        requires_freshness=requires_freshness,
        requires_coverage=requires_coverage,
        requires_reasoning=requires_reasoning,
    )
    if direct_support and value >= 0.65:
        return "directly_supported"
    if direct_support:
        return "likely_but_not_exhaustive"
    return "inferred"
