"""Deterministic sufficiency assessment and targeted retrieval expansion."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .compiler import MemoryIndex
from .executor import CompiledEvidence
from .query_ir import QueryAnalysis
from .ranking import RetrievalSignals


class RetrievalAssessment(BaseModel):
    """Why retrieval can stop or what the next round must recover."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(default=1, ge=1)
    sufficient: bool = False
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: dict[str, float] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def assess_retrieval(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    evidence: CompiledEvidence,
    signals: RetrievalSignals,
    *,
    round_index: int,
    min_confidence: float,
) -> RetrievalAssessment:
    """Conservatively assess evidence without treating vector score as proof."""

    satisfied = set(evidence.satisfied_requirements)
    required = set(evidence.proof_requirements)
    selected_claims = set(evidence.selected_claim_ids)
    conflicts = [
        conflict
        for conflict in index.compiled.conflicts
        if selected_claims & set(conflict.claim_ids)
    ]
    exhaustive = analysis.requires_exhaustive_coverage
    temporal = any(operator in {"order", "date_delta", "latest", "current", "previous"} for operator in analysis.operators)

    dimensions = {
        "candidate_recall": 1.0 if evidence.rows else 0.0,
        "entity_resolution": 1.0 if "entity_compatibility" in satisfied else (0.45 if evidence.rows else 0.0),
        "source_coverage": (
            1.0
            if "source_coverage" in satisfied or not exhaustive
            else 0.35 if evidence.rows else 0.0
        ),
        "temporal_resolution": (
            1.0
            if not temporal or "event_time_resolution" in satisfied or "state_history" in satisfied
            else 0.35 if evidence.rows else 0.0
        ),
        "consistency": 0.25 if conflicts else 1.0,
        "execution": max(0.0, min(1.0, float(evidence.confidence))),
    }
    required_dimensions = ["candidate_recall", "entity_resolution", "consistency", "execution"]
    if exhaustive:
        required_dimensions.append("source_coverage")
    if temporal:
        required_dimensions.append("temporal_resolution")
    overall = min(dimensions[name] for name in required_dimensions)

    missing = list(evidence.missing_requirements)
    reasons = []
    if not evidence.rows:
        reasons.append("no_typed_evidence_rows")
    if missing:
        reasons.append("missing_operator_requirements")
    if conflicts:
        reasons.append("selected_evidence_has_unresolved_conflicts")
    if evidence.confidence < min_confidence:
        reasons.append("execution_confidence_below_profile_threshold")

    sufficient = bool(
        evidence.rows
        and evidence.proof_complete
        and not conflicts
        and overall >= min_confidence
    )
    if sufficient:
        reasons.append("operator_proof_and_confidence_sufficient")

    next_queries = [] if sufficient else _targeted_queries(analysis, missing, dimensions)
    return RetrievalAssessment(
        round_index=round_index,
        sufficient=sufficient,
        overall_confidence=round(overall, 4),
        dimensions={key: round(value, 4) for key, value in dimensions.items()},
        missing_requirements=missing,
        conflict_ids=[conflict.conflict_id for conflict in conflicts],
        reasons=reasons,
        next_queries=next_queries,
        metadata={
            "proof_complete": evidence.proof_complete,
            "proof_requirements": sorted(required),
            "satisfied_requirements": sorted(satisfied),
            "signal_candidates": len(signals.claim_scores) + len(signals.turn_scores),
        },
    )


def _targeted_queries(
    analysis: QueryAnalysis,
    missing: list[str],
    dimensions: dict[str, float],
) -> list[str]:
    out: list[str] = []
    missing_set = set(missing)
    if missing_set & {"all_operands_found", "operand_coverage", "event_time_resolution"}:
        for operand in analysis.operands:
            out.append(f"{operand} date time occurrence")
    if missing_set & {"preference_scope", "constraint_coverage"}:
        out.append(f"user preferences dislikes constraints {' '.join(analysis.entity_terms)}")
    if missing_set & {"source_coverage", "bounded_candidate_set"} or dimensions["source_coverage"] < 0.8:
        scope = " ".join([*analysis.category_terms, *analysis.entity_terms])
        out.append(f"all matching {scope} {analysis.time_label}".strip())
    if "entity_compatibility" in missing_set or dimensions["entity_resolution"] < 0.8:
        out.append(" ".join(analysis.entity_terms or analysis.category_terms))
    if not out:
        out.extend(analysis.operands)
        out.append(" ".join([*analysis.entity_terms, *analysis.category_terms]))
    return _dedupe(value for value in out if value.strip() and value.strip().lower() != analysis.query.lower())


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out
