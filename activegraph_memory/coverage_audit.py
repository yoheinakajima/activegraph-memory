"""Measured source-to-extraction-to-selection coverage for one query."""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


class SourceCoverageAudit(BaseModel):
    """Conservative coverage ratios across the memory compilation pipeline."""

    model_config = ConfigDict(extra="forbid")

    candidate_source_ids: list[str] = Field(default_factory=list)
    extracted_source_ids: list[str] = Field(default_factory=list)
    compiled_source_ids: list[str] = Field(default_factory=list)
    selected_source_ids: list[str] = Field(default_factory=list)
    recovery_source_ids: list[str] = Field(default_factory=list)
    missing_extraction_ids: list[str] = Field(default_factory=list)
    missing_compilation_ids: list[str] = Field(default_factory=list)
    missing_selection_ids: list[str] = Field(default_factory=list)
    extraction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    compilation_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    selection_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    reader_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    complete: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def audit_source_coverage(
    candidate_source_ids: Iterable[str],
    *,
    extracted_source_ids: Iterable[str],
    compiled_source_ids: Iterable[str],
    selected_source_ids: Iterable[str],
    recovery_source_ids: Iterable[str] = (),
    minimum_ratio: float = 0.9,
    metadata: dict[str, Any] | None = None,
) -> SourceCoverageAudit:
    """Compare each pipeline plane against a bounded query-relevant source set."""

    candidates = set(candidate_source_ids)
    extracted = candidates & set(extracted_source_ids)
    compiled = candidates & set(compiled_source_ids)
    selected = candidates & set(selected_source_ids)
    recovery = candidates & set(recovery_source_ids)
    if not candidates:
        return SourceCoverageAudit(
            confidence=0.0,
            metadata={"bounded_candidate_set_empty": True, **(metadata or {})},
        )
    extraction_ratio = len(extracted) / len(candidates)
    compilation_ratio = len(compiled) / len(candidates)
    selection_ratio = len(selected) / len(candidates)
    reader_coverage_ratio = len(selected | recovery) / len(candidates)
    confidence = min(extraction_ratio, compilation_ratio, selection_ratio)
    return SourceCoverageAudit(
        candidate_source_ids=sorted(candidates),
        extracted_source_ids=sorted(extracted),
        compiled_source_ids=sorted(compiled),
        selected_source_ids=sorted(selected),
        recovery_source_ids=sorted(recovery),
        missing_extraction_ids=sorted(candidates - extracted),
        missing_compilation_ids=sorted(candidates - compiled),
        missing_selection_ids=sorted(candidates - selected),
        extraction_ratio=round(extraction_ratio, 4),
        compilation_ratio=round(compilation_ratio, 4),
        selection_ratio=round(selection_ratio, 4),
        reader_coverage_ratio=round(reader_coverage_ratio, 4),
        confidence=round(confidence, 4),
        complete=confidence >= minimum_ratio,
        metadata={"minimum_ratio": minimum_ratio, **(metadata or {})},
    )
