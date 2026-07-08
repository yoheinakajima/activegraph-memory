"""Coverage reporting helpers for memory answers."""

from __future__ import annotations

from collections.abc import Iterable

from .constants import QueryType
from .object_types import CoverageReport


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_coverage_report(
    *,
    query_id: str,
    searched_scopes: Iterable[str],
    not_searched_scopes: Iterable[str] = (),
    required_scopes: Iterable[str] = (),
    query_type: QueryType = "unknown",
    metadata: dict | None = None,
) -> CoverageReport:
    """Create a coverage report with conservative adequacy labels."""

    searched = _dedupe(searched_scopes)
    not_searched = _dedupe(not_searched_scopes)
    required = _dedupe(required_scopes)
    searched_set = set(searched)
    missing_required = [scope for scope in required if scope not in searched_set]

    if required:
        coverage_confidence = (len(required) - len(missing_required)) / len(required)
    else:
        denominator = len(searched) + len(not_searched)
        coverage_confidence = len(searched) / denominator if denominator else 0.0

    bounded = not not_searched and not missing_required and bool(searched)
    adequate_for: list[str] = []
    not_adequate_for: list[str] = []

    if coverage_confidence >= 0.8:
        adequate_for.append("likely_candidate_answer")
    if bounded:
        adequate_for.append("bounded_answer")
    else:
        not_adequate_for.extend(
            [
                "definitive_never",
                "definitive_latest",
                "exhaustive_aggregate",
            ]
        )

    if query_type in {"latest", "current", "final"} and not bounded:
        not_adequate_for.append("uncaveated_current_or_final_claim")
    if query_type == "negative_existence" and not bounded:
        not_adequate_for.append("unqualified_negative_answer")

    missing_data = [*missing_required, *not_searched]

    return CoverageReport(
        query_id=query_id,
        searched_scopes=searched,
        not_searched_scopes=not_searched,
        bounded=bounded,
        adequate_for=_dedupe(adequate_for),
        not_adequate_for=_dedupe(not_adequate_for),
        missing_data=_dedupe(missing_data),
        coverage_confidence=round(max(0.0, min(1.0, coverage_confidence)), 3),
        metadata=metadata or {},
    )


def is_adequate_for(report: CoverageReport, claim: str) -> bool:
    """Return whether a report explicitly supports a claim type."""

    return claim in set(report.adequate_for)
