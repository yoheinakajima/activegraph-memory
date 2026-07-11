"""Extraction-run coverage read from ingestion stages (ADR 0026 step 6).

Proof completeness for count/sum/temporal/absence must know which sources
an *extraction run* actually processed, not merely which sources produced
a surviving claim. ``memory_ingestion_stage`` records carry that: each
stage's ``source_ids`` is the exact source set one extraction run covered.

This module reads those stages off a compiled ``MemoryIndex`` (they live
in ``index.metadata["ingestion_runs"]``, materialized/replayed as
``memory_ingestion_stage`` objects). When no ingestion stage is present —
the caller supplied claims directly, no extraction run was recorded — the
signal is ``None`` and every coverage audit behaves exactly as it did
before extraction-run coverage existed.
"""

from __future__ import annotations

from typing import Any


def ingestion_stages(index: Any) -> list[dict[str, Any]]:
    """The recorded extraction/ingestion stages for this index, if any."""
    metadata = getattr(index, "metadata", None) or {}
    runs = metadata.get("ingestion_runs") or []
    return [run for run in runs if isinstance(run, dict)]


def extraction_run_source_ids(index: Any) -> set[str] | None:
    """Union of source ids every recorded extraction run processed.

    ``None`` when no ingestion stage exists (claims supplied directly), so
    ``audit_source_coverage`` falls back to its prior behavior. A set —
    possibly empty — once any extraction run has been recorded.
    """
    stages = ingestion_stages(index)
    if not stages:
        return None
    covered: set[str] = set()
    for stage in stages:
        for source_id in stage.get("source_ids") or []:
            covered.add(str(source_id))
    return covered


def extraction_runs_cover(index: Any, source_ids: Any) -> bool:
    """Whether every given source was processed by some extraction run.

    Vacuously True when no extraction run is recorded (claims supplied
    directly), so operators that consult this keep their prior behavior
    until ingestion stages are present.
    """
    covered = extraction_run_source_ids(index)
    if covered is None:
        return True
    return all(str(source_id) in covered for source_id in source_ids)


__all__ = [
    "extraction_run_source_ids",
    "extraction_runs_cover",
    "ingestion_stages",
]
