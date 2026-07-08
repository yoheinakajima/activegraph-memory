"""Helpers for composing activegraph-memory with memory_gateway."""

from __future__ import annotations

from typing import Any

from .object_types import EvidenceBundle, MemoryQuery, RetrievalPlan


def build_memory_retrieval_request(
    query: MemoryQuery,
    plan: RetrievalPlan,
    *,
    backend_url: str = ":memory:",
    min_score: float = 0.2,
    frame_id: str | None = None,
) -> dict[str, Any]:
    """Build memory_gateway-compatible MemoryRetrievalRequest data."""

    query_type = plan.metadata.get("query_type", query.query_type)
    return {
        "query": query.query,
        "top_k": query.top_k,
        "min_score": min_score,
        "category": None,
        "behavior_name": "activegraph_memory.gateway_adapter",
        "frame_id": frame_id,
        "backend_url": backend_url,
        "metadata": {
            "query_id": plan.query_id,
            "query_type": query_type,
            "plan_strategies": list(plan.strategies),
            "required_guarantees": list(plan.metadata.get("required_guarantees", [])),
            "subject_ref": query.subject_ref,
            "time_anchor": query.time_anchor,
        },
    }


def build_evidence_bundle_from_retrieval(
    *,
    query_id: str,
    retrieval: dict[str, Any],
    coverage_report_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceBundle:
    """Wrap memory_gateway retrieval data in this pack's evidence object."""

    item_ids = list(retrieval.get("item_ids") or [])
    retrieval_metadata = retrieval.get("metadata") or {}
    source_ids = list(retrieval_metadata.get("source_ids") or [])
    claim_ids = list(retrieval_metadata.get("claim_ids") or [])
    conflict_ids = list(retrieval_metadata.get("conflict_ids") or [])

    return EvidenceBundle(
        query_id=query_id,
        claim_ids=claim_ids,
        memory_item_ids=item_ids,
        source_ids=source_ids,
        coverage_report_id=coverage_report_id,
        conflict_ids=conflict_ids,
        metadata={
            "retrieval_id": retrieval.get("id") or retrieval.get("retrieval_id"),
            "results_count": retrieval.get("results_count", len(item_ids)),
            **(metadata or {}),
        },
    )
