"""Behaviors for activegraph-memory."""

from __future__ import annotations

from typing import Any

from activegraph.packs import behavior

from .object_types import MemoryQuery
from .planner import plan_query
from .query_ir import analyze_query
from .settings import ActiveGraphMemorySettings


def _event_object(event) -> dict[str, Any]:
    payload = getattr(event, "payload", None) or {}
    return payload.get("object", {}) or {}


@behavior(
    name="memory_query_planner",
    on=["object.created"],
    where={"object.type": "memory_query"},
    creates=["retrieval_plan"],
)
def memory_query_planner(
    event,
    graph,
    ctx,
    *,
    settings: ActiveGraphMemorySettings,
):
    """Create a deterministic retrieval_plan for each memory_query."""

    if not settings.enable_query_planning_behavior:
        return None

    obj = _event_object(event)
    query_id = obj.get("id")
    query_data = obj.get("data", {}) or {}
    if not query_id or not query_data.get("query"):
        return None

    query = MemoryQuery.model_validate(query_data)
    plan = plan_query(query, query_id=query_id, settings=settings)
    plan_obj = graph.add_object("retrieval_plan", plan.model_dump())

    try:
        graph.add_relation(plan_obj.id, query_id, "memory_retrieved_for")
    except Exception:
        # Relations are best-effort because external runtimes may load only a
        # subset of relation types while still wanting deterministic plans.
        pass

    return plan_obj


@behavior(
    name="memory_query_analyzer",
    on=["object.created"],
    where={"object.type": "memory_query"},
    creates=["memory_query_analysis"],
    priority=-10,
)
def memory_query_analyzer(event, graph, ctx, *, settings: ActiveGraphMemorySettings):
    """Create a graph-visible multi-operator query analysis."""

    if not settings.enable_query_analysis_behavior:
        return None
    obj = _event_object(event)
    query_id = obj.get("id")
    query_data = obj.get("data", {}) or {}
    if not query_id or not query_data.get("query"):
        return None
    query = MemoryQuery.model_validate(query_data)
    analysis = analyze_query(query)
    analysis_obj = graph.add_object(
        "memory_query_analysis",
        {
            "query_id": query_id,
            "query": analysis.query,
            "query_type": analysis.query_type,
            "operators": analysis.operators,
            "operands": analysis.operands,
            "entity_terms": analysis.entity_terms,
            "proof_requirements": analysis.proof_requirements,
            "deterministic_confidence": analysis.deterministic_confidence,
            "metadata": analysis.metadata,
        },
    )
    try:
        graph.add_relation(analysis_obj.id, query_id, "memory_stage_for")
    except Exception:
        pass
    return analysis_obj


from .shared_ingestion import SHARED_INGESTION_BEHAVIORS

BEHAVIORS = [
    memory_query_analyzer,
    memory_query_planner,
    *SHARED_INGESTION_BEHAVIORS,
]
