"""Tools exposed by activegraph-memory."""

from __future__ import annotations

from typing import Any

from activegraph.packs import tool

from .gateway import build_memory_retrieval_request
from .object_types import MemoryQuery, RetrievalPlan
from .planner import plan_query
from .profiles import runtime_profiles
from .query_ir import analyze_query
from .settings import ActiveGraphMemorySettings


def plan_memory_query_fn(
    query: str,
    *,
    query_id: str = "tool_query",
    query_type: str = "unknown",
    subject_ref: str | None = None,
    time_anchor: str | None = None,
    top_k: int = 10,
    enable_gateway_integration: bool = True,
) -> dict[str, Any]:
    """Return a deterministic retrieval plan as a plain dict."""

    settings = ActiveGraphMemorySettings(
        enable_gateway_integration=enable_gateway_integration,
        default_top_k=top_k,
    )
    memory_query = MemoryQuery(
        query=query,
        query_type=query_type,  # type: ignore[arg-type]
        subject_ref=subject_ref,
        time_anchor=time_anchor,
        top_k=top_k,
    )
    plan = plan_query(memory_query, query_id=query_id, settings=settings)
    return plan.model_dump()


@tool(
    name="plan_memory_query",
    description=(
        "Classify a memory query and produce a deterministic retrieval plan. "
        "This does not call an LLM or retrieve data."
    ),
)
def plan_memory_query(
    query: str,
    query_id: str = "tool_query",
    query_type: str = "unknown",
    subject_ref: str | None = None,
    time_anchor: str | None = None,
    top_k: int = 10,
    enable_gateway_integration: bool = True,
) -> dict[str, Any]:
    """Registered ActiveGraph tool wrapper."""

    return plan_memory_query_fn(
        query=query,
        query_id=query_id,
        query_type=query_type,
        subject_ref=subject_ref,
        time_anchor=time_anchor,
        top_k=top_k,
        enable_gateway_integration=enable_gateway_integration,
    )


def build_gateway_request_fn(
    query: str,
    plan: dict[str, Any],
    *,
    top_k: int = 10,
    backend_url: str = ":memory:",
    min_score: float = 0.2,
) -> dict[str, Any]:
    """Build a memory_gateway retrieval request from plain query/plan data."""

    memory_query = MemoryQuery(query=query, top_k=top_k)
    retrieval_plan = RetrievalPlan.model_validate(plan)
    return build_memory_retrieval_request(
        memory_query,
        retrieval_plan,
        backend_url=backend_url,
        min_score=min_score,
    )


@tool(
    name="analyze_memory_query",
    description="Build the deterministic multi-operator query IR and proof obligations.",
)
def analyze_memory_query(
    query: str,
    question_date: str | None = None,
) -> dict[str, Any]:
    return analyze_query(query, question_date=question_date).model_dump()


@tool(
    name="list_memory_profiles",
    description="List built-in memory quality, latency, context, embedding, and reasoning options.",
)
def list_memory_profiles(args, ctx) -> list[dict[str, Any]]:
    return [profile.model_dump() for profile in runtime_profiles()]


TOOLS = [plan_memory_query, analyze_memory_query, list_memory_profiles]
