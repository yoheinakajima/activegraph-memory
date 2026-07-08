"""Deterministic fixture runner for activegraph-memory."""

from __future__ import annotations

from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query


def run() -> dict:
    """Return a small deterministic fixture result without network or keys."""

    query = MemoryQuery(query="What is the latest launch plan?")
    plan = plan_query(query, query_id="fixture_query")
    return {
        "query_type": plan.metadata["query_type"],
        "requires_coverage": plan.requires_coverage,
        "strategies": plan.strategies,
    }


if __name__ == "__main__":
    print(run())
