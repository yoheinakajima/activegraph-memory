import pytest

from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import infer_query_type, plan_query
from activegraph_memory.settings import ActiveGraphMemorySettings


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the latest launch plan?", "latest"),
        ("What is currently scheduled?", "current"),
        ("Did we ever agree to exclusivity?", "negative_existence"),
        ("How many launch docs do we have?", "aggregate"),
        ("Why did we change the pricing after the calls?", "decision_reconstruction"),
        ("What tone does Yohei prefer for article feedback?", "preference"),
        (
            "Can you suggest some accessories that would complement my current photography setup?",
            "preference",
        ),
        ("What was true as of July 3?", "temporal"),
        ("What is the order of the museums I visited from earliest to latest?", "temporal"),
        ("Find where Sarah approved the budget.", "semantic_lookup"),
        ("What is the invoice date?", "lookup"),
    ],
)
def test_infer_query_type(query, expected):
    assert infer_query_type(query) == expected


def test_plan_for_latest_requires_coverage_and_freshness():
    query = MemoryQuery(query="What is the latest launch plan?")
    plan = plan_query(query, query_id="q_latest")

    assert plan.query_id == "q_latest"
    assert plan.requires_coverage is True
    assert plan.requires_freshness is True
    assert "memory_gateway_request" in plan.strategies
    assert "supersession_scan" in plan.strategies
    assert "stale_answer" in plan.risk_flags
    assert plan.metadata["query_type"] == "latest"


def test_plan_respects_gateway_setting():
    settings = ActiveGraphMemorySettings(enable_gateway_integration=False)
    plan = plan_query(
        MemoryQuery(query="Did we ever agree to exclusivity?"),
        query_id="q_no_gateway",
        settings=settings,
    )

    assert "memory_gateway_request" not in plan.strategies
    assert "gateway_request" not in plan.metadata
