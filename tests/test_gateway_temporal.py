from activegraph_memory.gateway import (
    build_evidence_bundle_from_retrieval,
    build_memory_retrieval_request,
)
from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query
from activegraph_memory.temporal import detect_relative_duration, resolve_relative_duration


def test_gateway_request_matches_memory_gateway_shape():
    query = MemoryQuery(query="What is the latest launch plan?", subject_ref="user:yohei")
    plan = plan_query(query, query_id="q1")
    request = build_memory_retrieval_request(query, plan, backend_url="memory.db")

    assert request["query"] == query.query
    assert request["top_k"] == 10
    assert request["backend_url"] == "memory.db"
    assert request["metadata"]["query_id"] == "q1"
    assert request["metadata"]["subject_ref"] == "user:yohei"


def test_evidence_bundle_from_retrieval():
    bundle = build_evidence_bundle_from_retrieval(
        query_id="q1",
        retrieval={
            "id": "retrieval_1",
            "item_ids": ["mem_1", "mem_2"],
            "metadata": {"source_ids": ["src_1"], "claim_ids": ["claim_1"]},
        },
    )

    assert bundle.memory_item_ids == ["mem_1", "mem_2"]
    assert bundle.source_ids == ["src_1"]
    assert bundle.claim_ids == ["claim_1"]
    assert bundle.metadata["retrieval_id"] == "retrieval_1"


def test_relative_duration_resolution():
    assert detect_relative_duration("I have used it for 3 weeks now.")
    temporal = resolve_relative_duration(
        "I have used it for 3 weeks now.",
        anchor_time="2026-07-08",
    )

    assert temporal.resolution_method == "duration_start"
    assert temporal.resolved_start == "2026-06-17"
    assert temporal.resolved_end == "2026-07-08"
