import pytest
from pydantic import ValidationError

from activegraph_memory.object_types import (
    CoverageReport,
    EvidenceBundle,
    MemoryAnswer,
    MemoryClaim,
    MemoryEpisode,
    MemoryEvaluation,
    MemoryPolicy,
    MemoryQuery,
    QuantityClaim,
    RetrievalPlan,
    TemporalRef,
)


def test_minimal_schema_instances_validate():
    MemoryClaim(text="Yohei prefers honest technical feedback.")
    MemoryEpisode(summary="ActiveGraph memory design discussion.")
    MemoryQuery(query="What is the latest launch plan?")
    RetrievalPlan(query_id="q1")
    CoverageReport(query_id="q1")
    EvidenceBundle(query_id="q1")
    MemoryAnswer(query_id="q1", answer="Not enough evidence yet.")
    TemporalRef(text="for three weeks")
    QuantityClaim(property_name="budget", source_text="$10,000")
    MemoryPolicy(name="strict_latest")
    MemoryEvaluation(query_id="q1", judgment="helpful")


def test_confidence_bounds_are_enforced():
    with pytest.raises(ValidationError):
        MemoryClaim(text="Invalid confidence", confidence=1.5)


def test_query_top_k_bounds_are_enforced():
    with pytest.raises(ValidationError):
        MemoryQuery(query="too many", top_k=101)
