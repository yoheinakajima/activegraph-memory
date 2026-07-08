from activegraph_memory.coverage import build_coverage_report
from activegraph_memory.scoring import (
    confidence_label,
    confidence_vector,
    overall_confidence,
    select_epistemic_status,
)


def test_coverage_report_marks_unbounded_latest_as_insufficient():
    report = build_coverage_report(
        query_id="q1",
        searched_scopes=["uploaded_docs"],
        not_searched_scopes=["email", "drive"],
        required_scopes=["uploaded_docs", "email"],
        query_type="latest",
    )

    assert report.bounded is False
    assert report.coverage_confidence == 0.5
    assert "uncaveated_current_or_final_claim" in report.not_adequate_for


def test_confidence_uses_minimum_required_dimension():
    confidence = confidence_vector(
        relevance=0.95,
        entity_match=0.9,
        authority=0.8,
        freshness=0.4,
        coverage=0.3,
        consistency=0.9,
        extraction=0.95,
        reasoning=0.7,
    )

    assert overall_confidence(confidence, requires_coverage=True) == 0.3
    assert confidence_label(0.7) == "moderate_high"


def test_status_prefers_coverage_warning_for_latest():
    report = build_coverage_report(
        query_id="q1",
        searched_scopes=["uploaded_docs"],
        not_searched_scopes=["drive"],
        query_type="latest",
    )
    status = select_epistemic_status(
        confidence_vector(
            relevance=0.9,
            entity_match=0.9,
            authority=0.9,
            freshness=0.9,
            coverage=0.5,
            consistency=0.9,
            extraction=0.9,
        ),
        coverage_report=report,
        requires_coverage=True,
        requires_freshness=True,
    )

    assert status == "insufficient_coverage"
