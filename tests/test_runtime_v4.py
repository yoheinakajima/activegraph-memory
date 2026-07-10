from __future__ import annotations

import json
from pathlib import Path

import pytest

from activegraph_memory import (
    ExtractedClaimInput,
    MemoryRuntime,
    SourceTurn,
    compile_memory_index,
    calibrate_operator_thresholds,
    apply_operator_calibration,
    runtime_profile,
)


def _turn(session, idx, content, date="2023-01-01", role="user"):
    return SourceTurn(
        turn_id=f"{session}#0",
        session_id=session,
        session_date=date,
        session_idx=idx,
        turn_idx=0,
        role=role,
        content=content,
        text=f"[Session {session} ({date})] {role}: {content}",
    )


def _claim(turn, text=None, **metadata):
    return ExtractedClaimInput(
        text=text or turn.content,
        session_id=turn.session_id,
        session_date=turn.session_date,
        session_idx=turn.session_idx,
        role=turn.role,
        mentioned_turn_idxs=(0,),
        metadata=metadata,
    )


def test_aggregate_coverage_recovers_uncompiled_source_without_certifying_count():
    first = _turn("wedding-a", 0, "I attended Rachel and Mike's wedding in March.", "2023-03-10")
    second = _turn("wedding-b", 1, "I attended Jen and Tom's wedding in June.", "2023-06-10")
    index = compile_memory_index(turns=[first, second], claims=[_claim(first)])

    result = MemoryRuntime("max_quality").retrieve(
        index,
        "How many weddings did I attend this year?",
        question_date="2023-12-31",
    )
    compiled = result.metadata["compiled_evidence"]
    audit = compiled["metadata"]["coverage_audit"]

    assert audit["extraction_ratio"] == 0.5
    assert audit["reader_coverage_ratio"] == 1.0
    assert audit["complete"] is False
    assert second.turn_id in compiled["selected_turn_ids"]
    assert any(slot["slot"] == "coverage_recovery" for slot in compiled["evidence_slots"])
    assert result.metadata["candidate_answer_rendered"] is False
    assert second.content in result.context_text


def test_preference_slots_preserve_positive_scope_and_negative_constraints():
    positive = _turn("hotel-positive", 0, "I prefer quiet hotels near train stations.")
    negative = _turn("hotel-negative", 1, "I avoid noisy hotels and nightclubs.")
    index = compile_memory_index(
        turns=[positive, negative],
        claims=[_claim(positive), _claim(negative)],
    )

    result = MemoryRuntime("max_quality").retrieve(index, "Recommend a hotel for my next trip.")
    compiled = result.metadata["compiled_evidence"]
    slots = compiled["evidence_slots"]

    assert {positive.turn_id, negative.turn_id} <= set(compiled["selected_turn_ids"])
    assert any(slot["slot"] == "positive_preference" and "quiet" in slot["evidence"] for slot in slots)
    assert any(slot["slot"] == "negative_constraint" and "noisy" in slot["evidence"] for slot in slots)
    assert "preference_coverage" in compiled["satisfied_requirements"]


def test_temporal_packet_has_one_explicit_slot_per_operand():
    alpha = _turn("alpha", 0, "We launched Alpha on March 1st.", "2023-03-01")
    beta = _turn("beta", 1, "We launched Beta on March 15th.", "2023-03-15")
    index = compile_memory_index(
        turns=[alpha, beta],
        claims=[_claim(alpha), _claim(beta)],
    )

    result = MemoryRuntime("quality").retrieve(
        index,
        "Which happened first, Alpha launch or Beta launch?",
        question_date="2023-03-20",
    )
    slots = result.metadata["compiled_evidence"]["evidence_slots"]

    assert len(slots) == 2
    assert all(slot["status"] == "found" for slot in slots)
    assert {slot["date"] for slot in slots} == {"2023-03-01", "2023-03-15"}


def test_operator_threshold_can_suppress_an_otherwise_complete_candidate():
    event = _turn("one", 0, "I attended Rachel's wedding.", "2023-03-01")
    index = compile_memory_index(turns=[event], claims=[_claim(event)])
    base = runtime_profile("quality")
    profile = base.model_copy(
        update={
            "operator_min_confidence": {**base.operator_min_confidence, "count": 0.99},
        }
    )

    result = MemoryRuntime(profile).retrieve(index, "How many weddings did I attend?")

    assert result.metadata["retrieval_assessment"]["metadata"]["effective_confidence_threshold"] == 0.99
    assert result.metadata["candidate_answer_rendered"] is False


def test_operator_calibration_uses_held_out_precision_and_applies_to_profile():
    records = [
        {"operator": "count", "confidence": confidence, "correct": correct}
        for confidence, correct in (
            (0.99, True),
            (0.95, True),
            (0.91, True),
            (0.88, True),
            (0.84, True),
            (0.8, False),
            (0.76, False),
        )
    ]

    calibration = calibrate_operator_thresholds(
        records,
        target_precision=1.0,
        minimum_accepted=5,
        thresholds=[0.75, 0.8, 0.84, 0.9],
    )
    calibrated = apply_operator_calibration(runtime_profile("quality"), calibration)

    assert calibration["count"].threshold == 0.84
    assert calibration["count"].accepted == 5
    assert calibrated.operator_min_confidence["count"] == 0.84


def test_operator_thresholds_reject_out_of_range_values():
    with pytest.raises(ValueError, match="between 0 and 1"):
        runtime_profile("quality").model_copy(
            update={"operator_min_confidence": {"count": 1.1}}
        ).model_validate(
            {
                **runtime_profile("quality").model_dump(),
                "operator_min_confidence": {"count": 1.1},
            }
        )


def test_non_benchmark_application_traces():
    fixture = Path(__file__).parents[1] / "examples" / "v4_application_traces.json"
    payload = json.loads(fixture.read_text())
    turns = [SourceTurn(**raw) for raw in payload["turns"]]
    claims = [
        ExtractedClaimInput(
            **{
                **raw,
                "mentioned_turn_idxs": tuple(raw.get("mentioned_turn_idxs", [])),
            }
        )
        for raw in payload["claims"]
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    for case in payload["cases"]:
        result = MemoryRuntime("max_quality").retrieve(
            index,
            case["query"],
            question_date=case.get("question_date"),
        )
        compiled = result.metadata["compiled_evidence"]
        if case.get("expected") is not None:
            assert compiled["candidate_answer"] == case["expected"], case["case_id"]
        for expected_text in (case.get("metadata") or {}).get("expected_context_contains", []):
            assert expected_text in result.context_text, case["case_id"]
