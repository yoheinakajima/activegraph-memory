from activegraph_memory.compiler import (
    ExtractedClaimInput,
    SourceTurn,
    compile_memory_index,
)
from activegraph_memory.retrieval import retrieve_memory
from activegraph_memory.temporal import resolve_relative_ago


def _turn(session_id, session_idx, turn_idx, role, content, date="2023-03-15"):
    return SourceTurn(
        turn_id=f"{session_id}#{turn_idx}",
        session_id=session_id,
        session_date=date,
        session_idx=session_idx,
        turn_idx=turn_idx,
        role=role,
        content=content,
        text=f"[Session {session_id} ({date})] {role}: {content}",
    )


def test_compile_memory_index_resolves_claim_anchors_and_quantities():
    turns = [
        _turn("s1", 0, 0, "user", "I bought a smoker for $120 today."),
        _turn("s1", 0, 1, "assistant", "Nice purchase."),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a smoker for $120.",
            session_id="s1",
            session_date="2023-03-15",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        )
    ]

    index = compile_memory_index(turns=turns, claims=claims)

    assert len(index.claims) == 1
    record = index.claims[0]
    assert record.source_turn_ids == ["s1#0"]
    assert record.quantity_claims[0].value == 120
    assert record.claim.source_ids == ["s1#0"]


def test_retrieve_memory_anchors_claim_above_source_turn():
    turns = [
        _turn("s1", 0, 0, "user", "I bought a smoker today."),
        _turn("s2", 1, 0, "user", "I bought a phone charger today.", "2023-03-10"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a smoker.",
            session_id="s1",
            session_date="2023-03-15",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought a phone charger.",
            session_id="s2",
            session_date="2023-03-10",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)
    claim_scores = {record.claim_id: (0.9 if "smoker" in record.text else 0.1) for record in index.claims}

    result = retrieve_memory(
        index,
        "What kitchen appliance did I buy 10 days ago?",
        query_id="q1",
        question_date="2023/03/25 (Sat)",
        claim_scores=claim_scores,
        token_budget=200,
    )

    assert "memory-claim: The user bought a smoker." in result.context_text
    assert "I bought a smoker today." in result.context_text
    assert "s1#0" in result.selected_turn_ids
    assert result.evidence_bundle.claim_ids
    assert result.coverage_report.coverage_confidence > 0


def test_relative_ago_resolves_against_slash_date_anchor():
    ref = resolve_relative_ago("10 days ago", anchor_time="2023/03/25 (Sat)")

    assert ref.resolved_start == "2023-03-15"
    assert ref.resolution_method == "relative_to_query"


def test_relative_ago_out_of_range_is_unresolved():
    ref = resolve_relative_ago("10000 years ago", anchor_time="2023/03/25 (Sat)")

    assert ref.resolution_method == "unresolved"
    assert ref.resolved_start is None
    assert ref.metadata["reason"] == "relative_date_out_of_range"


def test_past_duration_is_rendered_as_normalized_claim_time():
    turns = [
        _turn("s1", 0, 0, "user", "I have been taking Spanish classes for the past three months.", "2023-05-27"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user has been taking Spanish classes for the past three months.",
            session_id="s1",
            session_date="2023-05-27",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        )
    ]
    index = compile_memory_index(turns=turns, claims=claims)
    result = retrieve_memory(
        index,
        "Which event happened first, Spanish classes or a festival?",
        query_id="q2",
        question_date="2023/05/27 (Sat)",
        claim_scores={index.claims[0].claim_id: 0.9},
        token_budget=200,
    )

    assert "for the past three months => 2023-02-26..2023-05-27" in result.context_text
