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
    assert len(index.events) == 1
    assert index.events[0].predicate == "purchase"


def test_compile_memory_index_extracts_commas_word_quantities_and_loose_dates():
    turns = [
        _turn("q1", 0, 0, "user", "I donated $1,200 and bought three plants.", "2023-5-2"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user donated $1,200 and bought three plants.",
            session_id="q1",
            session_date="2023-5-2",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        )
    ]

    index = compile_memory_index(turns=turns, claims=claims)
    quantities = index.claims[0].quantity_claims

    assert index.claims[0].claim.valid_from == "2023-05-02"
    assert quantities[0].value == 1200
    assert quantities[0].unit == "usd"
    assert quantities[1].value == 3
    assert quantities[1].unit == "plants"


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


def test_aggregate_count_uses_compiled_events_before_source_context():
    turns = [
        _turn("p1", 0, 0, "user", "I bought a snake plant today.", "2023-05-02"),
        _turn("p2", 1, 0, "user", "I picked up a peace lily today.", "2023-05-10"),
        _turn("p3", 2, 0, "user", "I got a succulent today.", "2023-05-25"),
        _turn("p4", 3, 0, "user", "I bought a fern today.", "2023-04-10"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a snake plant.",
            session_id="p1",
            session_date="2023-05-02",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user picked up a peace lily.",
            session_id="p2",
            session_date="2023-05-10",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user got a succulent.",
            session_id="p3",
            session_date="2023-05-25",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought a fern.",
            session_id="p4",
            session_date="2023-04-10",
            session_idx=3,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many plants did I acquire in the last month?",
        query_id="plants",
        question_date="2023/05/30 (Tue)",
    )

    assert result.metadata["token_budget"] == 10000
    assert "[graph-query: aggregate/count]" in result.context_text
    assert "Computed count: 3" in result.context_text
    assert result.metadata["graph_query"]["matched_events"] == 3


def test_aggregate_count_skips_negated_events():
    turns = [
        _turn("n1", 0, 0, "user", "I bought a snake plant today.", "2023-05-02"),
        _turn("n2", 1, 0, "user", "I did not buy a fern today.", "2023-05-10"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a snake plant.",
            session_id="n1",
            session_date="2023-05-02",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user did not buy a fern.",
            session_id="n2",
            session_date="2023-05-10",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many plants did I buy in May?",
        query_id="negated-plants",
        question_date="2023/05/30 (Tue)",
    )

    assert "Computed count: 1" in result.context_text
    graph_events = [row["event"] for row in result.metadata["graph_query"]["evidence_rows"]]
    assert "The user did not buy a fern." not in graph_events


def test_aggregate_sum_uses_quantity_claims_and_category_filters():
    turns = [
        _turn("b1", 0, 0, "user", "I bought a bike helmet for $80.", "2023-01-12"),
        _turn("b2", 1, 0, "user", "I paid $35 for a bike tune-up.", "2023-02-20"),
        _turn("b3", 2, 0, "user", "I bought bike lights for $25.", "2023-05-01"),
        _turn("b4", 3, 0, "user", "I bought coffee for $10.", "2023-03-01"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a bike helmet for $80.",
            session_id="b1",
            session_date="2023-01-12",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user paid $35 for a bike tune-up.",
            session_id="b2",
            session_date="2023-02-20",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought bike lights for $25.",
            session_id="b3",
            session_date="2023-05-01",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought coffee for $10.",
            session_id="b4",
            session_date="2023-03-01",
            session_idx=3,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How much total money have I spent on bike-related expenses since the start of the year?",
        query_id="bike-spend",
        question_date="2023/05/05 (Fri)",
    )

    assert "[graph-query: aggregate/sum]" in result.context_text
    assert "Computed sum: $140" in result.context_text
    assert result.metadata["graph_query"]["matched_events"] == 3
    graph_events = [row["event"] for row in result.metadata["graph_query"]["evidence_rows"]]
    assert "The user bought coffee for $10." not in graph_events


def test_aggregate_sum_does_not_report_zero_when_quantities_are_missing():
    turns = [
        _turn("sq1", 0, 0, "user", "The bike repair cost more than expected.", "2023-02-20"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user's bike repair cost more than expected.",
            session_id="sq1",
            session_date="2023-02-20",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How much total money have I spent on bike expenses this year?",
        query_id="missing-quantity",
        question_date="2023/05/05 (Fri)",
    )

    assert "No matching quantity values found for sum." in result.context_text
    assert "Computed sum: $0" not in result.context_text


def test_aggregate_between_window_uses_two_temporal_refs():
    turns = [
        _turn("bw1", 0, 0, "user", "I bought a snake plant.", "2023-01-15"),
        _turn("bw2", 1, 0, "user", "I bought a fern.", "2023-02-10"),
        _turn("bw3", 2, 0, "user", "I bought a pothos.", "2023-03-15"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a snake plant.",
            session_id="bw1",
            session_date="2023-01-15",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought a fern.",
            session_id="bw2",
            session_date="2023-02-10",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought a pothos.",
            session_id="bw3",
            session_date="2023-03-15",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many plants did I buy between 2023-02-01 and 2023-02-28?",
        query_id="plant-window",
        question_date="2023/04/01 (Sat)",
    )

    assert "Computed count: 1" in result.context_text
    assert result.metadata["graph_query"]["matched_events"] == 1
    assert result.metadata["graph_query"]["evidence_rows"][0]["event"] == "The user bought a fern."


def test_graph_query_summary_is_kept_when_rows_exceed_budget():
    turns = [
        _turn(f"tg{i}", i, 0, "user", f"I bought plant number {i}.", f"2023-05-{i + 1:02d}")
        for i in range(8)
    ]
    claims = [
        ExtractedClaimInput(
            text=f"The user bought plant number {i}.",
            session_id=f"tg{i}",
            session_date=f"2023-05-{i + 1:02d}",
            session_idx=i,
            role="user",
            mentioned_turn_idxs=(0,),
        )
        for i in range(8)
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many plants did I buy in May?",
        query_id="tight-graph",
        question_date="2023/05/30 (Tue)",
        token_budget=35,
    )

    assert "[graph-query: aggregate/count]" in result.context_text
    assert "Computed count: 8" in result.context_text


def test_temporal_graph_query_renders_chronological_rows():
    turns = [
        _turn("m1", 0, 0, "user", "I visited the Museum of Modern Art.", "2023-01-03"),
        _turn("m2", 1, 0, "user", "I visited the local history museum.", "2023-02-10"),
        _turn("m3", 2, 0, "user", "I visited the science museum.", "2023-03-05"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user visited the Museum of Modern Art.",
            session_id="m1",
            session_date="2023-01-03",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user visited the local history museum.",
            session_id="m2",
            session_date="2023-02-10",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user visited the science museum.",
            session_id="m3",
            session_date="2023-03-05",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "Show me museum visits in chronological order.",
        query_id="museum-timeline",
        question_date="2023/03/10 (Fri)",
    )

    assert "[graph-query: temporal/timeline]" in result.context_text
    first = result.context_text.index("2023-01-03")
    second = result.context_text.index("2023-02-10")
    third = result.context_text.index("2023-03-05")
    assert first < second < third
