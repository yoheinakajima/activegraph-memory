from activegraph_memory.compiler import (
    ExtractedClaimInput,
    SourceTurn,
    compile_memory_index,
    extract_quantity_claims,
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


def test_extract_quantity_claims_infers_generic_count_units_and_money_suffixes():
    examples = {
        "The user has tried four Korean restaurants so far.": (4.0, "restaurants"),
        "The user has tried three of Emma's recipes.": (3.0, "recipes"),
        "The user bought 5 H&M tops from H&M so far.": (5.0, "tops"),
        "The user has collected five National Geographic issues so far.": (5.0, "issues"),
        "The user was pre-approved for $400k on the mortgage.": (400000.0, "usd"),
    }

    for text, expected in examples.items():
        quantities = extract_quantity_claims(text)
        assert (quantities[0].value, quantities[0].unit) == expected


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

    assert result.metadata["requested_token_budget"] == 10000
    assert result.metadata["token_budget"] == 3200
    assert result.metadata["adaptive_budget_applied"] is True
    assert "[memory-answer-packet]" in result.context_text
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


def test_aggregate_sum_compiles_fundraising_events_across_participation_forms():
    turns = [
        _turn("ch1", 0, 0, "user", "I participated in a charity walk and raised $250 through sponsors.", "2023-05-01"),
        _turn("ch2", 1, 0, "user", "I helped organize a charity yoga event that raised $600 for a local animal shelter.", "2023-05-01"),
        _turn("ch3", 2, 0, "user", "My team raised $5,000 in the Bike-a-Thon for Cancer Research.", "2023-05-01"),
        _turn("ch4", 3, 0, "user", "I am thinking about organizing a charity dinner someday.", "2023-05-01"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user participated in a charity walk and raised $250 through sponsors.",
            session_id="ch1",
            session_date="2023-05-01",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user helped organize a charity yoga event that raised $600 for a local animal shelter.",
            session_id="ch2",
            session_date="2023-05-01",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user's team raised $5,000 in the Bike-a-Thon for Cancer Research.",
            session_id="ch3",
            session_date="2023-05-01",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user is thinking about organizing a charity dinner someday.",
            session_id="ch4",
            session_date="2023-05-01",
            session_idx=3,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How much money did I raise in total through all the charity events I participated in?",
        query_id="charity-total",
        question_date="2023/05/01 (Mon)",
    )

    assert "[memory-answer-packet]" in result.context_text
    assert "Computed sum: $5,850" in result.context_text
    assert result.metadata["graph_context_rendered"] is True
    assert result.metadata["graph_query"]["matched_events"] == 3
    graph_events = [row["event"] for row in result.metadata["graph_query"]["evidence_rows"]]
    assert "The user is thinking about organizing a charity dinner someday." not in graph_events


def test_broad_event_count_does_not_render_computed_answer_packet():
    turns = [
        _turn(f"tf{i}", i, 0, "user", f"My friend got collectible card number {i}.", f"2023-05-{i + 1:02d}")
        for i in range(18)
    ]
    turns.append(_turn("tf18", 18, 0, "user", "My friend got a tank for their aquarium.", "2023-05-19"))
    claims = [
        ExtractedClaimInput(
            text=f"The user's friend got collectible card number {i}.",
            session_id=f"tf{i}",
            session_date=f"2023-05-{i + 1:02d}",
            session_idx=i,
            role="user",
            mentioned_turn_idxs=(0,),
        )
        for i in range(18)
    ]
    claims.append(
        ExtractedClaimInput(
            text="The user's friend got a tank for their aquarium.",
            session_id="tf18",
            session_date="2023-05-19",
            session_idx=18,
            role="user",
            mentioned_turn_idxs=(0,),
        )
    )
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many tanks did my friend get?",
        query_id="broad-tank-count",
        question_date="2023/05/30 (Tue)",
    )

    assert result.metadata["graph_query"]["operation"] == "aggregate/count"
    assert result.metadata["graph_query"]["matched_events"] > 12
    assert result.metadata["graph_context_rendered"] is False
    assert result.metadata["graph_context_skip_reason"] == "broad_event_count"
    assert result.metadata["dynamic_expansion"]["triggered"] is True
    assert "broad_event_count" in result.metadata["dynamic_expansion"]["reasons"]
    assert result.metadata["dynamic_added_claim_ids"]
    assert "[memory-answer-packet]" not in result.context_text
    assert "Computed count:" not in result.context_text


def test_snapshot_count_uses_latest_quantity_instead_of_adding_progress_updates():
    turns = [
        _turn(
            "kr1",
            0,
            0,
            "user",
            "I had tried three Korean restaurants in my city by April.",
            "2023-04-10",
        ),
        _turn(
            "kr2",
            1,
            0,
            "user",
            "I've now tried four Korean restaurants in my city so far.",
            "2023-05-12",
        ),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user had tried three Korean restaurants in their city by April.",
            session_id="kr1",
            session_date="2023-04-10",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user has now tried four Korean restaurants in their city so far.",
            session_id="kr2",
            session_date="2023-05-12",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many Korean restaurants have I tried in my city?",
        query_id="korean-restaurant-count",
        question_date="2023/05/13 (Sat)",
    )

    graph = result.metadata["graph_query"]
    assert "Latest matching count: 4" in result.context_text
    assert "Computed count: 7" not in result.context_text
    assert graph["count_method"] == "latest_quantity_snapshot"
    assert graph["quantity_values"] == [4.0]
    assert graph["snapshot_values"] == [3.0, 4.0]


def test_snapshot_count_handles_branded_item_units():
    turns = [
        _turn("hm1", 0, 0, "user", "I bought 3 H&M tops earlier this spring.", "2023-04-15"),
        _turn("hm2", 1, 0, "user", "I have bought 5 H&M tops so far.", "2023-05-15"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought 3 H&M tops earlier this spring.",
            session_id="hm1",
            session_date="2023-04-15",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user has bought 5 H&M tops so far.",
            session_id="hm2",
            session_date="2023-05-15",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many H&M tops have I bought so far?",
        query_id="hm-top-count",
        question_date="2023/05/16 (Tue)",
    )

    assert "Latest matching count: 5" in result.context_text
    assert result.metadata["graph_query"]["count_method"] == "latest_quantity_snapshot"


def test_value_snapshot_uses_latest_preapproval_instead_of_summing_values():
    turns = [
        _turn("mtg1", 0, 0, "user", "I was pre-approved for $350k on the mortgage.", "2023-03-01"),
        _turn("mtg2", 1, 0, "user", "The mortgage pre-approval increased to $400k.", "2023-03-20"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user was pre-approved for $350k on the mortgage.",
            session_id="mtg1",
            session_date="2023-03-01",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user's mortgage pre-approval increased to $400k.",
            session_id="mtg2",
            session_date="2023-03-20",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How much was I pre-approved for when I got my mortgage?",
        query_id="mortgage-preapproval",
        question_date="2023/03/21 (Tue)",
    )

    graph = result.metadata["graph_query"]
    assert "Latest matching value: $400,000" in result.context_text
    assert "Computed sum: $750,000" not in result.context_text
    assert graph["sum_method"] == "latest_quantity_snapshot"
    assert graph["sum_values"] == [400000.0]


def test_count_hours_in_total_sums_game_duration_rows():
    turns = [
        _turn("game1", 0, 0, "user", "I spent around 70 hours playing Assassin's Creed Odyssey.", "2023-05-20"),
        _turn("game2", 1, 0, "user", "Hyper Light Drifter took me 5 hours to finish.", "2023-05-23"),
        _turn("game3", 2, 0, "user", "The Last of Us Part II took me 30 hours on hard difficulty.", "2023-05-25"),
        _turn("game4", 3, 0, "user", "Celeste took me 10 hours to complete.", "2023-05-27"),
        _turn("game5", 4, 0, "user", "I finished The Last of Us Part II on normal difficulty in 25 hours.", "2023-05-29"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user spent around 70 hours playing Assassin's Creed Odyssey.",
            session_id="game1",
            session_date="2023-05-20",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="Hyper Light Drifter took the user 5 hours to finish.",
            session_id="game2",
            session_date="2023-05-23",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The Last of Us Part II took the user 30 hours on hard difficulty.",
            session_id="game3",
            session_date="2023-05-25",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="Celeste took the user 10 hours to complete.",
            session_id="game4",
            session_date="2023-05-27",
            session_idx=3,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user finished The Last of Us Part II on normal difficulty in 25 hours.",
            session_id="game5",
            session_date="2023-05-29",
            session_idx=4,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many hours have I spent playing games in total?",
        query_id="game-hours-total",
        question_date="2023/05/30 (Tue)",
    )

    assert "Computed count: 140" in result.context_text
    assert result.metadata["graph_query"]["quantity_values"] == [70.0, 5.0, 30.0, 10.0, 25.0]
    assert result.metadata["graph_context_rendered"] is True


def test_money_sum_multiplies_item_count_by_each_price():
    turns = [
        _turn("market1", 0, 0, "user", "I sold 15 jars of homemade jam at the market, earning $225.", "2023-05-29"),
        _turn("market2", 1, 0, "user", "I sold 20 potted herb plants at the Summer Solstice Market for $7.50 each.", "2023-06-01"),
        _turn("market3", 2, 0, "user", "I sold 12 bunches of herbs at the farmers' market, earning $120.", "2023-05-15"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user sold 15 jars of homemade jam at the market, earning $225.",
            session_id="market1",
            session_date="2023-05-29",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user sold 20 potted herb plants at the Summer Solstice Market for $7.50 each.",
            session_id="market2",
            session_date="2023-06-01",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user sold 12 bunches of herbs at the farmers' market, earning $120.",
            session_id="market3",
            session_date="2023-05-15",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "What is the total amount of money I earned from selling my products at the markets?",
        query_id="market-revenue",
        question_date="2023/06/02 (Fri)",
    )

    assert "Computed sum: $495" in result.context_text
    assert result.metadata["graph_query"]["sum_values"] == [120.0, 225.0, 150.0]


def test_aggregate_max_identifies_highest_grocery_merchant_spend():
    turns = [
        _turn("gm1", 0, 0, "user", "I spent around $120 at Walmart during my last grocery trip.", "2023-05-26"),
        _turn("gm2", 1, 0, "user", "I ordered from Publix last week and spent around $60.", "2023-05-30"),
        _turn("gm3", 2, 0, "user", "I placed an online order with Thrive Market last month and spent around $150.", "2023-05-26"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user spent around $120 at Walmart during their last grocery trip.",
            session_id="gm1",
            session_date="2023-05-26",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user ordered from Publix last week and spent around $60.",
            session_id="gm2",
            session_date="2023-05-30",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user placed an online order with Thrive Market last month and spent around $150.",
            session_id="gm3",
            session_date="2023-05-26",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "Which grocery store did I spend the most money at in the past month?",
        query_id="grocery-max",
        question_date="2023/05/30 (Tue)",
    )

    assert "[graph-query: aggregate/max]" in result.context_text
    assert "Maximum matching spend: Thrive Market ($150)" in result.context_text
    assert result.metadata["graph_query"]["max_group"] == "Thrive Market"


def test_valuation_ratio_claim_is_prioritized_over_generic_art_advice():
    turns = [
        _turn("art1", 0, 0, "assistant", "Artist reputation and condition can affect art value.", "2023-05-30"),
        _turn("art2", 1, 0, "user", "My flea market find is worth triple what I paid for it.", "2023-05-30"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The assistant explained that artist reputation and condition can affect artwork value.",
            session_id="art1",
            session_date="2023-05-30",
            session_idx=0,
            role="assistant",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user's flea market find is worth triple what they paid for it.",
            session_id="art2",
            session_date="2023-05-30",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How much is the painting worth in terms of the amount I paid for it?",
        query_id="painting-ratio",
        question_date="2023/05/30 (Tue)",
        token_budget=120,
    )

    assert "worth triple what" in result.context_text
    assert any(
        "worth triple" in index.by_claim_id[claim_id].text
        for claim_id in result.selected_claim_ids
    )


def test_aggregate_difference_computes_savings_from_matching_money_events():
    turns = [
        _turn("jc1", 0, 0, "user", "I bought Jimmy Choo heels at TK Maxx for $200.", "2023-04-12"),
        _turn("jc2", 1, 0, "user", "Those Jimmy Choo heels originally retailed for $500.", "2023-04-13"),
        _turn("jc3", 2, 0, "user", "I bought coffee for $8.", "2023-04-14"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought Jimmy Choo heels at TK Maxx for $200.",
            session_id="jc1",
            session_date="2023-04-12",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The Jimmy Choo heels originally retailed for $500.",
            session_id="jc2",
            session_date="2023-04-13",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought coffee for $8.",
            session_id="jc3",
            session_date="2023-04-14",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How much did I save on the Jimmy Choo heels?",
        query_id="jimmy-choo-savings",
        question_date="2023/04/15 (Sat)",
    )

    assert "[graph-query: aggregate/difference]" in result.context_text
    assert "Computed difference: $300 ($500 - $200)" in result.context_text
    assert result.metadata["graph_query"]["matched_events"] == 2
    graph_events = [row["event"] for row in result.metadata["graph_query"]["evidence_rows"]]
    assert "The user bought coffee for $8." not in graph_events


def test_temporal_lookup_uses_graph_rows_for_relative_ago_purchase():
    turns = [
        _turn("sm1", 0, 0, "user", "I just got a smoker today.", "2023-03-15"),
        _turn("sm2", 1, 0, "user", "I bought a phone charger today.", "2023-03-18"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user got a smoker on 2023/03/15.",
            session_id="sm1",
            session_date="2023-03-15",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought a phone charger on 2023/03/18.",
            session_id="sm2",
            session_date="2023-03-18",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "What kitchen appliance did I buy 10 days ago?",
        query_id="smoker-ago",
        question_date="2023/03/25 (Sat)",
    )

    assert "[graph-query: temporal/timeline]" in result.context_text
    assert "2023-03-15 | The user got a smoker on 2023/03/15." in result.context_text
    assert result.metadata["graph_query"]["matched_events"] == 1


def test_temporal_date_delta_computes_weeks_ago_from_matching_event():
    turns = [
        _turn(
            "td1",
            0,
            0,
            "user",
            "I met up with my aunt and received the crystal chandelier today.",
            "2023-05-02",
        ),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user met up with their aunt and received the crystal chandelier on 2023/05/02.",
            session_id="td1",
            session_date="2023-05-02",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many weeks ago did I meet up with my aunt and receive the crystal chandelier?",
        query_id="chandelier-weeks-ago",
        question_date="2023/05/30 (Tue)",
    )

    assert "[graph-query: temporal/date-delta]" in result.context_text
    assert "Computed date difference: 4 weeks ago (2023-05-02 to 2023-05-30)" in result.context_text


def test_current_queries_render_newer_evidence_first():
    turns = [
        _turn("cv1", 0, 0, "user", "My current project is a Ford Mustang Shelby GT350R model.", "2023-05-20"),
        _turn("cv2", 1, 0, "user", "I switched to working on a Ford F-150 pickup truck model.", "2023-05-26"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user is working on a Ford Mustang Shelby GT350R model.",
            session_id="cv1",
            session_date="2023-05-20",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user switched to working on a Ford F-150 pickup truck model.",
            session_id="cv2",
            session_date="2023-05-26",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "What type of vehicle model am I currently working on?",
        query_id="current-vehicle",
        question_date="2023/05/27 (Sat)",
    )

    assert result.context_text.index("Ford F-150") < result.context_text.index("Ford Mustang")


def test_event_timeline_filters_generic_event_category_with_specific_tokens():
    turns = [
        _turn(
            "ev1",
            0,
            0,
            "user",
            "I have been taking Spanish classes for the past three months.",
            "2023-05-27",
        ),
        _turn("ev2", 1, 0, "user", "I attended a cultural festival yesterday.", "2023-05-27"),
        _turn("ev3", 2, 0, "user", "I attended a work conference last week.", "2023-05-27"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user has been taking Spanish classes for the past three months.",
            session_id="ev1",
            session_date="2023-05-27",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user attended a cultural festival yesterday.",
            session_id="ev2",
            session_date="2023-05-27",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user attended a work conference last week.",
            session_id="ev3",
            session_date="2023-05-27",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "Which event happened first, my attendance at a cultural festival or the start of my Spanish classes?",
        query_id="spanish-festival-order",
        question_date="2023/05/27 (Sat)",
    )

    assert "[graph-query: temporal/timeline]" in result.context_text
    graph_events = [row["event"] for row in result.metadata["graph_query"]["evidence_rows"]]
    assert graph_events == [
        "The user has been taking Spanish classes for the past three months.",
        "The user attended a cultural festival yesterday.",
    ]
    assert "The user attended a work conference last week." not in graph_events


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


def test_personal_luxury_sum_uses_concept_matches_and_ignores_assistant_budgets():
    turns = [
        _turn("lux1", 0, 0, "user", "I bought a luxury evening gown for $800.", "2023-05-24"),
        _turn("lux2", 1, 0, "user", "I purchased a designer handbag from Gucci for $1,200.", "2023-05-25"),
        _turn("lux3", 2, 0, "assistant", "Try a $200 monthly discretionary budget for fashion.", "2023-05-25"),
        _turn("lux4", 3, 0, "user", "I bought high-end Italian designer boots for $500.", "2023-05-28"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user bought a luxury evening gown for $800.",
            session_id="lux1",
            session_date="2023-05-24",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user purchased a designer handbag from Gucci for $1,200.",
            session_id="lux2",
            session_date="2023-05-25",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The assistant suggested a $200 monthly discretionary budget for fashion.",
            session_id="lux3",
            session_date="2023-05-25",
            session_idx=2,
            role="assistant",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user bought high-end Italian designer boots for $500.",
            session_id="lux4",
            session_date="2023-05-28",
            session_idx=3,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "What is the total amount I spent on luxury items in the past few months?",
        query_id="luxury-sum",
        question_date="2023/06/01 (Thu)",
        token_budget=300,
    )

    assert "Computed sum: $2,500" in result.context_text
    assert result.metadata["graph_query"]["sum_values"] == [800.0, 1200.0, 500.0]
    assert "monthly discretionary budget" not in result.context_text


def test_personal_wedding_count_prefers_user_events_and_treats_got_back_as_attendance():
    turns = [
        _turn("wed1", 0, 0, "user", "I just got back from my college roommate's wedding.", "2023-04-12"),
        _turn("wed2", 1, 0, "user", "I attended my cousin's wedding at a vineyard.", "2023-08-05"),
        _turn("wed3", 2, 0, "assistant", "Ask friends who attended the wedding if they saw it.", "2023-08-06"),
        _turn("wed4", 3, 0, "user", "I attended a friend's wedding last weekend.", "2023-10-15"),
    ]
    claims = [
        ExtractedClaimInput(
            text="The user just got back from their college roommate's wedding.",
            session_id="wed1",
            session_date="2023-04-12",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user attended their cousin's wedding at a vineyard.",
            session_id="wed2",
            session_date="2023-08-05",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The assistant suggested asking friends who attended the wedding if they saw it.",
            session_id="wed3",
            session_date="2023-08-06",
            session_idx=2,
            role="assistant",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user attended a friend's wedding last weekend.",
            session_id="wed4",
            session_date="2023-10-15",
            session_idx=3,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many weddings have I attended in this year?",
        query_id="wedding-count",
        question_date="2023/10/15 (Sun)",
        token_budget=300,
    )

    assert "Computed count: 3" in result.context_text
    assert "assistant suggested" not in result.context_text.lower()


def test_birth_count_dedupes_named_babies_and_excludes_assistant_fiction():
    turns = [
        _turn("baby1", 0, 0, "user", "Rachel has a son named Max who was born in March.", "2023-05-13"),
        _turn("baby2", 1, 0, "user", "Mike and Emma have a daughter named Charlotte who was born around March.", "2023-05-13"),
        _turn("baby3", 2, 0, "user", "My aunt has twins named Ava and Lily who were born in April.", "2023-05-13"),
        _turn("baby4", 3, 0, "assistant", "Rey and Kylo Ren have two children born with the Force.", "2023-05-13"),
        _turn("baby5", 4, 0, "user", "David just had his third child, a baby boy named Jasper.", "2023-05-13"),
        _turn("baby6", 5, 0, "user", "Mike and Emma welcomed their first baby, a girl named Charlotte.", "2023-05-13"),
    ]
    claims = [
        ExtractedClaimInput(
            text="Rachel has a son named Max who was born in March.",
            session_id="baby1",
            session_date="2023-05-13",
            session_idx=0,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="Mike and Emma have a daughter named Charlotte who was born around March.",
            session_id="baby2",
            session_date="2023-05-13",
            session_idx=1,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The user has an aunt with twins named Ava and Lily who were born in April.",
            session_id="baby3",
            session_date="2023-05-13",
            session_idx=2,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="The assistant wrote that Rey and Kylo Ren have two children born with the Force.",
            session_id="baby4",
            session_date="2023-05-13",
            session_idx=3,
            role="assistant",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="David just had his third child, a baby boy named Jasper.",
            session_id="baby5",
            session_date="2023-05-13",
            session_idx=4,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
        ExtractedClaimInput(
            text="Mike and Emma welcomed their first baby, a girl named Charlotte.",
            session_id="baby6",
            session_date="2023-05-13",
            session_idx=5,
            role="user",
            mentioned_turn_idxs=(0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = retrieve_memory(
        index,
        "How many babies were born to friends and family members in the last few months?",
        query_id="baby-count",
        question_date="2023/05/13 (Sat)",
        token_budget=500,
    )

    graph = result.metadata["graph_query"]
    assert "Computed count: 5" in result.context_text
    assert graph["count_method"] == "named_entity_count"
    assert graph["entity_names"] == ["Max", "Charlotte", "Ava", "Lily", "Jasper"]
    assert "Rey and Kylo Ren" not in result.context_text
