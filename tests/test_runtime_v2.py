from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from activegraph import Graph
from activegraph.llm import HashEmbeddingProvider, LLMResponse

from activegraph_memory import (
    ActiveGraphMemorySettings,
    EmbeddingSignalProvider,
    ExtractedMemoryFact,
    ExtractedClaimInput,
    GraphMemoryRepository,
    MemoryBenchmarkCase,
    MemoryRuntime,
    MemoryExtractionResult,
    ReasoningResponse,
    SQLiteEmbeddingStore,
    SourceTurn,
    analyze_query,
    benchmark_profiles,
    benchmark_ingestion,
    benchmark_reasoning_ablations,
    compile_memory_index,
    extract_claim_inputs,
    load_memory_index,
    materialize_memory_index,
    materialize_retrieval_trace,
    runtime_profile,
)
from activegraph_memory.compiler import extract_quantity_claims
from activegraph_memory.ranking import RetrievalSignals, propagate_graph_signals


def _turn(session_id, session_idx, turn_idx, role, content, date):
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


def _index():
    turns = [
        _turn("s1", 0, 0, "user", "Rachel's team has five women.", "2023-01-01"),
        _turn("s2", 1, 0, "user", "Rachel's team now has six women.", "2023-02-01"),
        _turn("s3", 2, 0, "user", "I plan to buy a road bike next Saturday.", "2023-02-02"),
        _turn("s4", 3, 0, "user", "I bought a road bike last Saturday for $500.", "2023-02-10"),
        _turn(
            "s5",
            4,
            0,
            "assistant",
            "Jobs:\n1. Tutor\n2. Bookkeeper\n3. Transcriptionist",
            "2023-02-11",
        ),
        _turn("s6", 5, 0, "user", "I prefer hotels with ocean views and rooftop pools.", "2023-02-12"),
    ]
    claims = [
        ExtractedClaimInput("Rachel's team has five women.", "s1", "2023-01-01", 0, "user", (0,)),
        ExtractedClaimInput("Rachel's team now has six women.", "s2", "2023-02-01", 1, "user", (0,)),
        ExtractedClaimInput("The user plans to buy a road bike next Saturday.", "s3", "2023-02-02", 2, "user", (0,)),
        ExtractedClaimInput("The user bought a road bike last Saturday for $500.", "s4", "2023-02-10", 3, "user", (0,)),
        ExtractedClaimInput("The assistant provided a list of jobs.", "s5", "2023-02-11", 4, "assistant", (0,)),
        ExtractedClaimInput("The user prefers hotels with ocean views and rooftop pools.", "s6", "2023-02-12", 5, "user", (0,)),
    ]
    return compile_memory_index(turns=turns, claims=claims)


def test_query_analysis_is_multi_operator_and_proof_oriented():
    analysis = analyze_query(
        "How many bike purchases did I make in the last month?",
        question_date="2023-03-01",
    )

    assert analysis.operators[0] == "count"
    assert analysis.completed_only is True
    assert analysis.time_start == "2023-01-30"
    assert "bounded_candidate_set" in analysis.proof_requirements
    assert "canonical_event_deduplication" in analysis.proof_requirements
    assert analyze_query("Order the visits from earliest to latest").operators == ["order"]


def test_query_analysis_uses_ambiguous_roles_instead_of_personal_pronoun_shortcut():
    assert analyze_query("What restaurant did you recommend last time?").source_roles == ["assistant"]
    assert analyze_query(
        "I was looking back at our previous conversation. Can you remind me of the designer handle?"
    ).source_roles == ["assistant"]
    assert analyze_query("How many weddings have I attended?").source_roles == ["user"]
    assert analyze_query("What was the dessert shop from our old trip?").source_roles == ["user", "assistant"]


def test_quantity_parser_rejects_models_dates_ordinals_and_years():
    quantities = extract_quantity_claims(
        "The Dell XPS 13 arrived February 25th, 2023 and the Galaxy S22 cost $500."
    )

    assert [(item.value, item.unit) for item in quantities] == [(500.0, "usd")]


def test_typed_projection_separates_plans_and_builds_state_history():
    index = _index()

    planned = next(item for item in index.compiled.event_mentions if "plans to buy" in item.text)
    actual = next(item for item in index.compiled.event_mentions if "bought a road bike" in item.text)
    assert planned.modality == "planned"
    assert planned.event_start == "2023-02-04"
    assert actual.modality == "actual"
    assert actual.event_start == "2023-02-04"

    team_states = [state for state in index.compiled.state_versions if "women" in state.value_text]
    assert len(team_states) == 2
    assert [state.status for state in team_states] == ["superseded", "active"]


def test_canonical_events_merge_repeated_mentions_but_not_new_events():
    turns = [
        _turn("a", 0, 0, "user", "I bought bike lights for $40 today.", "2023-03-01"),
        _turn("a", 0, 1, "user", "The new bike lights cost me $40.", "2023-03-01"),
        _turn("b", 1, 0, "user", "I bought another bike light for $40 today.", "2023-03-08"),
    ]
    claims = [
        ExtractedClaimInput("The user bought bike lights for $40.", "a", "2023-03-01", 0, "user", (0,)),
        ExtractedClaimInput("The user's new bike lights cost $40.", "a", "2023-03-01", 0, "user", (1,)),
        ExtractedClaimInput("The user bought another bike light for $40.", "b", "2023-03-08", 1, "user", (0,)),
    ]

    index = compile_memory_index(turns=turns, claims=claims)
    events = [event for event in index.compiled.canonical_events if "bike light" in event.summary.lower()]

    assert len(events) == 2
    assert sorted(len(event.mention_ids) for event in events) == [1, 2]


def test_canonicalization_does_not_bridge_distinct_quantities_through_generic_claim():
    turns = [
        _turn(
            "luxury",
            0,
            0,
            "user",
            "I bought H&M tees for $20 and high-end designer boots for $500.",
            "2023-05-28",
        )
    ]
    claims = [
        ExtractedClaimInput(
            "The user swings between luxury and budget-friendly purchases.",
            "luxury",
            "2023-05-28",
            0,
            "user",
            (0,),
        ),
        ExtractedClaimInput(
            "The user bought H&M tees for $20.",
            "luxury",
            "2023-05-28",
            0,
            "user",
            (0,),
        ),
        ExtractedClaimInput(
            "The user bought high-end designer boots for $500.",
            "luxury",
            "2023-05-28",
            0,
            "user",
            (0,),
        ),
    ]

    index = compile_memory_index(turns=turns, claims=claims)
    priced = [
        event
        for event in index.compiled.canonical_events
        if any(quantity.get("unit") == "usd" for quantity in event.quantities)
    ]

    assert len(priced) == 2
    assert sorted(quantity["value"] for event in priced for quantity in event.quantities) == [20.0, 500.0]


def test_luxury_sum_uses_specific_category_instead_of_all_clothing():
    turns = [
        _turn("a", 0, 0, "user", "I bought a Gucci handbag for $1,200.", "2023-05-24"),
        _turn("b", 1, 0, "user", "I bought a luxury gown for $800.", "2023-05-25"),
        _turn("c", 2, 0, "user", "I bought designer boots for $500 and H&M tees for $20.", "2023-05-26"),
    ]
    claims = [
        ExtractedClaimInput("The user bought a Gucci handbag for $1,200.", "a", "2023-05-24", 0, "user", (0,)),
        ExtractedClaimInput("The user bought a luxury gown for $800.", "b", "2023-05-25", 1, "user", (0,)),
        ExtractedClaimInput("The user bought designer boots for $500.", "c", "2023-05-26", 2, "user", (0,)),
        ExtractedClaimInput("The user bought H&M tees for $20.", "c", "2023-05-26", 2, "user", (0,)),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = MemoryRuntime("fast").retrieve(index, "How much did I spend on luxury items?")

    assert result.metadata["compiled_evidence"]["candidate_answer"] == "$2500"
    assert "Proof-complete candidate" in result.context_text
    assert "Verified candidate" not in result.context_text
    assert result.context_text.index("I bought a Gucci handbag") < result.context_text.index("[compiled-memory:")
    assert set(result.metadata["compiled_evidence"]["selected_turn_ids"]) <= set(result.selected_turn_ids)


def test_relative_time_lookup_prioritizes_nearby_claims():
    turns = [
        _turn("milestone", 0, 0, "user", "I signed my first client contract today.", "2023-03-01"),
        _turn("distractor", 1, 0, "user", "I replaced my bathroom mat three weeks ago.", "2023-02-21"),
    ]
    claims = [
        ExtractedClaimInput("The user signed a contract with their first client today.", "milestone", "2023-03-01", 0, "user", (0,)),
        ExtractedClaimInput("The user replaced their bathroom mat three weeks ago.", "distractor", "2023-02-21", 1, "user", (0,)),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = MemoryRuntime("balanced").retrieve(
        index,
        "What business milestone did I mention four weeks ago?",
        question_date="2023-03-28",
    )

    rows = result.metadata["compiled_evidence"]["rows"]
    assert rows
    assert "first client" in rows[0]["claim"]


def test_user_memory_query_filters_assistant_events_across_retrieval_paths():
    turns = [
        _turn("user-bike", 0, 0, "user", "I bought a road bike yesterday.", "2023-03-01"),
        _turn("assistant-bike", 1, 0, "assistant", "I bought a mountain bike yesterday.", "2023-03-01"),
    ]
    claims = [
        ExtractedClaimInput("The user bought a road bike yesterday.", "user-bike", "2023-03-01", 0, "user", (0,)),
        ExtractedClaimInput(
            "The assistant bought a mountain bike yesterday.",
            "assistant-bike",
            "2023-03-01",
            1,
            "assistant",
            (0,),
        ),
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = MemoryRuntime("quality").retrieve(index, "How many bikes did I buy?")

    assert result.metadata["compiled_evidence"]["candidate_answer"] == "1"
    assert "assistant-bike#0" not in result.selected_turn_ids
    assert "mountain bike" not in result.context_text


def test_completed_event_count_does_not_use_numeric_state_snapshot():
    turns = [
        _turn("age", 0, 0, "user", "I am 32 years old.", "2023-01-01"),
        _turn("w1", 1, 0, "user", "I attended Rachel's wedding.", "2023-02-01"),
        _turn("w2", 2, 0, "user", "I attended Sam's wedding.", "2023-03-01"),
    ]
    claims = [
        ExtractedClaimInput("The user is 32 years old.", "age", "2023-01-01", 0, "user", (0,)),
        ExtractedClaimInput("The user attended Rachel's wedding.", "w1", "2023-02-01", 1, "user", (0,)),
        ExtractedClaimInput("The user attended Sam's wedding.", "w2", "2023-03-01", 2, "user", (0,)),
    ]

    result = MemoryRuntime("fast").retrieve(
        compile_memory_index(turns=turns, claims=claims),
        "How many weddings have I attended?",
    )

    assert result.metadata["compiled_evidence"]["operation"] == "aggregate/count"
    assert result.metadata["compiled_evidence"]["candidate_answer"] == "2"


def test_current_state_prefers_recent_explicit_transition():
    turns = [
        _turn("old", 0, 0, "user", "I am working on a Ford Mustang model.", "2023-05-20"),
        _turn(
            "new",
            1,
            0,
            "user",
            "I wrapped up that model and switched to a Ford F-150 pickup truck model.",
            "2023-05-26",
        ),
    ]
    claims = [
        ExtractedClaimInput("The user is working on a Ford Mustang model.", "old", "2023-05-20", 0, "user", (0,)),
        ExtractedClaimInput("The user owns a Ford F-150 pickup truck model.", "new", "2023-05-26", 1, "user", (0,)),
    ]

    result = MemoryRuntime("fast").retrieve(
        compile_memory_index(turns=turns, claims=claims),
        "What vehicle model am I currently working on?",
    )

    assert "F-150" in result.metadata["compiled_evidence"]["candidate_answer"]
    assert result.metadata["compiled_evidence"]["metadata"]["top_transition_score"] == 1.0


def test_temporal_operands_resolve_from_source_durations():
    turns = [
        _turn(
            "festival",
            0,
            0,
            "user",
            "I attended a cultural festival in my hometown yesterday.",
            "2023-05-27",
        ),
        _turn(
            "classes",
            1,
            0,
            "user",
            "I've been taking Spanish classes for the past three months.",
            "2023-05-27",
        ),
    ]
    claims = [
        ExtractedClaimInput("The user attended a cultural festival yesterday.", "festival", "2023-05-27", 0, "user", (0,)),
        ExtractedClaimInput("The user has taken Spanish classes for three months.", "classes", "2023-05-27", 1, "user", (0,)),
    ]

    result = MemoryRuntime("fast").retrieve(
        compile_memory_index(turns=turns, claims=claims),
        "Which happened first, attendance at a cultural festival or the start of my Spanish classes?",
    )

    evidence = result.metadata["compiled_evidence"]
    assert evidence["candidate_answer"].startswith("start of my Spanish classes")
    assert [row["date"] for row in evidence["rows"]] == ["2023-02-26", "2023-05-26"]


def test_source_timeline_orders_named_entities_and_excludes_other_venue_types():
    names = [
        ("Science Museum", "2023-01-01"),
        ("Museum of Contemporary Art", "2023-01-08"),
        ("Metropolitan Museum of Art", "2023-01-15"),
        ("Museum of History", "2023-01-22"),
        ("Modern Art Museum", "2023-01-29"),
        ("Natural History Museum", "2023-02-05"),
        ("Modern Art Gallery", "2023-01-10"),
    ]
    turns = [
        _turn(f"s{idx}", idx, 0, "user", f"I visited the {name} today.", observed)
        for idx, (name, observed) in enumerate(names)
    ]
    claims = [
        ExtractedClaimInput(f"The user visited the {name}.", f"s{idx}", observed, idx, "user", (0,))
        for idx, (name, observed) in enumerate(names)
    ]

    result = MemoryRuntime("fast").retrieve(
        compile_memory_index(turns=turns, claims=claims),
        "What is the order of the six museums I visited from earliest to latest?",
    )

    evidence = result.metadata["compiled_evidence"]
    assert evidence["metadata"]["source_timeline"] is True
    assert [row["entity"] for row in evidence["rows"]] == [name for name, _ in names[:6]]
    assert "Gallery" not in evidence["candidate_answer"]


def test_runtime_answers_snapshot_count_from_current_state():
    result = MemoryRuntime("fast").retrieve(
        _index(),
        "How many women are currently on Rachel's team?",
    )

    compiled = result.metadata["compiled_evidence"]
    assert compiled["operation"] == "state/snapshot-count"
    assert compiled["candidate_answer"] == "6"
    assert compiled["proof_complete"] is True


def test_count_sums_item_cardinality_inside_one_event():
    turns = [
        _turn(
            "plants",
            0,
            0,
            "user",
            "I acquired a peace lily and a succulent yesterday.",
            "2023-05-10",
        )
    ]
    claims = [
        ExtractedClaimInput(
            "The user acquired a peace lily and a succulent yesterday.",
            "plants",
            "2023-05-10",
            0,
            "user",
            (0,),
        )
    ]
    index = compile_memory_index(turns=turns, claims=claims)

    result = MemoryRuntime("fast").retrieve(
        index,
        "How many plants did I acquire?",
        question_date="2023-05-11",
    )

    compiled = result.metadata["compiled_evidence"]
    assert compiled["candidate_answer"] == "2"
    assert compiled["rows"][0]["matched_items"] == ["peace lily", "succulent"]
    assert compiled["proof_complete"] is True


def test_runtime_preserves_ordinal_list_positions():
    index = _index()
    scores = {turn.turn_id: (1.0 if turn.turn_id == "s5#0" else 0.0) for turn in index.turns}
    result = MemoryRuntime("fast").retrieve(
        index,
        "What was the 3rd job in the list you provided?",
        turn_scores=scores,
    )

    assert result.metadata["compiled_evidence"]["candidate_answer"] == "Transcriptionist"
    assert "position=3" in result.context_text


def test_embedding_signal_provider_scores_compiled_fields():
    index = _index()
    provider = EmbeddingSignalProvider(
        index,
        HashEmbeddingProvider(dimensions=32),
        model="hash-32",
        embed_entities=True,
        embed_events=True,
    )

    signals = provider.score("Rachel team women")

    assert len(signals.claim_scores) == len(index.claims)
    assert len(signals.turn_scores) == len(index.turns)
    assert len(signals.event_scores) == len(index.compiled.canonical_events)
    assert signals.input_tokens > 0


def test_entity_embedding_signal_propagates_over_compiled_edges():
    index = _index()
    entity = next(item for item in index.compiled.entities if "Rachel" in item.canonical_name)
    signals = RetrievalSignals(entity_scores={entity.entity_id: 1.0})

    propagate_graph_signals(index, signals)

    assert any(score > 0 for score in signals.claim_scores.values())
    assert any(score > 0 for score in signals.state_scores.values())
    assert signals.metadata["graph_signal_propagation"]["entity_edges_applied"] > 0


class _CountingEmbeddingProvider:
    default_model = "hash-16"

    def __init__(self):
        self.inner = HashEmbeddingProvider(dimensions=16)
        self.calls = []

    def embed(self, *, texts, model):
        self.calls.append(list(texts))
        return self.inner.embed(texts=texts, model=model)


def test_sqlite_embedding_store_survives_provider_recreation(tmp_path: Path):
    index = _index()
    path = tmp_path / "memory-vectors.sqlite3"
    first_provider = _CountingEmbeddingProvider()
    with SQLiteEmbeddingStore(path) as store:
        first = EmbeddingSignalProvider(
            index,
            first_provider,
            model="hash-16",
            vector_store=store,
        ).score("Rachel team")
    second_provider = _CountingEmbeddingProvider()
    with SQLiteEmbeddingStore(path) as store:
        second = EmbeddingSignalProvider(
            index,
            second_provider,
            model="hash-16",
            vector_store=store,
        ).score("Rachel team")
        stats = store.stats()

    assert first.input_tokens > second.input_tokens
    assert len(second_provider.calls) == 1
    assert second_provider.calls[0] == ["Rachel team"]
    assert stats["hits"] > 0


class _Reasoner:
    def __init__(self):
        self.stages = []

    def reason(self, request):
        self.stages.append(request.stage)
        if request.stage == "retrieval_analysis":
            return ReasoningResponse(
                data={"sufficient": True, "additional_queries": [], "missing_requirements": []},
                model="fixture",
                input_tokens=10,
                output_tokens=4,
                cost_usd=0.001,
                latency_ms=2.0,
            )
        return ReasoningResponse(data={})


def test_quality_profile_uses_reasoning_fallback_and_records_usage():
    reasoner = _Reasoner()
    result = MemoryRuntime("quality", reasoner=reasoner).retrieve(
        _index(),
        "Can you recommend a Miami hotel?",
    )

    telemetry = result.metadata["pipeline_telemetry"]
    assert "retrieval_analysis" in reasoner.stages
    assert telemetry["input_tokens"] == 10
    assert telemetry["output_tokens"] >= 4
    assert telemetry["cost_usd"] == 0.001


class _FailingReasoner:
    def reason(self, request):
        raise RuntimeError("fixture failure")


def test_optional_reasoning_fails_open_and_is_audited():
    result = MemoryRuntime("quality", reasoner=_FailingReasoner()).retrieve(
        _index(),
        "Can you recommend a Miami hotel?",
    )

    failed = [
        stage
        for stage in result.metadata["pipeline_telemetry"]["stages"]
        if stage["metadata"].get("failed")
    ]
    assert failed
    assert all(stage["metadata"]["fail_open"] is True for stage in failed)


class _CompleteReasoner:
    def reason(self, request):
        if request.stage == "query_classification":
            data = request.payload["deterministic_analysis"]
        elif request.stage == "retrieval_strategy":
            data = request.payload["deterministic_strategy"]
        elif request.stage == "retrieval_analysis":
            data = {"sufficient": True, "additional_queries": [], "missing_requirements": []}
        else:
            data = {
                "priority_claim_ids": [],
                "priority_source_ids": [],
                "drop_claim_ids": [],
                "drop_source_ids": [],
                "rationale": "fixture",
            }
        return ReasoningResponse(
            data=data,
            model="fixture",
            input_tokens=5,
            output_tokens=2,
            cost_usd=0.001,
            latency_ms=1.0,
        )


def test_reasoning_budget_limits_calls_and_audits_skipped_stages():
    reasoner = _Reasoner()
    base = runtime_profile("max_quality")
    profile = base.model_copy(
        update={"reasoning_budget": base.reasoning_budget.model_copy(update={"max_calls": 1})}
    )
    result = MemoryRuntime(profile, reasoner=reasoner).retrieve(
        _index(),
        "Can you recommend a Miami hotel based on all of my preferences?",
    )

    assert len(reasoner.stages) == 1
    skipped = [
        stage
        for stage in result.metadata["pipeline_telemetry"]["stages"]
        if stage["metadata"].get("reason") == "reasoning_budget_exhausted"
    ]
    assert skipped


def test_profile_switches_disable_compiler_packet_and_raw_sources():
    profile = runtime_profile("fast").model_copy(
        update={"use_compiled_projection": False, "include_raw_sources": False}
    )
    result = MemoryRuntime(profile).retrieve(_index(), "What do I know about Rachel?")

    assert result.metadata["compiled_evidence"]["operation"] == "disabled"
    assert result.context_text == ""


def test_graph_materialization_is_idempotent_and_links_state_versions():
    graph = Graph()
    index = _index()

    first = materialize_memory_index(graph, index)
    counts = (len(graph.objects()), len(graph.all_relations()))
    second = materialize_memory_index(graph, index)

    assert counts == (len(graph.objects()), len(graph.all_relations()))
    assert first.entity_object_ids == second.entity_object_ids
    assert first.quantity_object_ids == second.quantity_object_ids
    assert first.temporal_ref_object_ids == second.temporal_ref_object_ids
    assert graph.relations(type="memory_version_of")
    assert graph.relations(type="memory_supersedes")


def test_retrieval_trace_materialization_is_idempotent():
    graph = Graph()
    index = _index()
    materialization = materialize_memory_index(graph, index)
    query = graph.add_object(
        "memory_query",
        {"query": "How many women are currently on Rachel's team?"},
    )
    result = MemoryRuntime("fast").retrieve(index, query.data["query"], query_id=query.id)

    first = materialize_retrieval_trace(
        graph,
        query.id,
        result,
        materialization=materialization,
    )
    counts = (len(graph.objects()), len(graph.all_relations()))
    second = materialize_retrieval_trace(
        graph,
        query.id,
        result,
        materialization=materialization,
    )

    assert counts == (len(graph.objects()), len(graph.all_relations()))
    assert first == second


def test_profile_benchmark_reports_latency_context_and_cost():
    reports = benchmark_profiles(
        _index(),
        [MemoryBenchmarkCase("q1", "How many women are currently on Rachel's team?")],
        profiles=("fast", "balanced"),
    )

    assert [report.profile for report in reports] == ["fast", "balanced"]
    assert all(report.latency_mean_ms >= 0 for report in reports)
    assert all(report.mean_context_tokens > 0 for report in reports)
    assert all(report.cost_usd == 0 for report in reports)
    assert all(report.proof_complete_rate == 1.0 for report in reports)
    assert all(report.mean_retrieval_rounds >= 1 for report in reports)
    assert all(0 <= report.sufficiency_rate <= 1 for report in reports)
    assert all(0 <= report.candidate_answer_render_rate <= 1 for report in reports)


def test_ingestion_benchmark_measures_compile_and_graph_materialization():
    report = benchmark_ingestion(
        _index().turns,
        repetitions=2,
        graph_factory=Graph,
    )

    assert report.extractor == "DeterministicMemoryExtractor"
    assert report.turns == len(_index().turns)
    assert report.facts == len(_index().turns)
    assert report.latency_mean_ms >= 0
    assert report.mean_graph_objects > report.turns
    assert report.mean_graph_relations > 0


def test_reasoning_ablation_benchmark_reports_marginal_calls_and_cost():
    reports = benchmark_reasoning_ablations(
        _index(),
        [MemoryBenchmarkCase("q1", "Can you recommend a Miami hotel?")],
        runtime_factory=lambda profile: MemoryRuntime(profile, reasoner=_CompleteReasoner()),
        repetitions=1,
    )

    assert reports["reasoning_off"].reasoning_calls == 0
    assert reports["reasoning_off"].reasoning_cost_usd == 0
    assert reports["reasoning_all"].reasoning_calls > 0
    assert reports["reasoning_all"].reasoning_cost_usd > 0


def test_profiles_expose_distinct_cost_quality_knobs():
    fast = runtime_profile("fast")
    quality = runtime_profile("quality")

    assert fast.token_budget < quality.token_budget
    assert fast.max_retrieval_rounds < quality.max_retrieval_rounds
    assert fast.reasoning.retrieval_analysis == "off"
    assert quality.reasoning.retrieval_analysis == "always"


def test_deterministic_extraction_compiles_raw_turns_without_external_claims():
    turns = [_turn("raw", 0, 0, "user", "I bought a camera for $900 yesterday.", "2023-04-02")]

    claims, extraction = extract_claim_inputs(turns)
    index = compile_memory_index(turns=turns, claims=claims)

    assert extraction.metadata["lossless_turn_fallback"] is True
    assert claims[0].source_turn_ids == ("raw#0",)
    assert index.claims[0].source_turn_ids == ["raw#0"]
    assert index.claims[0].quantity_claims[0].value == 900


class _TypedExtractor:
    def extract(self, turns):
        return MemoryExtractionResult(
            facts=(
                ExtractedMemoryFact(
                    text="The user purchased a portrait lens.",
                    source_turn_ids=[turns[0].turn_id],
                    predicate="purchase",
                    entities=[{"name": "Portrait Lens", "kind": "camera_equipment", "aliases": ["lens"]}],
                    categories=["camera", "expense"],
                    modality="actual",
                    event_start="2023-04-01",
                    event_end="2023-04-01",
                    time_confidence=0.99,
                    quantities=[
                        {
                            "property_name": "money",
                            "value": 900,
                            "unit": "usd",
                            "exactness": "exact",
                            "source_text": "$900",
                            "confidence": 0.98,
                        }
                    ],
                ),
            ),
            extractor="typed-fixture",
        )


def test_typed_extraction_hints_drive_projection_instead_of_being_discarded():
    turns = [_turn("typed", 0, 0, "user", "That lens was nine hundred dollars.", "2023-04-02")]
    claims, _ = extract_claim_inputs(turns, extractor=_TypedExtractor())

    index = compile_memory_index(turns=turns, claims=claims)
    mention = index.compiled.event_mentions[0]
    entity = index.compiled.entities[0]

    assert mention.predicate == "purchase"
    assert mention.event_start == "2023-04-01"
    assert mention.metadata["time_basis"] == "typed_extraction"
    assert entity.kind == "camera_equipment"
    assert set(entity.aliases) == {"Portrait Lens", "lens"}
    assert index.claims[0].quantity_claims[0].value == 900


class _BatchExtractionProvider:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        import json

        payload = json.loads(kwargs["messages"][0].content)
        self.calls.append(payload)
        facts = [
            ExtractedMemoryFact(
                text=f"Remembered: {turn['content']}",
                source_turn_ids=[turn["turn_id"]],
            )
            for turn in payload
        ]
        return LLMResponse(
            raw_text="",
            parsed={"facts": [fact.model_dump() for fact in facts]},
            input_tokens=10,
            output_tokens=5,
            cost_usd=Decimal("0.001"),
            latency_seconds=0.02,
            model=kwargs["model"],
            finish_reason="stop",
            cache_hit=False,
            provider_meta={"fixture": True},
        )


def test_llm_extraction_batches_long_histories_and_aggregates_usage():
    from activegraph_memory import ActiveGraphLLMMemoryExtractor

    turns = [
        _turn(f"batch-{idx}", idx, 0, "user", f"fact {idx}", f"2023-04-{idx + 1:02d}")
        for idx in range(5)
    ]
    provider = _BatchExtractionProvider()
    result = ActiveGraphLLMMemoryExtractor(
        provider,
        model="fixture-model",
        max_turns_per_batch=2,
    ).extract(turns)

    assert [len(call) for call in provider.calls] == [2, 2, 1]
    assert [fact.source_turn_ids[0] for fact in result.facts] == [turn.turn_id for turn in turns]
    assert result.input_tokens == 30
    assert result.output_tokens == 15
    assert result.cost_usd == 0.003
    assert result.latency_ms == 60
    assert result.metadata["batch_sizes"] == [2, 2, 1]


def test_graph_materialization_can_rebuild_runtime_after_restart():
    graph = Graph()
    original = _index()
    materialize_memory_index(graph, original)

    loaded = load_memory_index(graph)
    result = MemoryRuntime("fast").retrieve(loaded, "How many women are currently on Rachel's team?")

    assert len(graph.objects(type="memory_source_turn")) == len(original.turns)
    assert result.metadata["compiled_evidence"]["candidate_answer"] == "6"


def test_graph_repository_append_survives_new_repository_instance():
    graph = Graph()
    old_turn = _turn("old", 0, 0, "user", "Rachel's team has five women.", "2023-01-01")
    new_turn = _turn("new", 1, 0, "user", "Rachel's team now has six women.", "2023-02-01")
    repository = GraphMemoryRepository(graph, runtime=MemoryRuntime("fast"))
    repository.compile(
        turns=[old_turn],
        claims=[ExtractedClaimInput("Rachel's team has five women.", "old", "2023-01-01", 0, "user", (0,))],
    )
    repository.append(
        turns=[new_turn],
        claims=[ExtractedClaimInput("Rachel's team now has six women.", "new", "2023-02-01", 1, "user", (0,))],
    )

    restarted = GraphMemoryRepository(graph, runtime=MemoryRuntime("fast"))
    restarted.load()
    result = restarted.retrieve("How many women are currently on Rachel's team?", query_id="restart-query")

    assert result.metadata["compiled_evidence"]["candidate_answer"] == "6"
    assert len(graph.objects(type="memory_source_turn")) == 2


def test_graph_repository_preserves_ingestion_usage_across_append_and_restart():
    graph = Graph()
    repository = GraphMemoryRepository(graph, runtime=MemoryRuntime("fast"))
    repository.compile(turns=[_turn("first", 0, 0, "user", "I bought a bike.", "2023-01-01")])
    repository.append(turns=[_turn("second", 1, 0, "user", "I sold the bike.", "2023-02-01")])

    assert len(graph.objects(type="memory_ingestion_stage")) == 2
    restarted = GraphMemoryRepository(graph, runtime=MemoryRuntime("fast"))
    loaded = restarted.load()

    assert len(loaded.metadata["ingestion_runs"]) == 2
    assert sum(run["fact_count"] for run in loaded.metadata["ingestion_runs"]) == 2


def test_repository_settings_control_compilation_and_runtime_profile():
    from types import SimpleNamespace

    graph = Graph()
    activegraph_runtime = SimpleNamespace(
        graph=graph,
        llm_provider=None,
        embedding_provider=None,
    )
    settings = ActiveGraphMemorySettings(
        runtime_profile="max_quality",
        enable_claim_extraction=False,
        enable_temporal_resolution=False,
        enable_conflict_detection=False,
        adaptive_retrieval=False,
    )
    repository = GraphMemoryRepository.from_activegraph(activegraph_runtime, settings=settings)
    turn = _turn("settings", 0, 0, "user", "I bought a bike yesterday.", "2023-01-02")

    assert repository.runtime.profile.name == "max_quality"
    assert repository.runtime.profile.adaptive_retrieval is False
    try:
        repository.compile(turns=[turn])
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("disabled extraction must require explicit claims")

    index = repository.compile(
        turns=[turn],
        claims=[ExtractedClaimInput("I bought a bike yesterday.", "settings", "2023-01-02", 0, "user", (0,))],
    )
    assert index.claims[0].temporal_refs == []
    assert index.compiled.conflicts == []


def test_conflicting_state_prevents_calibrated_candidate_rendering():
    turns = [
        _turn("left", 0, 0, "user", "The launch budget is $50,000.", "2023-04-01"),
        _turn("right", 1, 0, "user", "The launch budget is $75,000.", "2023-04-01"),
    ]
    claims = [
        ExtractedClaimInput(
            "The launch budget is $50,000.", "left", "2023-04-01", 0, "user", (0,),
            metadata={"state_key": "launch|budget"},
        ),
        ExtractedClaimInput(
            "The launch budget is $75,000.", "right", "2023-04-01", 1, "user", (0,),
            metadata={"state_key": "launch|budget"},
        ),
    ]

    index = compile_memory_index(turns=turns, claims=claims)
    result = MemoryRuntime("balanced").retrieve(index, "What is the current launch budget?")

    assert index.compiled.conflicts
    assert result.metadata["retrieval_assessment"]["conflict_ids"]
    assert result.metadata["candidate_answer_rendered"] is False
    assert "Proof-complete candidate" not in result.context_text


def test_confidence_driven_retrieval_records_round_queries_and_assessment():
    result = MemoryRuntime("balanced").retrieve(
        _index(),
        "Which happened first, the Alpha launch or the Beta launch?",
    )

    assert result.metadata["retrieval_rounds_used"] == 2
    assert len(result.metadata["retrieval_round_queries"]) == 2
    assert result.metadata["retrieval_assessment"]["sufficient"] is False
    assessment_stages = [
        stage
        for stage in result.metadata["pipeline_telemetry"]["stages"]
        if stage["stage"] == "retrieval_assessment"
    ]
    assert assessment_stages


def test_negative_existence_returns_bounded_certificate_and_counterevidence():
    index = _index()

    found = MemoryRuntime("fast").retrieve(index, "Did I never buy a road bike?")
    absent = MemoryRuntime("fast").retrieve(index, "Did I never buy a sailboat?")

    assert found.metadata["compiled_evidence"]["candidate_answer"] == "Matching evidence exists in memory."
    assert "road bike" in found.metadata["compiled_evidence"]["rows"][0]["counterevidence"]
    assert absent.metadata["compiled_evidence"]["candidate_answer"].startswith("No matching evidence")
    assert absent.metadata["compiled_evidence"]["metadata"]["world_level_absence_claim"] is False
    assert absent.metadata["compiled_evidence"]["proof_complete"] is True
