from __future__ import annotations

from pathlib import Path

from activegraph import Graph
from activegraph.llm import HashEmbeddingProvider

from activegraph_memory import (
    EmbeddingSignalProvider,
    ExtractedClaimInput,
    MemoryBenchmarkCase,
    MemoryRuntime,
    ReasoningResponse,
    SQLiteEmbeddingStore,
    SourceTurn,
    analyze_query,
    benchmark_profiles,
    compile_memory_index,
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


def test_profiles_expose_distinct_cost_quality_knobs():
    fast = runtime_profile("fast")
    quality = runtime_profile("quality")

    assert fast.token_budget < quality.token_budget
    assert fast.max_retrieval_rounds < quality.max_retrieval_rounds
    assert fast.reasoning.retrieval_analysis == "off"
    assert quality.reasoning.retrieval_analysis == "always"
