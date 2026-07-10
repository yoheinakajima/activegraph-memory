"""Profile-driven, instrumented memory retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .assessment import RetrievalAssessment, assess_retrieval
from .compiler import MemoryIndex
from .executor import CompiledEvidence, execute_query
from .object_types import MemoryQuery
from .profiles import MemoryRuntimeProfile, ReasoningMode, runtime_profile
from .query_ir import QueryAnalysis, analyze_query
from .ranking import (
    EmbeddingSignalProvider,
    RetrievalSignalProvider,
    RetrievalSignals,
    fuse_signals,
    lexical_signals,
    merge_rounds,
    propagate_graph_signals,
)
from .reasoning import (
    ActiveGraphLLMReasoningBackend,
    ContextPackagingReasoning,
    ReasoningBackend,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningStage,
    RetrievalAnalysisReasoning,
    RetrievalStrategyReasoning,
)
from .retrieval import MemoryRetrievalResult, retrieve_memory
from .telemetry import PipelineTelemetry, StageTelemetry
from .taxonomy import expanded_query_variants


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class MemoryRuntimeContext:
    analysis: QueryAnalysis
    evidence: CompiledEvidence
    telemetry: PipelineTelemetry
    profile: MemoryRuntimeProfile


class MemoryRuntime:
    """Execute deterministic and optional reasoned memory stages."""

    def __init__(
        self,
        profile: MemoryRuntimeProfile | str = "balanced",
        *,
        reasoner: ReasoningBackend | None = None,
        signal_provider: RetrievalSignalProvider | None = None,
        embedding_provider=None,
        embedding_model: str | None = None,
        embedding_cost_per_million_tokens: float = 0.0,
        embedding_store=None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.profile = runtime_profile(profile) if isinstance(profile, str) else profile
        self.reasoner = reasoner
        self.signal_provider = signal_provider
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_cost_per_million_tokens = embedding_cost_per_million_tokens
        self.embedding_store = embedding_store
        self.token_counter = token_counter or _rough_token_count

    @classmethod
    def from_activegraph(
        cls,
        activegraph_runtime,
        profile: MemoryRuntimeProfile | str = "balanced",
        *,
        reasoning_model: str | None = None,
        embedding_model: str | None = None,
        embedding_cost_per_million_tokens: float = 0.0,
        token_counter: TokenCounter | None = None,
        embedding_store=None,
    ) -> "MemoryRuntime":
        """Bind to ActiveGraph's configured LLM and embedding provider seams."""

        reasoner = None
        if activegraph_runtime.llm_provider is not None and reasoning_model:
            reasoner = ActiveGraphLLMReasoningBackend(
                activegraph_runtime.llm_provider,
                model=reasoning_model,
            )
        return cls(
            profile,
            reasoner=reasoner,
            embedding_provider=activegraph_runtime.embedding_provider,
            embedding_model=embedding_model,
            embedding_cost_per_million_tokens=embedding_cost_per_million_tokens,
            embedding_store=embedding_store,
            token_counter=token_counter,
        )

    def retrieve(
        self,
        index: MemoryIndex,
        query: MemoryQuery | str,
        *,
        query_id: str = "query",
        question_date: str | None = None,
        claim_scores: dict[str, float] | None = None,
        turn_scores: dict[str, float] | None = None,
        retrieval_plan=None,
        signal_provider: RetrievalSignalProvider | None = None,
    ) -> MemoryRetrievalResult:
        memory_query = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query)
        telemetry = PipelineTelemetry(profile=self.profile.name)

        with telemetry.measure("query_classification", "deterministic_query_ir") as stage:
            analysis = analyze_query(memory_query, question_date=question_date)
            stage.candidates_out = len(analysis.operators)
            stage.metadata.update({"confidence": analysis.deterministic_confidence})
        reasoned = self._reason_stage(
            "query_classification",
            mode=self.profile.reasoning.query_classification,
            fallback=analysis.deterministic_confidence < 0.8,
            payload={"query": memory_query.model_dump(), "deterministic_analysis": analysis.model_dump()},
            output_contract=QueryAnalysis.model_json_schema(),
            telemetry=telemetry,
        )
        if reasoned is not None:
            analysis = QueryAnalysis.model_validate({**analysis.model_dump(), **reasoned.data})

        strategy_data = {
            "query_variants": _targeted_variants(analysis),
            "token_budget": self.profile.token_budget,
            "max_retrieval_rounds": self.profile.max_retrieval_rounds,
        }
        strategy_reasoning = self._reason_stage(
            "retrieval_strategy",
            mode=self.profile.reasoning.retrieval_strategy,
            fallback=analysis.requires_reasoning,
            payload={"analysis": analysis.model_dump(), "deterministic_strategy": strategy_data},
            output_contract={
                "type": "object",
                "properties": {
                    "query_variants": {"type": "array", "items": {"type": "string"}},
                    "token_budget": {"type": "integer"},
                    "max_retrieval_rounds": {"type": "integer"},
                },
            },
            telemetry=telemetry,
        )
        if strategy_reasoning is not None:
            strategy_data = _merge_strategy(strategy_data, strategy_reasoning.data, self.profile)

        provider = signal_provider or self.signal_provider
        if provider is None and self.embedding_provider is not None and self.profile.use_embeddings:
            provider = EmbeddingSignalProvider(
                index,
                self.embedding_provider,
                model=self.embedding_model,
                embed_entities=self.profile.embed_entities,
                embed_events=self.profile.embed_events,
                estimate_tokens=self.token_counter,
                cost_per_million_tokens=self.embedding_cost_per_million_tokens,
                vector_store=self.embedding_store,
            )
        max_rounds = min(
            self.profile.max_retrieval_rounds,
            int(strategy_data.get("max_retrieval_rounds") or self.profile.max_retrieval_rounds),
        )
        rounds: list[RetrievalSignals] = []
        round_queries: list[str] = []

        def score_variant(value: str, *, include_caller_scores: bool = False) -> RetrievalSignals:
            signal_sets = [lexical_signals(index, value)]
            if include_caller_scores and (claim_scores or turn_scores):
                signal_sets.append(
                    RetrievalSignals(
                        claim_scores=dict(claim_scores or {}),
                        turn_scores=dict(turn_scores or {}),
                        metadata={"provider": "caller_scores"},
                    )
                )
            if provider is not None and self.profile.use_embeddings:
                signal_sets.append(provider.score(value))
            return fuse_signals(signal_sets) if self.profile.use_rank_fusion else signal_sets[-1]

        with telemetry.measure("candidate_generation", "hybrid_retrieval") as stage:
            rounds.append(score_variant(analysis.query, include_caller_scores=True))
            round_queries.append(analysis.query)
            signals = propagate_graph_signals(index, merge_rounds(rounds))
            stage.input_tokens = signals.input_tokens
            stage.cost_usd = signals.cost_usd
            stage.candidates_in = len(index.claims) + len(index.turns)
            stage.candidates_out = len(signals.claim_scores) + len(signals.turn_scores)
            stage.metadata.update({"rounds": len(rounds), "queries": list(round_queries), **signals.metadata})

        with telemetry.measure("operator_execution", "typed_projection_executor") as stage:
            if self.profile.use_compiled_projection:
                evidence = execute_query(index, analysis, signals)
            else:
                evidence = CompiledEvidence(
                    operation="disabled",
                    metadata={"disabled_by_profile": True},
                )
            stage.candidates_in = len(index.compiled.canonical_events) + len(index.compiled.state_versions)
            stage.candidates_out = len(evidence.rows)
            stage.metadata.update(
                {
                    "operation": evidence.operation,
                    "proof_complete": evidence.proof_complete,
                    "missing_requirements": evidence.missing_requirements,
                }
            )

        with telemetry.measure("retrieval_assessment", "deterministic_sufficiency") as stage:
            assessment = assess_retrieval(
                index,
                analysis,
                evidence,
                signals,
                round_index=len(rounds),
                min_confidence=self.profile.min_sufficiency_confidence,
            )
            stage.candidates_in = len(evidence.rows)
            stage.candidates_out = len(assessment.next_queries)
            stage.metadata.update(assessment.model_dump())

        analysis_reasoning = self._reason_stage(
            "retrieval_analysis",
            mode=self.profile.reasoning.retrieval_analysis,
            fallback=not assessment.sufficient,
            payload={
                "analysis": analysis.model_dump(),
                "compiled_evidence": evidence.__dict__,
                "deterministic_assessment": assessment.model_dump(),
            },
            output_contract={
                "type": "object",
                "properties": {
                    "sufficient": {"type": "boolean"},
                    "additional_queries": {"type": "array", "items": {"type": "string"}},
                    "missing_requirements": {"type": "array", "items": {"type": "string"}},
                },
            },
            telemetry=telemetry,
        )
        expansion_queries = list(assessment.next_queries)
        if analysis_reasoning is not None:
            expansion_queries.extend(
                str(value)
                for value in analysis_reasoning.data.get("additional_queries", [])
                if str(value).strip()
            )
        expansion_queries.extend(list(strategy_data.get("query_variants") or [])[1:])
        expansion_queries = _dedupe_queries(expansion_queries, exclude=set(round_queries))
        if self.profile.adaptive_retrieval and not assessment.sufficient and len(rounds) < max_rounds:
            with telemetry.measure("targeted_expansion", "confidence_driven_query_expansion") as stage:
                executed = []
                assessments = []
                for variant in expansion_queries:
                    if len(rounds) >= max_rounds or assessment.sufficient:
                        break
                    rounds.append(score_variant(variant))
                    round_queries.append(variant)
                    executed.append(variant)
                    signals = propagate_graph_signals(index, merge_rounds(rounds))
                    if self.profile.use_compiled_projection:
                        evidence = execute_query(index, analysis, signals)
                    assessment = assess_retrieval(
                        index,
                        analysis,
                        evidence,
                        signals,
                        round_index=len(rounds),
                        min_confidence=self.profile.min_sufficiency_confidence,
                    )
                    assessments.append(assessment.model_dump())
                stage.input_tokens = sum(item.input_tokens for item in rounds[1:])
                stage.cost_usd = sum(item.cost_usd for item in rounds[1:])
                stage.candidates_out = len(evidence.rows)
                stage.metadata.update(
                    {
                        "queries": executed,
                        "stopped_sufficient": assessment.sufficient,
                        "assessments": assessments,
                    }
                )

        _boost_compiled_evidence(signals, evidence)
        candidate_answer_rendered = _should_render_candidate(self.profile, evidence, assessment)
        packet = evidence.render(
            max_rows=self.profile.max_packet_rows if self.profile.compact_context else self.profile.max_packet_rows * 2,
            include_candidate=candidate_answer_rendered,
        )
        packet_tokens = self.token_counter(packet) if packet else 0
        requested_budget = max(256, int(strategy_data.get("token_budget") or self.profile.token_budget))
        source_ceiling = max(256, int(requested_budget * self.profile.source_budget_ratio))
        base_budget = max(256, source_ceiling - packet_tokens - 16)
        with telemetry.measure("source_packaging", "provenance_context_assembler") as stage:
            result = retrieve_memory(
                index,
                memory_query,
                query_id=query_id,
                question_date=question_date,
                token_budget=base_budget,
                claim_scores=signals.claim_scores,
                turn_scores=signals.turn_scores,
                token_counter=self.token_counter,
                retrieval_plan=retrieval_plan,
                enable_graph_query=self.profile.use_graph_reducers,
                max_claims_per_session=self.profile.max_claims_per_session if self.profile.use_diversity_selection else None,
                max_direct_turns_per_session=self.profile.max_direct_turns_per_session if self.profile.use_diversity_selection else None,
                enable_preference_packet=False,
                allowed_source_roles=set(analysis.source_roles),
            )
            if packet:
                result.context_text = f"{packet}\n\n{result.context_text}" if result.context_text else packet
            if not self.profile.include_raw_sources:
                result.context_text = packet
            stage.output_tokens = self.token_counter(result.context_text)
            stage.candidates_out = len(result.selected_claim_ids) + len(result.selected_turn_ids)

        packaging_reasoning = self._reason_stage(
            "context_packaging",
            mode=self.profile.reasoning.context_packaging,
            fallback=self.token_counter(result.context_text) > requested_budget or not assessment.sufficient,
            payload={
                "query": analysis.query,
                "analysis": analysis.model_dump(),
                "compiled_evidence": evidence.__dict__,
                "retrieval_assessment": assessment.model_dump(),
                "selected_claim_ids": result.selected_claim_ids,
                "selected_source_ids": result.selected_turn_ids,
                "token_budget": requested_budget,
                "instruction": "Select only from the supplied evidence ids. Do not write or rewrite evidence text.",
            },
            output_contract=ContextPackagingReasoning.model_json_schema(),
            telemetry=telemetry,
        )
        if packaging_reasoning is not None and self.profile.include_raw_sources:
            known_claims = set(result.selected_claim_ids)
            known_sources = set(result.selected_turn_ids)
            priority_claims = _known_ids(packaging_reasoning.data.get("priority_claim_ids"), known_claims)
            priority_sources = _known_ids(packaging_reasoning.data.get("priority_source_ids"), known_sources)
            drop_claims = set(_known_ids(packaging_reasoning.data.get("drop_claim_ids"), known_claims))
            drop_sources = set(_known_ids(packaging_reasoning.data.get("drop_source_ids"), known_sources))
            if priority_claims or priority_sources or drop_claims or drop_sources:
                for offset, claim_id in enumerate(priority_claims):
                    signals.claim_scores[claim_id] = max(2.0 - (offset * 0.01), signals.claim_scores.get(claim_id, 0.0))
                for offset, source_id in enumerate(priority_sources):
                    signals.turn_scores[source_id] = max(2.0 - (offset * 0.01), signals.turn_scores.get(source_id, 0.0))
                with telemetry.measure("reasoned_repackaging", "validated_evidence_selection") as stage:
                    result = retrieve_memory(
                        index,
                        memory_query,
                        query_id=query_id,
                        question_date=question_date,
                        token_budget=base_budget,
                        claim_scores=signals.claim_scores,
                        turn_scores=signals.turn_scores,
                        token_counter=self.token_counter,
                        retrieval_plan=retrieval_plan,
                        enable_graph_query=self.profile.use_graph_reducers,
                        max_claims_per_session=self.profile.max_claims_per_session if self.profile.use_diversity_selection else None,
                        max_direct_turns_per_session=self.profile.max_direct_turns_per_session if self.profile.use_diversity_selection else None,
                        enable_preference_packet=False,
                        exclude_claim_ids=drop_claims,
                        exclude_turn_ids=drop_sources,
                        allowed_source_roles=set(analysis.source_roles),
                    )
                    if packet:
                        result.context_text = f"{packet}\n\n{result.context_text}" if result.context_text else packet
                    stage.candidates_in = len(known_claims) + len(known_sources)
                    stage.candidates_out = len(result.selected_claim_ids) + len(result.selected_turn_ids)
                    stage.output_tokens = self.token_counter(result.context_text)

        result.metadata.update(
            {
                "runtime_profile": self.profile.model_dump(),
                "query_analysis": analysis.model_dump(),
                "compiled_evidence": evidence.__dict__,
                "retrieval_assessment": assessment.model_dump(),
                "pipeline_telemetry": telemetry.as_dict(),
                "requested_pipeline_token_budget": requested_budget,
                "source_context_token_ceiling": source_ceiling,
                "retrieval_rounds_used": len(rounds),
                "retrieval_round_queries": round_queries,
                "candidate_answer_rendered": candidate_answer_rendered,
                "estimated_context_tokens": self.token_counter(result.context_text),
            }
        )
        result.evidence_bundle.metadata.update(
            {
                "compiled_event_ids": evidence.selected_event_ids,
                "compiled_proof_complete": evidence.proof_complete,
                "retrieval_sufficient": assessment.sufficient,
                "conflict_ids": assessment.conflict_ids,
            }
        )
        return result

    def _reason_stage(
        self,
        stage: ReasoningStage,
        *,
        mode: ReasoningMode,
        fallback: bool,
        payload: dict[str, Any],
        output_contract: dict[str, Any],
        telemetry: PipelineTelemetry,
    ) -> ReasoningResponse | None:
        should_run = mode == "always" or (mode == "fallback" and fallback)
        if not should_run or self.reasoner is None:
            return None
        allowed, budget_state = _reasoning_budget_allows(self.profile, telemetry)
        if not allowed:
            telemetry.stages.append(
                StageTelemetry(
                    stage=stage,
                    implementation=f"reasoner:{type(self.reasoner).__name__}",
                    metadata={
                        "skipped": True,
                        "reason": "reasoning_budget_exhausted",
                        "budget_state": budget_state,
                    },
                )
            )
            return None
        request = ReasoningRequest(stage=stage, payload=payload, output_contract=output_contract)
        response: ReasoningResponse | None = None
        try:
            response = self.reasoner.reason(request)
            schema = {
                "query_classification": QueryAnalysis,
                "retrieval_strategy": RetrievalStrategyReasoning,
                "retrieval_analysis": RetrievalAnalysisReasoning,
                "context_packaging": ContextPackagingReasoning,
            }[stage]
            validated = schema.model_validate(response.data).model_dump()
            response = ReasoningResponse(
                data=validated,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                cached=response.cached,
                metadata=response.metadata,
            )
        except Exception as exc:
            telemetry.stages.append(
                StageTelemetry(
                    stage=stage,
                    implementation=f"reasoner:{type(self.reasoner).__name__}",
                    duration_ms=response.latency_ms if response is not None else 0.0,
                    input_tokens=response.input_tokens if response is not None else 0,
                    output_tokens=response.output_tokens if response is not None else 0,
                    cost_usd=response.cost_usd if response is not None else 0.0,
                    cached=response.cached if response is not None else False,
                    metadata={
                        "failed": True,
                        "error_type": type(exc).__name__,
                        "fail_open": self.profile.reasoning_fail_open,
                        "model": response.model if response is not None else "",
                    },
                )
            )
            if self.profile.reasoning_fail_open:
                return None
            raise
        telemetry.stages.append(
            StageTelemetry(
                stage=stage,
                implementation=f"reasoner:{type(self.reasoner).__name__}",
                duration_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                cached=response.cached,
                metadata={
                    "model": response.model,
                    "decision": response.data,
                    **response.metadata,
                },
            )
        )
        return response


def _targeted_variants(analysis: QueryAnalysis) -> list[str]:
    out = [analysis.query]
    out.extend(expanded_query_variants(analysis.query))
    for operand in analysis.operands:
        out.append(f"{analysis.primary_operator} {operand}")
    if analysis.requires_exhaustive_coverage and analysis.entity_terms:
        out.append(" ".join(analysis.entity_terms))
    seen: set[str] = set()
    return [value for value in out if value and not (value.lower() in seen or seen.add(value.lower()))]


def _merge_strategy(
    base: dict[str, Any],
    reasoned: dict[str, Any],
    profile: MemoryRuntimeProfile,
) -> dict[str, Any]:
    out = dict(base)
    if isinstance(reasoned.get("query_variants"), list):
        variants = [str(value) for value in reasoned["query_variants"] if str(value).strip()]
        if variants:
            out["query_variants"] = [base["query_variants"][0], *variants]
    if reasoned.get("token_budget") is not None:
        out["token_budget"] = max(256, min(profile.token_budget, int(reasoned["token_budget"])))
    if reasoned.get("max_retrieval_rounds") is not None:
        out["max_retrieval_rounds"] = max(1, min(profile.max_retrieval_rounds, int(reasoned["max_retrieval_rounds"])))
    return out


def _boost_compiled_evidence(signals: RetrievalSignals, evidence: CompiledEvidence) -> None:
    for claim_id in evidence.selected_claim_ids:
        signals.claim_scores[claim_id] = max(1.15, signals.claim_scores.get(claim_id, 0.0))
    for turn_id in evidence.selected_turn_ids:
        signals.turn_scores[turn_id] = max(1.15, signals.turn_scores.get(turn_id, 0.0))


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _known_ids(values, known: set[str]) -> list[str]:
    return [str(value) for value in (values or []) if str(value) in known]


def _dedupe_queries(values, *, exclude: set[str] | None = None) -> list[str]:
    seen = {value.lower().strip() for value in (exclude or set())}
    out = []
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _should_render_candidate(
    profile: MemoryRuntimeProfile,
    evidence: CompiledEvidence,
    assessment: RetrievalAssessment,
) -> bool:
    if not evidence.candidate_answer or profile.candidate_answer_mode == "never":
        return False
    if profile.candidate_answer_mode == "proof_complete":
        return evidence.proof_complete
    return bool(
        evidence.proof_complete
        and assessment.sufficient
        and evidence.confidence >= profile.min_sufficiency_confidence
        and not assessment.conflict_ids
    )


def _reasoning_budget_allows(
    profile: MemoryRuntimeProfile,
    telemetry: PipelineTelemetry,
) -> tuple[bool, dict[str, float]]:
    stages = [
        stage
        for stage in telemetry.stages
        if stage.implementation.startswith("reasoner:")
        and not stage.metadata.get("skipped")
    ]
    state = {
        "calls": len(stages),
        "input_tokens": sum(stage.input_tokens for stage in stages),
        "output_tokens": sum(stage.output_tokens for stage in stages),
        "cost_usd": round(sum(stage.cost_usd for stage in stages), 8),
        "latency_ms": round(sum(stage.duration_ms for stage in stages), 3),
    }
    budget = profile.reasoning_budget
    checks = (
        state["calls"] < budget.max_calls,
        state["input_tokens"] < budget.max_input_tokens,
        state["output_tokens"] < budget.max_output_tokens,
        state["cost_usd"] < budget.max_cost_usd,
        state["latency_ms"] < budget.max_latency_ms,
    )
    return all(checks), state
