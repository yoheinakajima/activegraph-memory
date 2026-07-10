"""Profile benchmark helpers for latency, context size, usage, cost, and quality."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .compiler import MemoryIndex
from .compiler import SourceTurn, compile_memory_index
from .extraction import MemoryExtractor, extract_claim_inputs
from .profiles import MemoryRuntimeProfile, StageReasoningPolicy, runtime_profile
from .runtime import MemoryRuntime


@dataclass(frozen=True)
class MemoryBenchmarkCase:
    case_id: str
    query: str
    question_date: str | None = None
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryBenchmarkResult:
    name: str
    profile: str
    cases: int
    repetitions: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    cold_latency_ms: float
    warm_latency_mean_ms: float | None
    mean_context_tokens: float
    proof_complete_rate: float
    sufficiency_rate: float
    mean_retrieval_rounds: float
    candidate_answer_render_rate: float
    reasoning_calls: int
    reasoning_cost_usd: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: float | None = None
    per_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class MemoryIngestionBenchmarkResult:
    name: str
    extractor: str
    turns: int
    facts: int
    repetitions: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    cold_latency_ms: float
    warm_latency_mean_ms: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    mean_graph_objects: float
    mean_graph_relations: float
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def benchmark_ingestion(
    turns: Iterable[SourceTurn],
    *,
    extractor: MemoryExtractor | None = None,
    repetitions: int = 1,
    graph_factory: Callable[[], Any] | None = None,
    name: str = "activegraph-memory-ingestion",
) -> MemoryIngestionBenchmarkResult:
    """Measure typed extraction, compilation, and optional graph materialization."""

    turn_list = list(turns)
    records = []
    extractor_name = ""
    facts = 0
    for repetition in range(repetitions):
        started = time.perf_counter()
        claims, extraction = extract_claim_inputs(turn_list, extractor=extractor)
        index = compile_memory_index(turns=turn_list, claims=claims)
        graph_objects = 0
        graph_relations = 0
        if graph_factory is not None:
            from .graph_runtime import materialize_memory_index

            graph = graph_factory()
            materialize_memory_index(graph, index)
            graph_objects = len(graph.objects())
            graph_relations = len(graph.relations())
        latency_ms = (time.perf_counter() - started) * 1000.0
        extractor_name = extraction.extractor
        facts = len(extraction.facts)
        records.append(
            {
                "repetition": repetition,
                "latency_ms": latency_ms,
                "facts": facts,
                "input_tokens": extraction.input_tokens,
                "output_tokens": extraction.output_tokens,
                "cost_usd": extraction.cost_usd,
                "cached": extraction.cached,
                "graph_objects": graph_objects,
                "graph_relations": graph_relations,
                "compiled_entities": len(index.compiled.entities),
                "compiled_events": len(index.compiled.canonical_events),
                "compiled_states": len(index.compiled.state_versions),
                "compiled_preferences": len(index.compiled.preferences),
                "compiled_conflicts": len(index.compiled.conflicts),
            }
        )
    latencies = [record["latency_ms"] for record in records]
    warm = latencies[1:]
    return MemoryIngestionBenchmarkResult(
        name=name,
        extractor=extractor_name,
        turns=len(turn_list),
        facts=facts,
        repetitions=repetitions,
        latency_mean_ms=round(statistics.fmean(latencies), 3) if latencies else 0.0,
        latency_p50_ms=round(_percentile(latencies, 0.5), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
        cold_latency_ms=round(latencies[0], 3) if latencies else 0.0,
        warm_latency_mean_ms=round(statistics.fmean(warm), 3) if warm else None,
        input_tokens=sum(record["input_tokens"] for record in records),
        output_tokens=sum(record["output_tokens"] for record in records),
        cost_usd=round(sum(record["cost_usd"] for record in records), 8),
        mean_graph_objects=round(statistics.fmean(record["graph_objects"] for record in records), 2) if records else 0.0,
        mean_graph_relations=round(statistics.fmean(record["graph_relations"] for record in records), 2) if records else 0.0,
        records=records,
    )


def benchmark_runtime(
    index: MemoryIndex,
    cases: Iterable[MemoryBenchmarkCase],
    *,
    profile: str | MemoryRuntimeProfile,
    runtime_factory: Callable[[str | MemoryRuntimeProfile], MemoryRuntime] | None = None,
    repetitions: int = 1,
    evaluator: Callable[[MemoryBenchmarkCase, Any], float | None] | None = None,
    name: str = "activegraph-memory",
) -> MemoryBenchmarkResult:
    """Benchmark one profile with optional application-specific quality scoring."""

    case_list = list(cases)
    resolved_profile = runtime_profile(profile) if isinstance(profile, str) else profile
    runtime = (runtime_factory or (lambda value: MemoryRuntime(value)))(resolved_profile)
    records = []
    quality_values = []
    stage_durations: dict[str, list[float]] = {}
    stage_input_tokens: dict[str, list[float]] = {}
    stage_output_tokens: dict[str, list[float]] = {}
    stage_costs: dict[str, list[float]] = {}
    for repetition in range(repetitions):
        for case in case_list:
            started = time.perf_counter()
            result = runtime.retrieve(
                index,
                case.query,
                query_id=case.case_id,
                question_date=case.question_date,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            telemetry = result.metadata["pipeline_telemetry"]
            assessment = result.metadata.get("retrieval_assessment") or {}
            reasoning_stages = [
                stage
                for stage in telemetry["stages"]
                if str(stage.get("implementation") or "").startswith("reasoner:")
                and not (stage.get("metadata") or {}).get("failed")
            ]
            record = {
                "case_id": case.case_id,
                "repetition": repetition,
                "latency_ms": elapsed_ms,
                "context_tokens": result.metadata["estimated_context_tokens"],
                "input_tokens": telemetry["input_tokens"],
                "output_tokens": telemetry["output_tokens"],
                "cost_usd": telemetry["cost_usd"],
                "proof_complete": result.metadata["compiled_evidence"]["proof_complete"],
                "retrieval_sufficient": bool(assessment.get("sufficient")),
                "retrieval_rounds": int(result.metadata.get("retrieval_rounds_used") or 1),
                "candidate_answer_rendered": bool(result.metadata.get("candidate_answer_rendered")),
                "reasoning_calls": len(reasoning_stages),
                "reasoning_cost_usd": sum(float(stage.get("cost_usd") or 0.0) for stage in reasoning_stages),
            }
            if evaluator is not None:
                evaluated = evaluator(case, result)
                if evaluated is not None:
                    record["quality"] = float(evaluated)
                    quality_values.append(record["quality"])
            records.append(record)
            for stage in telemetry["stages"]:
                stage_durations.setdefault(stage["stage"], []).append(float(stage["duration_ms"]))
                stage_input_tokens.setdefault(stage["stage"], []).append(float(stage["input_tokens"]))
                stage_output_tokens.setdefault(stage["stage"], []).append(float(stage["output_tokens"]))
                stage_costs.setdefault(stage["stage"], []).append(float(stage["cost_usd"]))

    latencies = [record["latency_ms"] for record in records]
    contexts = [record["context_tokens"] for record in records]
    warm_latencies = latencies[1:]
    return MemoryBenchmarkResult(
        name=name,
        profile=resolved_profile.name,
        cases=len(case_list),
        repetitions=repetitions,
        latency_mean_ms=round(statistics.fmean(latencies), 3) if latencies else 0.0,
        latency_p50_ms=round(_percentile(latencies, 0.5), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
        cold_latency_ms=round(latencies[0], 3) if latencies else 0.0,
        warm_latency_mean_ms=(
            round(statistics.fmean(warm_latencies), 3) if warm_latencies else None
        ),
        mean_context_tokens=round(statistics.fmean(contexts), 2) if contexts else 0.0,
        proof_complete_rate=(
            round(
                sum(bool(record["proof_complete"]) for record in records) / len(records),
                4,
            )
            if records
            else 0.0
        ),
        sufficiency_rate=(
            round(sum(bool(record["retrieval_sufficient"]) for record in records) / len(records), 4)
            if records
            else 0.0
        ),
        mean_retrieval_rounds=(
            round(statistics.fmean(record["retrieval_rounds"] for record in records), 3)
            if records
            else 0.0
        ),
        candidate_answer_render_rate=(
            round(sum(bool(record["candidate_answer_rendered"]) for record in records) / len(records), 4)
            if records
            else 0.0
        ),
        reasoning_calls=sum(record["reasoning_calls"] for record in records),
        reasoning_cost_usd=round(sum(record["reasoning_cost_usd"] for record in records), 8),
        input_tokens=sum(record["input_tokens"] for record in records),
        output_tokens=sum(record["output_tokens"] for record in records),
        cost_usd=round(sum(record["cost_usd"] for record in records), 8),
        quality_score=round(statistics.fmean(quality_values), 4) if quality_values else None,
        per_stage={
            stage: {
                "mean_ms": round(statistics.fmean(values), 3),
                "p95_ms": round(_percentile(values, 0.95), 3),
                "input_tokens": round(sum(stage_input_tokens.get(stage, [])), 3),
                "output_tokens": round(sum(stage_output_tokens.get(stage, [])), 3),
                "cost_usd": round(sum(stage_costs.get(stage, [])), 8),
                "invocations": len(values),
            }
            for stage, values in stage_durations.items()
        },
        records=records,
    )


def benchmark_profiles(
    index: MemoryIndex,
    cases: Iterable[MemoryBenchmarkCase],
    profiles: Iterable[str] = ("fast", "balanced", "quality", "max_quality"),
    **kwargs,
) -> list[MemoryBenchmarkResult]:
    case_list = list(cases)
    return [benchmark_runtime(index, case_list, profile=profile, **kwargs) for profile in profiles]


def reasoning_ablation_profiles(
    base_profile: str = "quality",
) -> dict[str, MemoryRuntimeProfile]:
    """Return isolated and all-stage reasoning policies on one fixed profile."""

    base = runtime_profile(base_profile)
    policies = {
        "reasoning_off": StageReasoningPolicy(),
        "classification_only": StageReasoningPolicy(query_classification="always"),
        "strategy_only": StageReasoningPolicy(retrieval_strategy="always"),
        "analysis_only": StageReasoningPolicy(retrieval_analysis="always"),
        "packaging_only": StageReasoningPolicy(context_packaging="always"),
        "reasoning_all": StageReasoningPolicy(
            query_classification="always",
            retrieval_strategy="always",
            retrieval_analysis="always",
            context_packaging="always",
        ),
    }
    return {
        label: base.model_copy(update={"reasoning": policy})
        for label, policy in policies.items()
    }


def benchmark_reasoning_ablations(
    index: MemoryIndex,
    cases: Iterable[MemoryBenchmarkCase],
    *,
    runtime_factory: Callable[[str | MemoryRuntimeProfile], MemoryRuntime],
    base_profile: str = "quality",
    **kwargs,
) -> dict[str, MemoryBenchmarkResult]:
    """Measure marginal latency/cost/quality for each reasoning stage."""

    case_list = list(cases)
    return {
        label: benchmark_runtime(
            index,
            case_list,
            profile=profile,
            runtime_factory=runtime_factory,
            name=label,
            **kwargs,
        )
        for label, profile in reasoning_ablation_profiles(base_profile).items()
    }


def runtime_option_profiles(base_profile: str = "quality") -> dict[str, MemoryRuntimeProfile]:
    """Return isolated deterministic feature switches on one fixed budget."""

    base = runtime_profile(base_profile)
    return {
        "full": base,
        "embeddings_off": base.model_copy(update={"use_embeddings": False}),
        "adaptive_retrieval_off": base.model_copy(update={"adaptive_retrieval": False}),
        "compiled_projection_off": base.model_copy(update={"use_compiled_projection": False}),
        "raw_sources_off": base.model_copy(update={"include_raw_sources": False}),
        "candidate_answer_off": base.model_copy(update={"candidate_answer_mode": "never"}),
    }


def benchmark_runtime_options(
    index: MemoryIndex,
    cases: Iterable[MemoryBenchmarkCase],
    *,
    base_profile: str = "quality",
    runtime_factory: Callable[[str | MemoryRuntimeProfile], MemoryRuntime] | None = None,
    **kwargs,
) -> dict[str, MemoryBenchmarkResult]:
    """Benchmark marginal deterministic features at a constant profile budget."""

    case_list = list(cases)
    return {
        label: benchmark_runtime(
            index,
            case_list,
            profile=profile,
            runtime_factory=runtime_factory,
            name=label,
            **kwargs,
        )
        for label, profile in runtime_option_profiles(base_profile).items()
    }


def render_benchmark_markdown(results: Iterable[MemoryBenchmarkResult]) -> str:
    """Render stable, commit-friendly profile comparison rows."""

    lines = [
        "| Profile | Cases | Mean ms | P95 ms | Context tokens | Rounds | Sufficient | Proof rate | Reason calls | Cost USD | Quality |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        label = result.profile if result.name == "activegraph-memory" else result.name
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(result.cases),
                    f"{result.latency_mean_ms:.3f}",
                    f"{result.latency_p95_ms:.3f}",
                    f"{result.mean_context_tokens:.2f}",
                    f"{result.mean_retrieval_rounds:.3f}",
                    f"{result.sufficiency_rate:.4f}",
                    f"{result.proof_complete_rate:.4f}",
                    str(result.reasoning_calls),
                    f"{result.cost_usd:.8f}",
                    "" if result.quality_score is None else f"{result.quality_score:.4f}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]
