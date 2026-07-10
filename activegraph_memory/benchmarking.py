"""Profile benchmark helpers for latency, context size, usage, cost, and quality."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .compiler import MemoryIndex
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
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: float | None = None
    per_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def benchmark_runtime(
    index: MemoryIndex,
    cases: Iterable[MemoryBenchmarkCase],
    *,
    profile: str,
    runtime_factory: Callable[[str], MemoryRuntime] | None = None,
    repetitions: int = 1,
    evaluator: Callable[[MemoryBenchmarkCase, Any], float | None] | None = None,
    name: str = "activegraph-memory",
) -> MemoryBenchmarkResult:
    """Benchmark one profile with optional application-specific quality scoring."""

    case_list = list(cases)
    runtime = (runtime_factory or (lambda value: MemoryRuntime(value)))(profile)
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
            record = {
                "case_id": case.case_id,
                "repetition": repetition,
                "latency_ms": elapsed_ms,
                "context_tokens": result.metadata["estimated_context_tokens"],
                "input_tokens": telemetry["input_tokens"],
                "output_tokens": telemetry["output_tokens"],
                "cost_usd": telemetry["cost_usd"],
                "proof_complete": result.metadata["compiled_evidence"]["proof_complete"],
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
        profile=profile,
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


def render_benchmark_markdown(results: Iterable[MemoryBenchmarkResult]) -> str:
    """Render stable, commit-friendly profile comparison rows."""

    lines = [
        "| Profile | Cases | Mean ms | P95 ms | Cold ms | Warm mean ms | Context tokens | Proof rate | Cost USD | Quality |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                (
                    result.profile,
                    str(result.cases),
                    f"{result.latency_mean_ms:.3f}",
                    f"{result.latency_p95_ms:.3f}",
                    f"{result.cold_latency_ms:.3f}",
                    "" if result.warm_latency_mean_ms is None else f"{result.warm_latency_mean_ms:.3f}",
                    f"{result.mean_context_tokens:.2f}",
                    f"{result.proof_complete_rate:.4f}",
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
