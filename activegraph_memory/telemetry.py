"""Per-stage latency, token, cost, and candidate telemetry."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class StageTelemetry:
    stage: str
    implementation: str
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    candidates_in: int = 0
    candidates_out: int = 0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTelemetry:
    profile: str
    stages: list[StageTelemetry] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return round(sum(stage.duration_ms for stage in self.stages), 3)

    @property
    def input_tokens(self) -> int:
        return sum(stage.input_tokens for stage in self.stages)

    @property
    def output_tokens(self) -> int:
        return sum(stage.output_tokens for stage in self.stages)

    @property
    def cost_usd(self) -> float:
        return round(sum(stage.cost_usd for stage in self.stages), 8)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "stages": [stage.__dict__ for stage in self.stages],
        }

    @contextmanager
    def measure(
        self,
        stage: str,
        implementation: str,
        **metadata: Any,
    ) -> Iterator[StageTelemetry]:
        record = StageTelemetry(stage=stage, implementation=implementation, metadata=metadata)
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self.stages.append(record)
