"""Typed, provider-neutral ingestion for source-grounded memory facts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .compiler import ExtractedClaimInput, SourceTurn
from .constants import AuthorityLevel, ClaimKind
from .object_types import QuantityClaim


class ExtractedEntityInput(BaseModel):
    """One canonicalizable entity mention supplied by an extractor."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = "unknown"
    aliases: list[str] = Field(default_factory=list)


class ExtractedMemoryFact(BaseModel):
    """A typed fact that remains anchored to immutable source turns."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source_turn_ids: list[str] = Field(min_length=1)
    claim_kind: ClaimKind = "unknown"
    subject_ref: str | None = None
    scope: list[str] = Field(default_factory=list)
    authority: AuthorityLevel = "unknown"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    predicate: str | None = None
    entities: list[ExtractedEntityInput] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    modality: Literal["actual", "planned", "hypothetical", "recommendation"] | None = None
    polarity: Literal["affirmative", "negative"] | None = None
    event_start: str | None = None
    event_end: str | None = None
    time_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str | None = None
    state_key: str | None = None
    preference_polarity: Literal["positive", "negative"] | None = None
    preference_scope: list[str] = Field(default_factory=list)
    quantities: list[QuantityClaim] = Field(default_factory=list)
    contradicts_claim_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryExtractionOutput(BaseModel):
    """Schema returned by an extraction model."""

    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedMemoryFact] = Field(default_factory=list)


@dataclass(frozen=True)
class MemoryExtractionResult:
    """Extraction output plus measured provider usage."""

    facts: tuple[ExtractedMemoryFact, ...]
    extractor: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryExtractor(Protocol):
    """Extraction seam for hosted, local, recorded, or deterministic providers."""

    def extract(self, turns: list[SourceTurn]) -> MemoryExtractionResult: ...


class CallableMemoryExtractor:
    """Adapt an application's existing extraction function."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def extract(self, turns: list[SourceTurn]) -> MemoryExtractionResult:
        value = self._fn(turns)
        if isinstance(value, MemoryExtractionResult):
            return value
        output = MemoryExtractionOutput.model_validate(value)
        return MemoryExtractionResult(
            facts=tuple(output.facts),
            extractor=type(self).__name__,
        )


class DeterministicMemoryExtractor:
    """Lossless, zero-provider fallback: one source-grounded fact per turn."""

    def extract(self, turns: list[SourceTurn]) -> MemoryExtractionResult:
        facts = []
        for turn in turns:
            text = " ".join(turn.content.split())
            if not text:
                continue
            facts.append(
                ExtractedMemoryFact(
                    text=text,
                    source_turn_ids=[turn.turn_id],
                    subject_ref=(
                        "user" if turn.role == "user" else "assistant" if turn.role == "assistant" else None
                    ),
                    confidence=0.6,
                    observed_at=turn.session_date,
                    metadata={"deterministic_fallback": True},
                )
            )
        return MemoryExtractionResult(
            facts=tuple(facts),
            extractor=type(self).__name__,
            metadata={"lossless_turn_fallback": True},
        )


class ActiveGraphLLMMemoryExtractor:
    """Typed fact extraction through an ActiveGraph ``LLMProvider``."""

    def __init__(
        self,
        provider,
        *,
        model: str,
        max_tokens: int = 4_000,
        max_turns_per_batch: int = 40,
        max_characters_per_batch: int = 60_000,
        temperature: float = 0.0,
        timeout_seconds: float = 90.0,
    ) -> None:
        if max_turns_per_batch < 1:
            raise ValueError("max_turns_per_batch must be at least 1")
        if max_characters_per_batch < 1:
            raise ValueError("max_characters_per_batch must be at least 1")
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns_per_batch = max_turns_per_batch
        self.max_characters_per_batch = max_characters_per_batch
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def extract(self, turns: list[SourceTurn]) -> MemoryExtractionResult:
        batches = _batch_turns(
            turns,
            max_turns=self.max_turns_per_batch,
            max_characters=self.max_characters_per_batch,
        )
        results = [self._extract_batch(batch) for batch in batches]
        if not results:
            return MemoryExtractionResult(
                facts=(),
                extractor=type(self).__name__,
                model=self.model,
                cached=True,
                metadata={"batch_count": 0, "batch_sizes": []},
            )
        return MemoryExtractionResult(
            facts=tuple(fact for result in results for fact in result.facts),
            extractor=type(self).__name__,
            model=results[-1].model or self.model,
            input_tokens=sum(result.input_tokens for result in results),
            output_tokens=sum(result.output_tokens for result in results),
            cost_usd=sum(result.cost_usd for result in results),
            latency_ms=sum(result.latency_ms for result in results),
            cached=all(result.cached for result in results),
            metadata={
                "batch_count": len(results),
                "batch_sizes": [len(batch) for batch in batches],
                "batches": [result.metadata for result in results],
            },
        )

    def _extract_batch(self, turns: list[SourceTurn]) -> MemoryExtractionResult:
        from activegraph.llm import LLMMessage

        source_ids = {turn.turn_id for turn in turns}
        payload = [
            {
                "turn_id": turn.turn_id,
                "session_id": turn.session_id,
                "observed_at": turn.session_date,
                "role": turn.role,
                "content": turn.content,
            }
            for turn in turns
        ]
        response = self.provider.complete(
            system=(
                "Extract durable, source-grounded memory facts. Atomize compound statements; "
                "preserve exact source_turn_ids; separate event time from observed time; "
                "distinguish actual, planned, hypothetical, and recommended events; preserve "
                "quantities, preference polarity, scope, and state validity. Do not infer facts "
                "that are not supported by the supplied turns."
            ),
            messages=[
                LLMMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=True, sort_keys=True),
                )
            ],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            output_schema=MemoryExtractionOutput,
            timeout_seconds=self.timeout_seconds,
            tools=None,
            structured_output_mode="prompt",
        )
        parsed = response.parsed
        output = parsed if isinstance(parsed, MemoryExtractionOutput) else MemoryExtractionOutput.model_validate(parsed)
        unknown = sorted(
            {
                source_id
                for fact in output.facts
                for source_id in fact.source_turn_ids
                if source_id not in source_ids
            }
        )
        if unknown:
            raise ValueError(f"Extractor returned unknown source_turn_ids: {unknown}")
        return MemoryExtractionResult(
            facts=tuple(output.facts),
            extractor=type(self).__name__,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=float(response.cost_usd),
            latency_ms=response.latency_seconds * 1000.0,
            cached=response.cache_hit,
            metadata=dict(response.provider_meta),
        )


def _batch_turns(
    turns: list[SourceTurn],
    *,
    max_turns: int,
    max_characters: int,
) -> list[list[SourceTurn]]:
    """Create stable extraction batches without splitting source provenance."""

    batches: list[list[SourceTurn]] = []
    current: list[SourceTurn] = []
    current_characters = 0
    for turn in turns:
        turn_characters = len(turn.content)
        would_overflow = current and (
            len(current) >= max_turns
            or current_characters + turn_characters > max_characters
        )
        if would_overflow:
            batches.append(current)
            current = []
            current_characters = 0
        current.append(turn)
        current_characters += turn_characters
    if current:
        batches.append(current)
    return batches


def extract_claim_inputs(
    turns: list[SourceTurn],
    *,
    extractor: MemoryExtractor | None = None,
) -> tuple[list[ExtractedClaimInput], MemoryExtractionResult]:
    """Extract validated claim inputs while preserving direct turn provenance."""

    result = (extractor or DeterministicMemoryExtractor()).extract(turns)
    by_turn_id = {turn.turn_id: turn for turn in turns}
    claims: list[ExtractedClaimInput] = []
    for fact in result.facts:
        source_turns = [by_turn_id[source_id] for source_id in fact.source_turn_ids if source_id in by_turn_id]
        if len(source_turns) != len(fact.source_turn_ids):
            missing = sorted(set(fact.source_turn_ids) - set(by_turn_id))
            raise ValueError(f"Extracted fact references unknown source turns: {missing}")
        anchor = min(source_turns, key=lambda turn: turn.sort_key)
        roles = {turn.role for turn in source_turns}
        role = next(iter(roles)) if len(roles) == 1 else "unknown"
        hints = {
            "claim_kind": fact.claim_kind,
            "subject_ref": fact.subject_ref,
            "scope": fact.scope,
            "authority": fact.authority,
            "predicate": fact.predicate,
            "entities": [entity.model_dump() for entity in fact.entities],
            "categories": fact.categories,
            "modality": fact.modality,
            "polarity": fact.polarity,
            "event_start": fact.event_start,
            "event_end": fact.event_end,
            "time_confidence": fact.time_confidence,
            "valid_from": fact.valid_from,
            "valid_until": fact.valid_until,
            "observed_at": fact.observed_at,
            "state_key": fact.state_key,
            "preference_polarity": fact.preference_polarity,
            "preference_scope": fact.preference_scope,
            "quantity_claims": [quantity.model_dump() for quantity in fact.quantities],
            "contradicts_claim_ids": fact.contradicts_claim_ids,
            "extraction": {
                "extractor": result.extractor,
                "model": result.model,
            },
            **fact.metadata,
        }
        claims.append(
            ExtractedClaimInput(
                text=fact.text,
                session_id=anchor.session_id,
                session_date=anchor.session_date,
                session_idx=anchor.session_idx,
                role=role,
                confidence=fact.confidence,
                source=result.extractor,
                metadata={key: value for key, value in hints.items() if value not in (None, [], {})},
                source_turn_ids=tuple(fact.source_turn_ids),
            )
        )
    return claims, result
