"""Consume the shared annotation layer (ADR 0026 steps 5-7).

Memory stops running its own extractor as the default path. Instead it
consumes ``semantic_annotation`` records produced by the shared
extraction layer (activegraph-packs' ``semantic_extraction`` pack) and
the canonical ``entity`` ids the entity pack resolves — the same
annotation contract, one extraction, every consumer projecting from it.

This module never imports activegraph-packs: the annotation contract is a
graph-object contract (facet, body, selector, extractor id, evidence
provenance), consumed here as plain dicts. Two things live here:

* ``claims_from_shared_annotations`` — turn shared assertion/
  entity_mention annotations into ``ExtractedClaimInput`` + a
  ``MemoryExtractionResult`` whose ``extractor`` is the *shared*
  extractor id, so the recorded ``memory_ingestion_stage`` attributes
  coverage to the shared run. Makes ZERO provider calls — the facts were
  already extracted upstream.

* ``CompatibilityMemoryExtractor`` — the standalone Python API's own
  extractor kept only as an explicit compatibility adapter. When shared
  extraction is active it is inert and makes zero provider calls; it is
  never silently active beside shared extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .compiler import ExtractedClaimInput, SourceTurn
from .extraction import (
    ExtractedEntityInput,
    ExtractedMemoryFact,
    MemoryExtractionResult,
    MemoryExtractor,
)

# Shared facets memory ingests. Assertions become claims; entity mentions
# contribute canonical entity refs; preferences and events enrich claims.
CLAIM_FACETS: frozenset[str] = frozenset(
    {"assertion", "preference_expression", "relation_mention", "event_mention"}
)
ENTITY_FACET = "entity_mention"

SHARED_EXTRACTION_SOURCE = "shared_extraction"


def _entity_ref_for(annotation: dict[str, Any]) -> ExtractedEntityInput | None:
    """Map one entity_mention annotation to a canonical entity ref.

    Prefers the canonical ``entity`` id the entity pack resolved
    (``metadata.canonical_entity_id`` / the mention's ``entity_id``);
    falls back to the normalized surface form. Memory consumes canonical
    ids — it never re-resolves.
    """
    body = annotation.get("body") or {}
    metadata = annotation.get("metadata") or {}
    name = (
        metadata.get("canonical_entity_id")
        or metadata.get("entity_id")
        or body.get("normalized")
        or body.get("text")
    )
    if not name:
        return None
    return ExtractedEntityInput(
        name=str(name),
        kind=str(body.get("kind") or "unknown"),
        aliases=[body["text"]] if body.get("text") and body["text"] != name else [],
    )


def _fact_from_annotation(
    annotation: dict[str, Any],
    entity_refs: list[ExtractedEntityInput],
) -> ExtractedMemoryFact | None:
    """One claim-bearing annotation → one source-grounded memory fact."""
    body = annotation.get("body") or {}
    text = body.get("text") or annotation.get("selector", {}).get("exact")
    evidence_id = annotation.get("evidence_id")
    if not text or not evidence_id:
        return None
    facet = annotation.get("facet")
    modality_map = {"stated": "actual", "hypothetical": "hypothetical"}
    modality = modality_map.get(annotation.get("modality") or "stated")
    polarity = "negative" if annotation.get("polarity") == "negative" else "affirmative"
    preference_polarity = None
    if facet == "preference_expression":
        preference_polarity = "negative" if polarity == "negative" else "positive"
    return ExtractedMemoryFact(
        text=str(text),
        source_turn_ids=[str(evidence_id)],
        claim_kind="preference" if facet == "preference_expression" else "unknown",
        confidence=float(annotation.get("confidence") or 0.6),
        entities=entity_refs,
        modality=modality,
        polarity=polarity,
        event_start=annotation.get("event_time"),
        observed_at=annotation.get("observation_time"),
        preference_polarity=preference_polarity,
        metadata={
            "shared_annotation_id": annotation.get("annotation_identity"),
            "shared_extractor_id": annotation.get("extractor_id"),
            "shared_extractor_version": annotation.get("extractor_version"),
            "facet": facet,
        },
    )


def shared_extraction_result(
    annotations: Iterable[dict[str, Any]],
) -> MemoryExtractionResult:
    """Build a MemoryExtractionResult from shared annotations — no provider.

    ``extractor`` is the shared extractor id (all annotations in one run
    share it), so the downstream ``memory_ingestion_stage`` records the
    shared extraction provenance and its source coverage.
    """
    annotations = list(annotations)
    by_evidence_entities: dict[str, list[ExtractedEntityInput]] = {}
    for annotation in annotations:
        if annotation.get("facet") != ENTITY_FACET:
            continue
        if annotation.get("status") not in (None, "active"):
            continue
        ref = _entity_ref_for(annotation)
        if ref is None:
            continue
        by_evidence_entities.setdefault(
            str(annotation.get("evidence_id")), []
        ).append(ref)

    facts: list[ExtractedMemoryFact] = []
    extractor_ids: set[str] = set()
    for annotation in annotations:
        facet = annotation.get("facet")
        if facet not in CLAIM_FACETS:
            continue
        if annotation.get("status") not in (None, "active"):
            continue
        entity_refs = by_evidence_entities.get(str(annotation.get("evidence_id")), [])
        fact = _fact_from_annotation(annotation, list(entity_refs))
        if fact is None:
            continue
        facts.append(fact)
        if annotation.get("extractor_id"):
            extractor_ids.add(str(annotation["extractor_id"]))

    # A stable single-string extractor identity for the ingestion stage.
    extractor = (
        "+".join(sorted(extractor_ids)) if extractor_ids else SHARED_EXTRACTION_SOURCE
    )
    return MemoryExtractionResult(
        facts=tuple(facts),
        extractor=extractor,
        model="",
        cached=True,
        metadata={
            "shared_extraction": True,
            "annotation_count": len(annotations),
            "shared_extractor_ids": sorted(extractor_ids),
        },
    )


@dataclass
class _AnnotatedTurn:
    turn_id: str
    session_id: str
    session_date: str
    session_idx: int
    role: str
    content: str


def claims_from_shared_annotations(
    annotations: Iterable[dict[str, Any]],
    *,
    turns_by_evidence: dict[str, SourceTurn] | None = None,
) -> tuple[list[ExtractedClaimInput], MemoryExtractionResult]:
    """Shared annotations → validated claim inputs (canonical entity refs).

    ``turns_by_evidence`` maps an annotation's ``evidence_id`` to the
    SourceTurn it anchors to (session id/date/idx). When absent, session
    fields fall back to the annotation's observation_time and evidence id
    so the claim still carries honest provenance. Makes zero provider
    calls — the facts are already extracted.
    """
    annotations = list(annotations)
    result = shared_extraction_result(annotations)
    turns_by_evidence = turns_by_evidence or {}
    claims: list[ExtractedClaimInput] = []
    for fact in result.facts:
        evidence_id = fact.source_turn_ids[0]
        turn = turns_by_evidence.get(evidence_id)
        if turn is not None:
            session_id = turn.session_id
            session_date = turn.session_date
            session_idx = turn.session_idx
            role = turn.role
        else:
            session_id = evidence_id
            session_date = fact.observed_at or ""
            session_idx = 0
            role = "unknown"
        hints = {
            "claim_kind": fact.claim_kind,
            "entities": [entity.model_dump() for entity in fact.entities],
            "modality": fact.modality,
            "polarity": fact.polarity,
            "event_start": fact.event_start,
            "observed_at": fact.observed_at,
            "preference_polarity": fact.preference_polarity,
            "role": role,
            **fact.metadata,
        }
        claims.append(
            ExtractedClaimInput(
                text=fact.text,
                session_id=session_id,
                session_date=session_date,
                session_idx=session_idx,
                role=role,
                confidence=fact.confidence,
                source=SHARED_EXTRACTION_SOURCE,
                metadata={k: v for k, v in hints.items() if v not in (None, [], {})},
                source_turn_ids=tuple(fact.source_turn_ids),
            )
        )
    return claims, result


class SharedExtractionActive(RuntimeError):
    """Raised when the compatibility adapter is asked to extract while the
    shared extraction layer is the configured source of truth."""


class CompatibilityMemoryExtractor:
    """The standalone extractor kept only as an explicit compat adapter.

    Wraps an inner MemoryExtractor (deterministic or LLM-backed). When
    shared extraction is active the adapter is INERT — it makes zero
    provider calls and, depending on ``strict``, either raises or returns
    an empty result flagged as shared-suppressed. It is never silently
    active beside shared extraction.
    """

    def __init__(
        self,
        inner: MemoryExtractor,
        *,
        shared_extraction_active: bool,
        strict: bool = False,
    ) -> None:
        self._inner = inner
        self._shared_active = shared_extraction_active
        self._strict = strict
        self.provider_calls = 0

    @property
    def shared_extraction_active(self) -> bool:
        return self._shared_active

    def extract(self, turns: list[SourceTurn]) -> MemoryExtractionResult:
        if self._shared_active:
            if self._strict:
                raise SharedExtractionActive(
                    "shared extraction is configured; the standalone "
                    "compatibility extractor must not run beside it"
                )
            # Inert: zero provider calls, honest empty result.
            return MemoryExtractionResult(
                facts=(),
                extractor=type(self).__name__,
                cached=True,
                metadata={
                    "suppressed_by_shared_extraction": True,
                    "provider_calls": 0,
                },
            )
        self.provider_calls += 1
        return self._inner.extract(turns)


__all__ = [
    "CLAIM_FACETS",
    "ENTITY_FACET",
    "SHARED_EXTRACTION_SOURCE",
    "CompatibilityMemoryExtractor",
    "SharedExtractionActive",
    "claims_from_shared_annotations",
    "shared_extraction_result",
]
