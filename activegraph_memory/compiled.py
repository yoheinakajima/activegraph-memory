"""Typed, source-grounded projections compiled from claims and turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryEntityRecord:
    entity_id: str
    canonical_name: str
    kind: str = "unknown"
    aliases: tuple[str, ...] = ()
    source_claim_ids: tuple[str, ...] = ()
    source_turn_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventMentionRecord:
    mention_id: str
    claim_id: str
    text: str
    predicate: str
    entity_ids: tuple[str, ...] = ()
    category_ids: tuple[str, ...] = ()
    modality: str = "actual"
    polarity: str = "affirmative"
    event_start: str | None = None
    event_end: str | None = None
    observed_at: str | None = None
    time_confidence: float = 0.0
    source_turn_ids: tuple[str, ...] = ()
    quantity_indexes: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalEventRecord:
    event_id: str
    predicate: str
    summary: str
    entity_ids: tuple[str, ...] = ()
    category_ids: tuple[str, ...] = ()
    mention_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    source_turn_ids: tuple[str, ...] = ()
    modality: str = "actual"
    polarity: str = "affirmative"
    event_start: str | None = None
    event_end: str | None = None
    quantities: tuple[dict[str, Any], ...] = ()
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateVersionRecord:
    state_id: str
    state_key: str
    subject_ref: str
    predicate: str
    value_text: str
    entity_ids: tuple[str, ...] = ()
    quantity: dict[str, Any] | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str | None = None
    status: str = "active"
    source_claim_id: str | None = None
    source_turn_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreferenceEvidenceRecord:
    preference_id: str
    subject_ref: str
    text: str
    polarity: str
    scope_terms: tuple[str, ...] = ()
    explicit: bool = False
    observed_at: str | None = None
    source_claim_id: str | None = None
    source_turn_ids: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class ListItemRecord:
    item_id: str
    list_id: str
    position: int
    text: str
    role: str
    source_turn_id: str
    session_id: str
    observed_at: str | None = None


@dataclass(frozen=True)
class MemoryConflictRecord:
    conflict_id: str
    claim_ids: tuple[str, ...]
    state_key: str | None = None
    reason: str = "incompatible_claims"
    status: str = "unresolved"
    confidence: float = 0.0
    source_turn_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledMemoryProjection:
    entities: list[MemoryEntityRecord] = field(default_factory=list)
    event_mentions: list[EventMentionRecord] = field(default_factory=list)
    canonical_events: list[CanonicalEventRecord] = field(default_factory=list)
    state_versions: list[StateVersionRecord] = field(default_factory=list)
    preferences: list[PreferenceEvidenceRecord] = field(default_factory=list)
    list_items: list[ListItemRecord] = field(default_factory=list)
    conflicts: list[MemoryConflictRecord] = field(default_factory=list)
    current_state_by_key: dict[str, StateVersionRecord] = field(default_factory=dict)
    by_entity_id: dict[str, MemoryEntityRecord] = field(default_factory=dict)
    by_event_id: dict[str, CanonicalEventRecord] = field(default_factory=dict)
    by_conflict_id: dict[str, MemoryConflictRecord] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
