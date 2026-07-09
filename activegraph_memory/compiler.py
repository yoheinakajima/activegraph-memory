"""Compile source turns and extracted claims into a memory index.

This module is intentionally runtime-agnostic: callers can provide claims from
an LLM extractor, a deterministic parser, a connector, or a benchmark cache.
The compiler turns those inputs into source-grounded ``MemoryClaim`` records
with temporal, quantity, provenance, and lightweight supersession metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .constants import AuthorityLevel, ClaimKind
from .object_types import MemoryClaim, QuantityClaim, TemporalRef
from .taxonomy import category_label, infer_category_ids, infer_polarity, infer_predicate
from .temporal import extract_temporal_refs


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_QUANTITY_RE = re.compile(
    r"(?P<prefix>\$)?(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>[kKmM])?"
    r"\s*(?P<unit>%|percent|dollars?|usd|"
    r"pages?|days?|weeks?|months?|years?|hours?|minutes?|ounces?|oz|cups?|"
    r"plants?|stories?|babies?|weddings?|museums?|doctors?)?",
    re.IGNORECASE,
)
_WORD_QUANTITY_RE = re.compile(
    r"\b(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty)\s+(?P<unit>pages?|days?|weeks?|months?|years?|hours?|minutes?|"
    r"ounces?|oz|cups?|plants?|stories?|babies?|weddings?|museums?|doctors?)\b",
    re.IGNORECASE,
)
_WORD_NUMBER_RE = re.compile(
    r"\b(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty)\b",
    re.IGNORECASE,
)
_FOLLOWING_UNIT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z&'.-]*")
_GENERIC_UNIT_STOPWORDS = {
    "a",
    "about",
    "after",
    "ago",
    "all",
    "already",
    "and",
    "around",
    "as",
    "at",
    "before",
    "but",
    "by",
    "currently",
    "during",
    "each",
    "earlier",
    "far",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "just",
    "last",
    "now",
    "of",
    "on",
    "or",
    "over",
    "so",
    "that",
    "the",
    "this",
    "through",
    "to",
    "today",
    "total",
    "up",
    "was",
    "were",
    "when",
    "which",
    "who",
    "with",
    "yet",
}
_GENERIC_UNIT_LEADING_WORDS = {
    "a",
    "an",
    "another",
    "big",
    "current",
    "different",
    "emma's",
    "extra",
    "first",
    "h",
    "korean",
    "last",
    "m",
    "my",
    "national",
    "new",
    "old",
    "our",
    "recent",
    "same",
    "small",
    "the",
    "their",
    "these",
    "those",
}
_GENERIC_UNIT_BLOCKLIST = {
    "am",
    "pm",
    "st",
    "nd",
    "rd",
    "th",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "assistant",
    "at",
    "be",
    "because",
    "by",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "user",
    "was",
    "were",
    "with",
    "you",
    "your",
}
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


@dataclass(frozen=True)
class SourceTurn:
    """A single immutable source turn in the event log."""

    turn_id: str
    session_id: str
    session_date: str
    session_idx: int
    turn_idx: int
    role: str
    content: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[str, int, int]:
        return (self.session_date, self.session_idx, self.turn_idx)


@dataclass(frozen=True)
class ExtractedClaimInput:
    """Claim-like input produced by an extractor or deterministic fallback."""

    text: str
    session_id: str
    session_date: str
    session_idx: int
    role: str = "unknown"
    mentioned_turn_idxs: tuple[int, ...] = ()
    confidence: float = 0.82
    source: str = "external_extract"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryClaimRecord:
    """Compiled claim plus resolved provenance and derived annotations."""

    claim_id: str
    claim: MemoryClaim
    source_turn_ids: list[str]
    temporal_refs: list[TemporalRef] = field(default_factory=list)
    quantity_claims: list[QuantityClaim] = field(default_factory=list)
    sort_key: tuple[str, int, int] = ("", 0, 0)
    topic_key: str = ""
    superseded_by: str | None = None
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.claim.text


@dataclass(frozen=True)
class EntityRef:
    """A normalized entity-like reference compiled from memory claims."""

    entity_id: str
    label: str
    kind: str = "unknown"
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CategoryRef:
    """A coarse deterministic category used for graph query filtering."""

    category_id: str
    label: str
    confidence: float = 0.7


@dataclass(frozen=True)
class MemoryEventRecord:
    """A compact event row compiled from claims for reducers and timelines."""

    event_id: str
    text: str
    predicate: str
    entity_refs: tuple[str, ...] = ()
    category_ids: tuple[str, ...] = ()
    quantity_claims: tuple[QuantityClaim, ...] = ()
    temporal_refs: tuple[TemporalRef, ...] = ()
    event_start: str | None = None
    event_end: str | None = None
    observed_at: str | None = None
    source_claim_id: str | None = None
    source_turn_ids: tuple[str, ...] = ()
    sort_key: tuple[str, int, int] = ("", 0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryIndex:
    """Compiled memory graph projection used by the retriever."""

    turns: list[SourceTurn]
    claims: list[MemoryClaimRecord]
    by_turn_id: dict[str, SourceTurn]
    by_claim_id: dict[str, MemoryClaimRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[EntityRef] = field(default_factory=list)
    categories: list[CategoryRef] = field(default_factory=list)
    events: list[MemoryEventRecord] = field(default_factory=list)
    by_entity_id: dict[str, EntityRef] = field(default_factory=dict)
    by_event_id: dict[str, MemoryEventRecord] = field(default_factory=dict)

    @property
    def session_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for turn in self.turns:
            if turn.session_id not in seen:
                seen.add(turn.session_id)
                out.append(turn.session_id)
        return out


def stable_claim_id(session_id: str, role: str, text: str) -> str:
    """Stable content id for a claim produced from a source session."""

    digest = hashlib.sha256(f"{session_id}|{role}|{text}".encode("utf-8")).hexdigest()
    return f"claim:{digest[:16]}"


def compile_memory_index(
    *,
    turns: Iterable[SourceTurn],
    claims: Iterable[ExtractedClaimInput],
    metadata: dict[str, Any] | None = None,
) -> MemoryIndex:
    """Build a source-grounded memory index from turns and claim inputs."""

    turn_list = sorted(list(turns), key=lambda turn: turn.sort_key)
    by_turn_id = {turn.turn_id: turn for turn in turn_list}
    turns_by_session: dict[str, list[SourceTurn]] = {}
    for turn in turn_list:
        turns_by_session.setdefault(turn.session_id, []).append(turn)

    records: list[MemoryClaimRecord] = []
    for raw in claims:
        text = " ".join((raw.text or "").split())
        if not text:
            continue
        session_turns = turns_by_session.get(raw.session_id, [])
        source_turns: list[SourceTurn] = []
        seen_turns: set[str] = set()
        for idx in raw.mentioned_turn_idxs:
            if idx < 0 or idx >= len(session_turns):
                continue
            turn = session_turns[idx]
            if turn.turn_id in seen_turns:
                continue
            seen_turns.add(turn.turn_id)
            source_turns.append(turn)
        if not source_turns and session_turns:
            # Keep provenance rather than orphaning the claim. A fact with
            # empty anchors is still tied to the session that produced it.
            source_turns = [session_turns[0]]

        source_turn_ids = [turn.turn_id for turn in source_turns]
        claim_id = stable_claim_id(raw.session_id, raw.role, text)
        kind = infer_claim_kind(text, role=raw.role)
        temporal_refs = extract_temporal_refs(
            text,
            anchor_time=_date_prefix(raw.session_date),
        )
        quantities = extract_quantity_claims(text)
        sort_key = (
            raw.session_date,
            raw.session_idx,
            min((turn.turn_idx for turn in source_turns), default=0),
        )
        claim = MemoryClaim(
            text=text,
            claim_kind=kind,
            subject_ref=_subject_for_role(raw.role),
            scope=infer_scope(text),
            status="active",
            confidence=max(0.0, min(1.0, raw.confidence)),
            authority=infer_authority(text, role=raw.role),
            source_ids=source_turn_ids,
            valid_from=_date_prefix(raw.session_date),
            observed_at=_date_prefix(raw.session_date),
            metadata={
                "claim_id": claim_id,
                "role": raw.role,
                "session_id": raw.session_id,
                "session_date": raw.session_date,
                "session_idx": raw.session_idx,
                "source": raw.source,
                "mentioned_turn_idxs": list(raw.mentioned_turn_idxs),
                "temporal_refs": [ref.model_dump() for ref in temporal_refs],
                "quantity_claims": [q.model_dump() for q in quantities],
                **raw.metadata,
            },
        )
        records.append(
            MemoryClaimRecord(
                claim_id=claim_id,
                claim=claim,
                source_turn_ids=source_turn_ids,
                temporal_refs=temporal_refs,
                quantity_claims=quantities,
                sort_key=sort_key,
                topic_key=topic_key(text),
            )
        )

    _mark_supersession(records)
    by_claim_id = {record.claim_id: record for record in records}
    entities, categories, events = _compile_event_projection(records)
    return MemoryIndex(
        turns=turn_list,
        claims=records,
        by_turn_id=by_turn_id,
        by_claim_id=by_claim_id,
        metadata=metadata or {},
        entities=entities,
        categories=categories,
        events=events,
        by_entity_id={entity.entity_id: entity for entity in entities},
        by_event_id={event.event_id: event for event in events},
    )


def _compile_event_projection(
    records: Iterable[MemoryClaimRecord],
) -> tuple[list[EntityRef], list[CategoryRef], list[MemoryEventRecord]]:
    """Compile claims into coarse entity/category/event rows."""

    entities_by_id: dict[str, EntityRef] = {}
    category_ids: set[str] = set()
    events: list[MemoryEventRecord] = []
    seen_events: set[str] = set()

    for record in records:
        label = _entity_label(record)
        entity_id = stable_entity_id(label)
        if entity_id not in entities_by_id:
            entities_by_id[entity_id] = EntityRef(
                entity_id=entity_id,
                label=label,
                kind=_entity_kind(record.text),
                aliases=tuple(_entity_aliases(label, record.topic_key)),
                metadata={"source_claim_ids": [record.claim_id]},
            )
        else:
            entity = entities_by_id[entity_id]
            source_claim_ids = list(entity.metadata.get("source_claim_ids", []))
            if record.claim_id not in source_claim_ids:
                source_claim_ids.append(record.claim_id)
            entities_by_id[entity_id] = EntityRef(
                entity_id=entity.entity_id,
                label=entity.label,
                kind=entity.kind,
                aliases=entity.aliases,
                metadata={**entity.metadata, "source_claim_ids": source_claim_ids},
            )

        categories = infer_category_ids(record.text)
        category_ids.update(categories)
        event_start, event_end = _event_dates(record)
        predicate = infer_predicate(record.text)
        dedupe_key = _event_dedupe_key(
            entity_id=entity_id,
            predicate=predicate,
            event_start=event_start,
            text=record.text,
        )
        if dedupe_key in seen_events:
            continue
        seen_events.add(dedupe_key)
        event_id = stable_event_id(dedupe_key)
        events.append(
            MemoryEventRecord(
                event_id=event_id,
                text=record.text,
                predicate=predicate,
                entity_refs=(entity_id,),
                category_ids=categories,
                quantity_claims=tuple(record.quantity_claims),
                temporal_refs=tuple(record.temporal_refs),
                event_start=event_start,
                event_end=event_end,
                observed_at=record.claim.observed_at,
                source_claim_id=record.claim_id,
                source_turn_ids=tuple(record.source_turn_ids),
                sort_key=record.sort_key,
                metadata={
                    "dedupe_key": dedupe_key,
                    "topic_key": record.topic_key,
                    "claim_status": record.claim.status,
                    "polarity": infer_polarity(record.text),
                    "role": record.claim.metadata.get("role"),
                },
            )
        )

    categories = [
        CategoryRef(category_id=category_id, label=category_label(category_id))
        for category_id in sorted(category_ids)
    ]
    entities = sorted(entities_by_id.values(), key=lambda entity: entity.label)
    events.sort(key=lambda event: event.sort_key)
    return entities, categories, events


def stable_entity_id(label: str) -> str:
    """Stable id for a normalized entity label."""

    digest = hashlib.sha256(label.lower().encode("utf-8")).hexdigest()
    return f"entity:{digest[:16]}"


def stable_event_id(dedupe_key: str) -> str:
    """Stable id for a compiled memory event."""

    digest = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()
    return f"event:{digest[:16]}"


def _entity_label(record: MemoryClaimRecord) -> str:
    topic = " ".join(record.topic_key.split()[:6])
    if topic:
        return topic
    cleaned = re.sub(r"\s+", " ", record.text.strip())
    return cleaned[:80] or record.claim_id


def _entity_kind(text: str) -> str:
    categories = set(infer_category_ids(text))
    if "plant" in categories:
        return "plant"
    if "project" in categories:
        return "project"
    if "travel" in categories:
        return "place_or_trip"
    if "event" in categories:
        return "event"
    if "expense" in categories:
        return "expense"
    return "unknown"


def _entity_aliases(label: str, topic: str) -> list[str]:
    aliases = [label]
    if topic and topic != label:
        aliases.append(topic)
    return aliases


def _event_dates(record: MemoryClaimRecord) -> tuple[str | None, str | None]:
    for ref in record.temporal_refs:
        if ref.resolved_start or ref.resolved_end:
            return (
                ref.resolved_start or ref.resolved_end,
                ref.resolved_end or ref.resolved_start,
            )
    return record.claim.valid_from, record.claim.valid_until or record.claim.valid_from


def _event_dedupe_key(
    *,
    entity_id: str,
    predicate: str,
    event_start: str | None,
    text: str,
) -> str:
    normalized_text = " ".join(_TOKEN_RE.findall(text.lower()))[:160]
    return "|".join((entity_id, predicate, event_start or "", normalized_text))


def infer_claim_kind(text: str, *, role: str = "unknown") -> ClaimKind:
    lower = text.lower()
    if any(word in lower for word in ("prefer", "favorite", "likes", "style", "tone")):
        return "preference"
    if any(word in lower for word in ("going forward", "always", "never", "should", "don't")):
        return "instruction"
    if any(word in lower for word in ("decided", "agreed", "approved", "final", "signed")):
        return "decision"
    if any(word in lower for word in ("step", "procedure", "process", "recipe")):
        return "procedure"
    if role == "assistant" and any(
        word in lower for word in ("recommended", "suggested", "computed", "told")
    ):
        return "fact"
    return "fact"


def infer_authority(text: str, *, role: str = "unknown") -> AuthorityLevel:
    lower = text.lower()
    if any(word in lower for word in ("signed", "approved", "final", "decided")):
        return "high"
    if role in {"user", "assistant"}:
        return "medium"
    return "unknown"


def infer_scope(text: str) -> list[str]:
    lower = text.lower()
    scopes: list[str] = []
    if any(word in lower for word in ("article", "writing", "tone", "style")):
        scopes.append("writing")
    if any(word in lower for word in ("schedule", "calendar", "appointment")):
        scopes.append("schedule")
    if any(word in lower for word in ("price", "budget", "cost", "$")):
        scopes.append("finance")
    if any(word in lower for word in ("trip", "travel", "flight", "hotel")):
        scopes.append("travel")
    if not scopes:
        scopes.append("general")
    return scopes


def extract_quantity_claims(text: str) -> list[QuantityClaim]:
    """Extract simple quantity mentions as structured annotations."""

    out: list[QuantityClaim] = []
    seen: set[tuple[float, str | None]] = set()
    for match in _QUANTITY_RE.finditer(text):
        raw_unit = match.group("unit")
        prefix = match.group("prefix")
        suffix = (match.group("suffix") or "").lower()
        unit = "usd" if prefix == "$" else (raw_unit.lower() if raw_unit else None)
        value = float(match.group("value").replace(",", ""))
        if suffix == "k":
            value *= 1000
            if prefix == "$":
                unit = "usd"
        elif suffix == "m":
            value *= 1000000
            if prefix == "$":
                unit = "usd"
        if unit is None and suffix == "":
            unit = _infer_following_count_unit(text, match.end())
        key = (value, unit)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            QuantityClaim(
                property_name="quantity",
                value=value,
                unit=unit,
                exactness="exact",
                source_text=match.group(0).strip() if raw_unit or prefix or suffix else f"{match.group(0).strip()} {unit}".strip(),
                confidence=0.8,
            )
        )
    for match in _WORD_QUANTITY_RE.finditer(text):
        raw_value = match.group("value").lower()
        raw_unit = match.group("unit")
        value = float(_WORD_NUMBERS[raw_value])
        unit = raw_unit.lower() if raw_unit else None
        key = (value, unit)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            QuantityClaim(
                property_name="quantity",
                value=value,
                unit=unit,
                exactness="exact",
                source_text=match.group(0),
                confidence=0.78,
            )
        )
    for match in _WORD_NUMBER_RE.finditer(text):
        raw_value = match.group("value").lower()
        value = float(_WORD_NUMBERS[raw_value])
        unit = _infer_following_count_unit(text, match.end())
        if not unit:
            continue
        key = (value, unit)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            QuantityClaim(
                property_name="quantity",
                value=value,
                unit=unit,
                exactness="exact",
                source_text=f"{match.group(0)} {unit}",
                confidence=0.72,
            )
        )
    return out


def _infer_following_count_unit(text: str, start: int) -> str | None:
    """Infer the counted noun after a number without relying on a closed unit list."""

    tail = text[start : start + 96]
    if not tail or re.match(r"\s*[/:-]", tail):
        return None
    if re.match(r"\s*(?:am|pm)\b", tail, re.IGNORECASE):
        return None

    tokens: list[str] = []
    for match in _FOLLOWING_UNIT_TOKEN_RE.finditer(tail):
        raw = match.group(0).strip(" .'\"").lower()
        if not raw:
            continue
        if match.start() > 0 and re.search(r"[.;!?]", tail[: match.start()]):
            break
        if raw in _GENERIC_UNIT_STOPWORDS:
            if tokens:
                break
            continue
        if raw.endswith("'s"):
            continue
        tokens.append(raw)
        if len(tokens) >= 5:
            break

    while tokens and tokens[0] in _GENERIC_UNIT_LEADING_WORDS:
        tokens.pop(0)
    if not tokens:
        return None

    unit = tokens[-1]
    if unit in _GENERIC_UNIT_BLOCKLIST or len(unit) < 2:
        return None
    return unit


def topic_key(text: str) -> str:
    """Coarse deterministic topic key for support/supersession grouping."""

    tokens = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token in _STOPWORDS or len(token) < 3 or token.isdigit():
            continue
        tokens.append(token)
    # Preserve order while dropping repeated values.
    seen: set[str] = set()
    kept: list[str] = []
    for token in tokens:
        normalized = _normalize_topic_token(token)
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    return " ".join(kept[:10])


def claim_tokens(text: str) -> set[str]:
    """Significant normalized tokens used by retrieval and grouping."""

    return {token for token in topic_key(text).split() if token}


def _mark_supersession(records: list[MemoryClaimRecord]) -> None:
    """Mark obvious later claims on the same topic as superseding older ones."""

    groups: dict[tuple[str | None, str], list[MemoryClaimRecord]] = {}
    for record in records:
        if len(record.topic_key.split()) < 3:
            continue
        key = (record.claim.subject_ref, record.topic_key)
        groups.setdefault(key, []).append(record)

    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda record: record.sort_key)
        for older, newer in zip(group, group[1:]):
            # Only mark as superseded when the same topic has different
            # extracted text. Exact duplicates are treated as support.
            if older.text == newer.text:
                older.supports.append(newer.claim_id)
                continue
            if _distinct_event_pair(older, newer):
                continue
            older.superseded_by = newer.claim_id
            older.claim = older.claim.model_copy(
                update={
                    "status": "superseded",
                    "valid_until": newer.claim.valid_from,
                    "metadata": {
                        **older.claim.metadata,
                        "superseded_by": newer.claim_id,
                    },
                }
            )


def _distinct_event_pair(older: MemoryClaimRecord, newer: MemoryClaimRecord) -> bool:
    """Return True when similar claims should remain separate event rows."""

    if older.sort_key[0] == newer.sort_key[0]:
        return False
    older_predicate = infer_predicate(older.text)
    newer_predicate = infer_predicate(newer.text)
    return older_predicate != "state" and newer_predicate != "state"


def _subject_for_role(role: str) -> str:
    if role == "assistant":
        return "assistant"
    if role == "user":
        return "user"
    return "unknown"


def _date_prefix(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})", value)
    if not match:
        return None
    return (
        f"{int(match.group('year')):04d}-"
        f"{int(match.group('month')):02d}-"
        f"{int(match.group('day')):02d}"
    )


def _normalize_topic_token(token: str) -> str:
    endings = ("ing", "ed", "es", "s")
    for ending in endings:
        if len(token) > len(ending) + 3 and token.endswith(ending):
            return token[: -len(ending)]
    return token
