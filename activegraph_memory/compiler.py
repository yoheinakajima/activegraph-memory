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
from .temporal import extract_temporal_refs


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_QUANTITY_RE = re.compile(
    r"(?P<prefix>\$)?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|dollars?|usd|"
    r"pages?|days?|weeks?|months?|years?|hours?|minutes?|ounces?|oz|cups?|"
    r"plants?|stories?|babies?|weddings?|museums?|doctors?)?",
    re.IGNORECASE,
)

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


@dataclass
class MemoryIndex:
    """Compiled memory graph projection used by the retriever."""

    turns: list[SourceTurn]
    claims: list[MemoryClaimRecord]
    by_turn_id: dict[str, SourceTurn]
    by_claim_id: dict[str, MemoryClaimRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

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
    return MemoryIndex(
        turns=turn_list,
        claims=records,
        by_turn_id=by_turn_id,
        by_claim_id=by_claim_id,
        metadata=metadata or {},
    )


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
        unit = "usd" if prefix == "$" else (raw_unit.lower() if raw_unit else None)
        value = float(match.group("value"))
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
                confidence=0.8,
            )
        )
    return out


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


def _subject_for_role(role: str) -> str:
    if role == "assistant":
        return "assistant"
    if role == "user":
        return "user"
    return "unknown"


def _date_prefix(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", value)
    if not match:
        return None
    return match.group(0).replace("/", "-")


def _normalize_topic_token(token: str) -> str:
    endings = ("ing", "ed", "es", "s")
    for ending in endings:
        if len(token) > len(ending) + 3 and token.endswith(ending):
            return token[: -len(ending)]
    return token

