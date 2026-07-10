"""Compile canonical entities, events, state histories, preferences, and lists."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

from .compiled import (
    CanonicalEventRecord,
    CompiledMemoryProjection,
    EventMentionRecord,
    ListItemRecord,
    MemoryEntityRecord,
    PreferenceEvidenceRecord,
    StateVersionRecord,
)
from .taxonomy import (
    infer_category_ids,
    infer_polarity,
    infer_predicate,
    predicates_compatible,
    significant_tokens,
)


_CAPITALIZED_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'+.-]*(?:\s+(?:[A-Z][A-Za-z0-9&'+.-]*|of|the|and)){0,5})\b"
)
_QUOTED_ENTITY_RE = re.compile(r"['\"]([^'\"]{2,100})['\"]")
_LIST_ITEM_RE = re.compile(r"(?:^|\n)\s*(?P<position>\d{1,3})[.):]\s+(?P<text>[^\n]+)")
_ENTITY_BLOCKLIST = {
    "A",
    "An",
    "Assistant",
    "According",
    "Based",
    "I",
    "It",
    "My",
    "The",
    "Their",
    "They",
    "This",
    "User",
    "We",
}
_PLANNED_RE = re.compile(
    r"\b(plan|plans|planned|planning|schedule|scheduled|upcoming|will|would like|"
    r"want(?:s|ed)? to|thinking of|considering|hoping to|pre-?ordered)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(r"\b(if|could|might|may|hypothetical|example|fictional)\b", re.IGNORECASE)
_RECOMMENDATION_RE = re.compile(r"\b(recommend|recommended|suggest|suggested|advised)\b", re.IGNORECASE)
_HISTORICAL_UNRESOLVED_RE = re.compile(
    r"\b(previous|previously|last trip|before that|earlier|used to|once)\b",
    re.IGNORECASE,
)
_SNAPSHOT_RE = re.compile(
    r"\b(currently|current|now|as of|has|have|had|owns?|contains?|includes?|"
    r"there (?:is|are|were)|total|so far|already|need(?:s|ed)?|worn|status|level|balance)\b",
    re.IGNORECASE,
)
_EVENT_RE = re.compile(
    r"\b(attend|attended|buy|bought|purchase|purchased|spend|spent|visit|visited|"
    r"went|fly|flew|finish|finished|fix|fixed|service|serviced|receive|received|"
    r"redeem|redeemed|download|downloaded|add|added|launch|launched|sign|signed)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(prefer|favorite|like|likes|liked|love|enjoy|want|interested in|avoid|"
    r"dislike|doesn't like|don't like|works? well|struggling with|works? in|"
    r"research(?:es|ing)?|focus(?:es|ed)? on|specializ(?:es|ed|ing)|field of)\b",
    re.IGNORECASE,
)
_NEGATIVE_PREFERENCE_RE = re.compile(r"\b(avoid|dislike|doesn't like|don't like|not prefer|without)\b", re.IGNORECASE)
_STATE_STOPWORDS = {
    "active",
    "already",
    "assistant",
    "been",
    "claim",
    "current",
    "currently",
    "have",
    "latest",
    "memory",
    "more",
    "most",
    "now",
    "only",
    "previous",
    "recent",
    "recently",
    "said",
    "state",
    "than",
    "that",
    "their",
    "them",
    "they",
    "this",
    "total",
    "user",
    "with",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
}


def compile_typed_projection(index) -> CompiledMemoryProjection:
    """Build the richer projection without changing source authority."""

    entity_claims: dict[str, set[str]] = defaultdict(set)
    entity_turns: dict[str, set[str]] = defaultdict(set)
    entity_names: dict[str, str] = {}
    entity_kinds: dict[str, str] = {}
    mentions: list[EventMentionRecord] = []
    states: list[StateVersionRecord] = []
    preferences: list[PreferenceEvidenceRecord] = []

    for record in index.claims:
        names = _entity_names(record.text, record.claim.metadata)
        entity_ids: list[str] = []
        for name in names:
            entity_id = _stable_id("memory-entity", _normalize_entity(name))
            entity_ids.append(entity_id)
            entity_names.setdefault(entity_id, name)
            entity_kinds.setdefault(entity_id, _entity_kind(name, record.text))
            entity_claims[entity_id].add(record.claim_id)
            entity_turns[entity_id].update(record.source_turn_ids)

        predicate = infer_predicate(record.text)
        modality = _modality(record.text, role=str(record.claim.metadata.get("role") or ""), predicate=predicate)
        event_start, event_end, time_confidence, time_basis = _event_time(
            record,
            modality=modality,
            predicate=predicate,
        )
        mention_id = _stable_id("event-mention", record.claim_id)
        mention = EventMentionRecord(
            mention_id=mention_id,
            claim_id=record.claim_id,
            text=record.text,
            predicate=predicate,
            entity_ids=tuple(entity_ids),
            category_ids=infer_category_ids(record.text),
            modality=modality,
            polarity=infer_polarity(record.text),
            event_start=event_start,
            event_end=event_end,
            observed_at=record.claim.observed_at,
            time_confidence=time_confidence,
            source_turn_ids=tuple(record.source_turn_ids),
            quantity_indexes=tuple(range(len(record.quantity_claims))),
            metadata={"time_basis": time_basis, "role": record.claim.metadata.get("role")},
        )
        mentions.append(mention)

        if _is_state_claim(record.text, predicate=predicate, modality=modality):
            quantities = record.quantity_claims or [None]
            for quantity in quantities:
                state_key = _state_key(
                    record,
                    entity_ids=entity_ids,
                    predicate=predicate,
                    quantity=quantity,
                )
                state_id = _stable_id("memory-state", f"{state_key}|{record.claim_id}")
                states.append(
                    StateVersionRecord(
                        state_id=state_id,
                        state_key=state_key,
                        subject_ref=record.claim.subject_ref or "unknown",
                        predicate=predicate,
                        value_text=record.text,
                        entity_ids=tuple(entity_ids),
                        quantity=quantity.model_dump() if quantity is not None else None,
                        valid_from=record.claim.valid_from,
                        valid_until=record.claim.valid_until,
                        observed_at=record.claim.observed_at,
                        status=record.claim.status,
                        source_claim_id=record.claim_id,
                        source_turn_ids=tuple(record.source_turn_ids),
                        confidence=record.claim.confidence,
                        metadata={
                            "topic_key": record.topic_key,
                            "role": record.claim.metadata.get("role"),
                        },
                    )
                )

        if record.claim.metadata.get("role") == "user" and (
            record.claim.claim_kind in {"preference", "instruction"} or _PREFERENCE_RE.search(record.text)
        ):
            preferences.append(
                PreferenceEvidenceRecord(
                    preference_id=_stable_id("memory-preference", record.claim_id),
                    subject_ref=record.claim.subject_ref or "user",
                    text=record.text,
                    polarity="negative" if _NEGATIVE_PREFERENCE_RE.search(record.text) else "positive",
                    scope_terms=tuple(_scope_terms(record.text)),
                    explicit=record.claim.claim_kind in {"preference", "instruction"},
                    observed_at=record.claim.observed_at,
                    source_claim_id=record.claim_id,
                    source_turn_ids=tuple(record.source_turn_ids),
                    confidence=record.claim.confidence,
                )
            )

    entities = [
        MemoryEntityRecord(
            entity_id=entity_id,
            canonical_name=entity_names[entity_id],
            kind=entity_kinds[entity_id],
            aliases=(entity_names[entity_id],),
            source_claim_ids=tuple(sorted(entity_claims[entity_id])),
            source_turn_ids=tuple(sorted(entity_turns[entity_id])),
        )
        for entity_id in sorted(entity_names, key=lambda eid: entity_names[eid].lower())
    ]
    events = _canonicalize_events(index, mentions)
    state_versions, current_state = _compile_state_histories(states)
    list_items = _compile_list_items(index.turns)
    return CompiledMemoryProjection(
        entities=entities,
        event_mentions=mentions,
        canonical_events=events,
        state_versions=state_versions,
        preferences=preferences,
        list_items=list_items,
        current_state_by_key=current_state,
        by_entity_id={entity.entity_id: entity for entity in entities},
        by_event_id={event.event_id: event for event in events},
        metadata={
            "compiler": "typed_projection_v1",
            "n_entities": len(entities),
            "n_event_mentions": len(mentions),
            "n_canonical_events": len(events),
            "n_state_versions": len(state_versions),
            "n_preferences": len(preferences),
            "n_list_items": len(list_items),
        },
    )


def _entity_names(text: str, metadata: dict) -> list[str]:
    supplied = metadata.get("entities") or metadata.get("entity_mentions") or []
    out: list[str] = []
    for item in supplied:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
    out.extend(match.group(1).strip() for match in _QUOTED_ENTITY_RE.finditer(text))
    for match in _CAPITALIZED_ENTITY_RE.finditer(text):
        value = match.group(0).strip(" .,'\"")
        if value in _ENTITY_BLOCKLIST or value.lower() == "s" or len(value) < 2:
            continue
        if value.lower().startswith(("the user", "the assistant")):
            continue
        out.append(value)
    return _dedupe_names(out)[:12]


def _entity_kind(name: str, text: str) -> str:
    lower = f"{name} {text}".lower()
    if re.search(r"\b(team|group|company|organization|airlines?)\b", lower):
        return "organization"
    if re.search(r"\b(trip|city|hotel|museum|park|beach|church|cathedral)\b", lower):
        return "place_or_trip"
    if re.search(r"\b(phone|laptop|camera|lens|bike|car|device|album|book|painting)\b", lower):
        return "object"
    if len(name.split()) <= 3 and not re.search(r"\d", name):
        return "person_or_named_entity"
    return "unknown"


def _modality(text: str, *, role: str, predicate: str) -> str:
    if _HYPOTHETICAL_RE.search(text):
        return "hypothetical"
    if role == "assistant" and (predicate == "recommend" or _RECOMMENDATION_RE.search(text)):
        return "recommendation"
    if _PLANNED_RE.search(text):
        return "planned"
    return "actual"


def _event_time(record, *, modality: str, predicate: str) -> tuple[str | None, str | None, float, str]:
    resolved = [ref for ref in record.temporal_refs if ref.resolved_start or ref.resolved_end]
    if resolved:
        best = max(resolved, key=lambda ref: (_temporal_ref_score(record.text, ref.text), ref.confidence))
        return (
            best.resolved_start or best.resolved_end,
            best.resolved_end or best.resolved_start,
            best.confidence,
            f"temporal_ref:{best.text}",
        )
    if _HISTORICAL_UNRESOLVED_RE.search(record.text):
        return None, None, 0.2, "historical_reference_unresolved"
    if modality in {"planned", "hypothetical", "recommendation"}:
        return None, None, 0.25, f"{modality}_without_resolved_time"
    if predicate != "state" or _EVENT_RE.search(record.text) or _SNAPSHOT_RE.search(record.text):
        value = record.claim.observed_at
        return value, value, 0.55, "observed_time_proxy"
    return None, None, 0.0, "unresolved"


def _temporal_ref_score(text: str, ref_text: str) -> float:
    lower = text.lower()
    pos = lower.find(ref_text.lower())
    if pos < 0:
        return 0.0
    left = lower[max(0, pos - 56) : pos]
    score = 0.5
    if re.search(r"\b(arrived|purchased|bought|attended|finished|completed|received|launched|signed)\b", left):
        score += 0.7
    if re.search(r"\b(expected|planned|scheduled|before|previous)\b", left):
        score -= 0.55
    return score


def _is_state_claim(text: str, *, predicate: str, modality: str) -> bool:
    if modality in {"hypothetical", "recommendation"}:
        return False
    if predicate in {"attend", "visit", "purchase", "acquire", "repair", "donate", "birth"}:
        return False
    return bool(predicate == "state" or _SNAPSHOT_RE.search(text))


def _state_key(record, *, entity_ids: list[str], predicate: str, quantity) -> str:
    units = ""
    if quantity is not None:
        units = str(quantity.unit or quantity.property_name or "value").lower()
    topic_tokens = [
        token
        for token in record.topic_key.split()
        if token not in _STATE_STOPWORDS and not token.isdigit()
    ]
    topic = " ".join(topic_tokens[:6])
    entity = entity_ids[0] if entity_ids else ""
    return "|".join((record.claim.subject_ref or "unknown", predicate, entity, units, topic))


def _canonicalize_events(index, mentions: list[EventMentionRecord]) -> list[CanonicalEventRecord]:
    groups: dict[str, list[EventMentionRecord]] = defaultdict(list)
    group_tokens: dict[str, set[str]] = {}
    for mention in mentions:
        if mention.modality != "actual" or mention.polarity != "affirmative":
            continue
        record = index.by_claim_id[mention.claim_id]
        object_tokens = sorted(
            token
            for token in significant_tokens(record.text)
            if token not in _STATE_STOPWORDS and not token.isdigit()
        )
        object_set = set(object_tokens)
        key = ""
        for candidate_key, tokens in group_tokens.items():
            group_quantities = set().union(
                *(
                    _quantity_signatures(index.by_claim_id[prior.claim_id].quantity_claims)
                    for prior in groups[candidate_key]
                )
            )
            mention_quantities = _quantity_signatures(record.quantity_claims)
            if mention_quantities and group_quantities and not mention_quantities & group_quantities:
                continue
            if any(
                _mentions_can_merge(index, mention, prior, object_set, tokens)
                for prior in groups[candidate_key]
            ):
                key = candidate_key
                break
        if not key:
            key = "|".join(
                (
                    mention.predicate,
                    mention.entity_ids[0] if mention.entity_ids else "",
                    mention.event_start or "unknown-time",
                    mention.mention_id,
                )
            )
            group_tokens[key] = object_set
        else:
            group_tokens[key].update(object_set)
        groups[key].append(mention)

    events: list[CanonicalEventRecord] = []
    for key, grouped in groups.items():
        first = max(
            grouped,
            key=lambda mention: (
                mention.time_confidence,
                int(bool(index.by_claim_id[mention.claim_id].quantity_claims)),
            ),
        )
        claims = [index.by_claim_id[mention.claim_id] for mention in grouped]
        quantities = []
        seen_quantities: set[tuple] = set()
        for claim in claims:
            for quantity in claim.quantity_claims:
                signature = (quantity.value, quantity.unit, quantity.property_name)
                if signature in seen_quantities:
                    continue
                seen_quantities.add(signature)
                quantities.append(quantity.model_dump())
        event_id = _stable_id("canonical-event", key)
        events.append(
            CanonicalEventRecord(
                event_id=event_id,
                predicate=first.predicate,
                summary=first.text,
                entity_ids=tuple(_dedupe(item for mention in grouped for item in mention.entity_ids)),
                category_ids=tuple(_dedupe(item for mention in grouped for item in mention.category_ids)),
                mention_ids=tuple(mention.mention_id for mention in grouped),
                claim_ids=tuple(mention.claim_id for mention in grouped),
                source_turn_ids=tuple(_dedupe(item for mention in grouped for item in mention.source_turn_ids)),
                modality=first.modality,
                polarity=first.polarity,
                event_start=first.event_start,
                event_end=first.event_end,
                quantities=tuple(quantities),
                confidence=min(1.0, max(mention.time_confidence for mention in grouped) + min(0.2, 0.05 * (len(grouped) - 1))),
                metadata={
                    "canonical_key": key,
                    "supporting_mentions": len(grouped),
                    "time_bases": _dedupe(
                        str(mention.metadata.get("time_basis") or "")
                        for mention in grouped
                    ),
                    "roles": _dedupe(
                        str(mention.metadata.get("role") or "")
                        for mention in grouped
                    ),
                    "observed_at": max(
                        (mention.observed_at for mention in grouped if mention.observed_at),
                        default=None,
                    ),
                },
            )
        )
    return sorted(events, key=lambda event: (event.event_start or "", event.event_id))


def _mentions_can_merge(
    index,
    left: EventMentionRecord,
    right: EventMentionRecord,
    left_tokens: set[str],
    group_tokens: set[str],
) -> bool:
    if not (
        predicates_compatible(left.predicate, right.predicate)
        and predicates_compatible(right.predicate, left.predicate)
    ):
        return False
    left_claim = index.by_claim_id[left.claim_id]
    right_claim = index.by_claim_id[right.claim_id]
    left_quantities = _quantity_signatures(left_claim.quantity_claims)
    right_quantities = _quantity_signatures(right_claim.quantity_claims)
    shared_entity = bool(set(left.entity_ids) & set(right.entity_ids))
    if left_quantities and right_quantities and not left_quantities & right_quantities:
        return False
    left_proxy = left.metadata.get("time_basis") == "observed_time_proxy"
    right_proxy = right.metadata.get("time_basis") == "observed_time_proxy"
    if left.event_start and right.event_start and left.event_start != right.event_start:
        if not left_proxy and not right_proxy:
            return False
        if not shared_entity and not (left_quantities & right_quantities):
            return False
    if re.search(r"\b(?:another|additional|again|second|third)\b", left.text, re.IGNORECASE):
        if not set(left.source_turn_ids) & set(right.source_turn_ids):
            return False
    shared_turn = bool(set(left.source_turn_ids) & set(right.source_turn_ids))
    shared_categories = set(left.category_ids) & set(right.category_ids)
    meaningful_categories = shared_categories - {"expense", "event", "vehicle"}
    similarity = _jaccard(left_tokens, group_tokens)
    if shared_turn and similarity >= 0.2 and (shared_entity or meaningful_categories):
        return True
    if shared_entity and similarity >= 0.24:
        return True
    if left_quantities & right_quantities and shared_categories and similarity >= 0.32:
        return True
    if meaningful_categories and similarity >= 0.58:
        return True
    return similarity >= 0.72


def _quantity_signatures(quantities) -> set[tuple[float, str]]:
    return {
        (float(quantity.value), str(quantity.unit or quantity.property_name or ""))
        for quantity in quantities
        if quantity.value is not None
    }


def _compile_state_histories(states: list[StateVersionRecord]) -> tuple[list[StateVersionRecord], dict[str, StateVersionRecord]]:
    grouped: dict[str, list[StateVersionRecord]] = defaultdict(list)
    for state in states:
        grouped[state.state_key].append(state)
    out: list[StateVersionRecord] = []
    current: dict[str, StateVersionRecord] = {}
    for key, versions in grouped.items():
        versions.sort(key=lambda state: (state.observed_at or "", state.state_id))
        for index, state in enumerate(versions):
            if index < len(versions) - 1:
                next_state = versions[index + 1]
                state = StateVersionRecord(
                    **{
                        **state.__dict__,
                        "status": "superseded",
                        "valid_until": next_state.valid_from or next_state.observed_at,
                        "metadata": {**state.metadata, "superseded_by": next_state.state_id},
                    }
                )
            out.append(state)
        current[key] = out[-1]
    out.sort(key=lambda state: (state.observed_at or "", state.state_key, state.state_id))
    return out, current


def _compile_list_items(turns: Iterable) -> list[ListItemRecord]:
    out: list[ListItemRecord] = []
    for turn in turns:
        matches = list(_LIST_ITEM_RE.finditer(turn.content))
        if not matches:
            continue
        list_id = _stable_id("memory-list", turn.turn_id)
        for match in matches:
            position = int(match.group("position"))
            text = match.group("text").strip()
            out.append(
                ListItemRecord(
                    item_id=_stable_id("memory-list-item", f"{list_id}|{position}|{text}"),
                    list_id=list_id,
                    position=position,
                    text=text,
                    role=turn.role,
                    source_turn_id=turn.turn_id,
                    session_id=turn.session_id,
                    observed_at=turn.session_date,
                )
            )
    return out


def _scope_terms(text: str) -> list[str]:
    return sorted(
        token
        for token in significant_tokens(text)
        if token not in _STATE_STOPWORDS and len(token) >= 3
    )[:16]


def _normalize_entity(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _dedupe_names(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = _normalize_entity(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(value)
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _session_id(turn_id: str) -> str:
    return turn_id.rsplit("#", 1)[0]
