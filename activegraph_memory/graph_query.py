"""Deterministic graph-query reducers over a compiled memory index."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from .compiler import MemoryEventRecord, MemoryIndex
from .object_types import QuantityClaim
from .taxonomy import (
    category_label,
    infer_category_ids,
    infer_predicate,
    normalize_token,
    predicates_compatible,
    significant_tokens,
)
from .temporal import extract_temporal_refs


_MONEY_QUERY_RE = re.compile(
    r"\b(money|spend|spent|cost|costs|expense|expenses|paid|price|total|"
    r"amount|earn|earned|earning|earnings|sell|sold|selling|"
    r"save|saved|saving|savings|discount|retail|retailed|originally|"
    r"pre-?approved|approved|mortgage|loan|balance|limit|worth|value|valued|salary|budget)\b|\$",
    re.IGNORECASE,
)
_COUNT_QUERY_RE = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
_SUM_QUERY_RE = re.compile(r"\b(how much|amount|total|sum|spent|cost|expenses?)\b|\$", re.IGNORECASE)
_COUNT_SNAPSHOT_QUERY_RE = re.compile(
    r"\b(so far|as of now|currently|current|latest|now|already|yet)\b|"
    r"\b(?:how many|number of|count)\b.*\b(?:have i|have we|i've|we've|do i have|do we have)\b",
    re.IGNORECASE,
)
_COUNT_STRONG_SNAPSHOT_QUERY_RE = re.compile(
    r"\b(so far|as of now|currently|current|latest|now|already|yet)\b",
    re.IGNORECASE,
)
_COUNT_ADDITIVE_QUERY_RE = re.compile(
    r"\b(in total|total number|overall|all together|altogether|during|between|from .+ to|"
    r"this year|this month|this week|last year|last month|past year|past month|past few|"
    r"each|per)\b",
    re.IGNORECASE,
)
_VALUE_SNAPSHOT_QUERY_RE = re.compile(
    r"\b(current|currently|latest|now|as of now|balance|limit|pre-?approved|approved for|"
    r"worth|valued|value|estimate|offer|salary|budget|rate)\b|"
    r"\bhow much\s+(?:is|was|are|were|am)\b",
    re.IGNORECASE,
)
_VALUE_ADDITIVE_QUERY_RE = re.compile(
    r"\b(total|sum|spent|spend|cost|costs|paid|pay|earn|earned|earning|earnings|"
    r"sell|sold|selling|raise|raised|donate|donated|save|saved|saving|savings|"
    r"expenses?|all together|altogether)\b",
    re.IGNORECASE,
)
_EVENT_SNAPSHOT_TEXT_RE = re.compile(
    r"\b(so far|as of|currently|current|now|already|yet|total|overall|in all|up to)\b",
    re.IGNORECASE,
)
_MAX_QUERY_RE = re.compile(
    r"\b(which|what)\b.*\b(most|highest|largest|biggest|max(?:imum)?|priciest|expensive)\b.*"
    r"\b(money|spend|spent|cost|paid|price|amount)\b|"
    r"\b(most|highest|largest|biggest|max(?:imum)?|priciest|expensive)\b.*"
    r"\b(money|spend|spent|cost|paid|price|amount)\b",
    re.IGNORECASE,
)
_DIFFERENCE_QUERY_RE = re.compile(
    r"\b(how much .*sav|save|saved|saving|savings|discount)\b",
    re.IGNORECASE,
)
_DATE_DELTA_QUERY_RE = re.compile(
    r"\bhow (?:many|long)\b.*\b(days?|weeks?|months?|years?)\b.*"
    r"\b(ago|since|after|before|passed|take|took)\b|"
    r"\bhow long\b.*\b(since|after|before|use|used)\b",
    re.IGNORECASE,
)
_TIMELINE_QUERY_RE = re.compile(
    r"\b(first|earliest|latest|order|ordered|timeline|history|chronological|when)\b",
    re.IGNORECASE,
)
_ORDER_QUERY_RE = re.compile(
    r"\b(which|who|what)\b.*\b(first|earliest|before|after)\b|"
    r"\b(first|second|third)\b.*\bamong\b|"
    r"\border\b.*\bamong\b",
    re.IGNORECASE,
)
_QUOTED_OPERAND_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_AMONG_RE = re.compile(r"\bamong\s+(?P<items>.+?)[?.!]?$", re.IGNORECASE)
_COMPARISON_TAIL_RE = re.compile(r",\s*(?P<items>[^?!.]+?)[?!.]?$")
_ORDINAL_WORDS = {"first", "second", "third", "fourth", "fifth"}
_ORDER_ACTION_CUES = {
    "arriv",
    "arrived",
    "attend",
    "attended",
    "becam",
    "became",
    "begin",
    "began",
    "bought",
    "buy",
    "complet",
    "completed",
    "decid",
    "decided",
    "finish",
    "finished",
    "fix",
    "fixed",
    "get",
    "got",
    "graduat",
    "graduated",
    "mov",
    "moved",
    "participat",
    "participated",
    "purchas",
    "purchased",
    "receiv",
    "received",
    "set",
    "start",
    "started",
    "visit",
    "visited",
    "watch",
    "watched",
}
_ORDER_NEGATIVE_CUES = {"delay", "delayed", "expected", "pre-order", "preordered", "pre-ordered"}
_LATEST_QUERY_TYPES = {"current", "latest", "final"}
_AUXILIARY_CATEGORIES = {"event", "expense"}
_EVENT_GENERIC_MATCH_TOKENS = {
    "attendance",
    "attend",
    "class",
    "conference",
    "concert",
    "event",
    "festival",
    "workshop",
}
_GENERIC_MATCH_TOKENS = {
    "ago",
    "all",
    "before",
    "after",
    "and",
    "appliance",
    "acquire",
    "acquired",
    "between",
    "bought",
    "buy",
    "city",
    "current",
    "currently",
    "date",
    "day",
    "days",
    "did",
    "event",
    "first",
    "for",
    "from",
    "get",
    "got",
    "go",
    "happen",
    "happened",
    "have",
    "how",
    "includ",
    "including",
    "kitchen",
    "last",
    "many",
    "april",
    "august",
    "december",
    "february",
    "january",
    "july",
    "june",
    "march",
    "may",
    "november",
    "october",
    "september",
    "money",
    "month",
    "much",
    "number",
    "one",
    "paid",
    "relat",
    "purchase",
    "purchased",
    "related",
    "expens",
    "few",
    "save",
    "saved",
    "saving",
    "savings",
    "since",
    "set",
    "spend",
    "spent",
    "start",
    "store",
    "past",
    "most",
    "the",
    "this",
    "that",
    "today",
    "total",
    "through",
    "throughout",
    "play",
    "playing",
    "tried",
    "try",
    "type",
    "item",
    "items",
    "what",
    "when",
    "went",
    "week",
    "weeks",
    "which",
    "with",
    "year",
    "years",
}
_CONCEPT_TOKEN_EXPANSIONS = {
    "babi": {
        "baby",
        "child",
        "children",
        "daughter",
        "son",
        "twin",
        "twins",
    },
    "baby": {
        "babi",
        "child",
        "children",
        "daughter",
        "son",
        "twin",
        "twins",
    },
    "born": {
        "baby",
        "birth",
        "child",
        "children",
        "daughter",
        "son",
        "twin",
        "twins",
        "welcomed",
    },
    "luxury": {
        "designer",
        "fashion",
        "gucci",
        "premium",
        "luxury",
    },
    "luxuri": {
        "designer",
        "fashion",
        "gucci",
        "premium",
        "luxury",
    },
}
_PERSONAL_QUERY_RE = re.compile(r"\b(i|me|my|mine)\b", re.IGNORECASE)
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_KNOWN_MERCHANT_LABELS = {
    "aldi": "Aldi",
    "amazon fresh": "Amazon Fresh",
    "costco": "Costco",
    "instacart": "Instacart",
    "kroger": "Kroger",
    "publix": "Publix",
    "safeway": "Safeway",
    "target": "Target",
    "thrive market": "Thrive Market",
    "trader joe's": "Trader Joe's",
    "trader joes": "Trader Joe's",
    "walmart": "Walmart",
    "whole foods": "Whole Foods",
}
_KNOWN_MERCHANT_RE = re.compile(
    r"\b(" + "|".join(re.escape(label) for label in sorted(_KNOWN_MERCHANT_LABELS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_PREPOSITION_GROUP_RE = re.compile(
    r"\b(?:at|from|via|through|with)\s+"
    r"(?P<label>[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,3})"
)


@dataclass(frozen=True)
class QueryTimeWindow:
    """A normalized time window inferred from a query."""

    start: date | None = None
    end: date | None = None
    label: str = ""

    def contains(self, event: MemoryEventRecord) -> bool:
        if self.start is None and self.end is None:
            return True
        event_start = _parse_date(event.event_start or event.observed_at)
        event_end = _parse_date(event.event_end or event.event_start or event.observed_at)
        if event_start is None and event_end is None:
            return False
        event_start = event_start or event_end
        event_end = event_end or event_start
        if self.start and event_end and event_end < self.start:
            return False
        if self.end and event_start and event_start > self.end:
            return False
        return True


@dataclass
class GraphQueryResult:
    """A computed graph-query result plus the evidence rows behind it."""

    operation: str
    answer_hint: str | None
    selected_event_ids: list[str]
    selected_claim_ids: list[str]
    selected_turn_ids: list[str]
    evidence_rows: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, *, max_rows: int = 24, include_answer_hint: bool = True) -> str:
        """Render a compact, source-oriented block for answer synthesis."""

        lines = [f"[graph-query: {self.operation}]"]
        if self.answer_hint and include_answer_hint:
            lines.append(self.answer_hint)
        filters = self.metadata.get("filters") or {}
        filter_parts = []
        for key in ("categories", "predicate", "time_window", "unit"):
            value = filters.get(key)
            if value:
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(item) for item in value)
                filter_parts.append(f"{key}={value}")
        if filter_parts:
            lines.append(f"Filters: {'; '.join(filter_parts)}")
        if self.evidence_rows:
            lines.append("Evidence rows:")
            for row in self.evidence_rows[:max_rows]:
                pieces = [
                    row.get("date") or "unknown-date",
                    row.get("event") or "",
                ]
                quantities = row.get("quantities") or []
                if quantities:
                    pieces.append(f"quantities={', '.join(quantities)}")
                categories = row.get("categories") or []
                if categories:
                    pieces.append(f"categories={', '.join(categories)}")
                if row.get("claim_id"):
                    pieces.append(f"claim={row['claim_id']}")
                turn_ids = row.get("turn_ids") or []
                if turn_ids:
                    pieces.append(f"turns={', '.join(turn_ids)}")
                lines.append("- " + " | ".join(pieces))
            if len(self.evidence_rows) > max_rows:
                lines.append(f"- ... {len(self.evidence_rows) - max_rows} more evidence rows")
        return "\n".join(lines)


def run_graph_query(
    index: MemoryIndex,
    query: str,
    *,
    query_type: str,
    anchor_time: str | None = None,
) -> GraphQueryResult | None:
    """Run an executable graph query when the plan calls for one."""

    if not index.events:
        return None

    operation = _infer_operation(query, query_type=query_type)
    inferred_categories = infer_category_ids(query)
    query_categories = _required_query_categories(inferred_categories)
    query_predicate = infer_predicate(query)
    if query_predicate == "purchase" and re.search(r"\bchronological\s+order\b|\bin\s+order\b", query, re.IGNORECASE):
        query_predicate = "state"
    if operation == "count" and (
        _looks_like_count_snapshot_query(query) or _requested_unit(query) in {"hour", "minute", "day", "week", "month", "year"}
    ):
        query_predicate = "state"
    if operation == "sum" and _looks_like_value_snapshot_query(query):
        query_predicate = "state"
    if re.search(r"\b(pre-?approved|approved for|balance|credit limit|mortgage|loan)\b", query, re.IGNORECASE):
        query_predicate = "state"
    time_window = infer_query_time_window(query, anchor_time=anchor_time)
    has_temporal_filter = time_window.start is not None or time_window.end is not None
    if (
        query_type not in {"aggregate", "temporal", *_LATEST_QUERY_TYPES}
        and not _looks_like_graph_query(query)
        and not (has_temporal_filter and query_predicate != "state")
    ):
        return None
    matched = _matching_events(
        index,
        query,
        inferred_categories=inferred_categories,
        query_categories=query_categories,
        query_predicate=query_predicate,
        time_window=time_window,
    )
    matched = _dedupe_events(matched)

    if operation == "order":
        return _order_result(
            index,
            matched,
            query=query,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
    if operation == "date_delta":
        return _date_delta_result(
            index,
            matched,
            query=query,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
            anchor_time=anchor_time,
        )
    if operation == "latest":
        return _latest_result(
            index,
            matched,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
    if operation == "difference":
        return _difference_result(
            index,
            matched,
            query=query,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
    if operation == "max":
        return _max_result(
            index,
            matched,
            query=query,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
    if operation == "sum":
        return _sum_result(
            index,
            matched,
            query=query,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
    if operation == "count":
        return _count_result(
            index,
            matched,
            query=query,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
    return _timeline_result(
        index,
        matched,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
    )


def infer_query_time_window(query: str, *, anchor_time: str | None = None) -> QueryTimeWindow:
    """Infer a time window from common benchmark-style temporal phrases."""

    anchor = _parse_anchor(anchor_time)
    lower = query.lower()
    if anchor and re.search(r"\bsince the start of the year\b|\bthis year\b|\byear to date\b", lower):
        return QueryTimeWindow(
            start=date(anchor.year, 1, 1),
            end=anchor,
            label=f"{anchor.year}-01-01..{anchor.isoformat()}",
        )
    if anchor and re.search(r"\b(past|last|previous)\s+month\b", lower):
        start = anchor - timedelta(days=30)
        return QueryTimeWindow(start=start, end=anchor, label=f"{start.isoformat()}..{anchor.isoformat()}")
    if anchor and re.search(r"\b(past|last|previous)\s+week\b", lower):
        start = anchor - timedelta(days=7)
        return QueryTimeWindow(start=start, end=anchor, label=f"{start.isoformat()}..{anchor.isoformat()}")
    if anchor and re.search(r"\b(past|last|previous)\s+year\b", lower):
        start = anchor - timedelta(days=365)
        return QueryTimeWindow(start=start, end=anchor, label=f"{start.isoformat()}..{anchor.isoformat()}")

    month_match = re.search(r"\bin\s+(?P<month>[A-Za-z]+)(?:\s+(?P<year>20\d{2}))?\b", query)
    if anchor and month_match:
        month = _MONTHS.get(month_match.group("month").lower())
        if month:
            year = int(month_match.group("year") or anchor.year)
            last_day = calendar.monthrange(year, month)[1]
            return QueryTimeWindow(
                start=date(year, month, 1),
                end=date(year, month, last_day),
                label=f"{year:04d}-{month:02d}-01..{year:04d}-{month:02d}-{last_day:02d}",
            )

    refs = extract_temporal_refs(query, anchor_time=anchor_time)
    if len(refs) >= 2 and re.search(r"\bbetween\b|\bfrom\b", lower):
        dates = [
            parsed
            for ref in refs[:2]
            for parsed in (_parse_date(ref.resolved_start or ref.resolved_end),)
            if parsed is not None
        ]
        if len(dates) == 2:
            start, end = sorted(dates)
            return QueryTimeWindow(start=start, end=end, label=_window_label(start, end))
    if refs:
        ref = refs[0]
        start = _parse_date(ref.resolved_start)
        end = _parse_date(ref.resolved_end)
        if start or end:
            if "before" in lower and start:
                return QueryTimeWindow(end=start, label=f"..{start.isoformat()}")
            if ("after" in lower or "since" in lower) and start:
                return QueryTimeWindow(
                    start=start,
                    end=anchor,
                    label=f"{start.isoformat()}..{anchor.isoformat() if anchor else ''}",
                )
            return QueryTimeWindow(start=start, end=end or start, label=_window_label(start, end or start))
    return QueryTimeWindow()


def _looks_like_graph_query(query: str) -> bool:
    return bool(
        _COUNT_QUERY_RE.search(query)
        or _SUM_QUERY_RE.search(query)
        or _MAX_QUERY_RE.search(query)
        or _DATE_DELTA_QUERY_RE.search(query)
        or _TIMELINE_QUERY_RE.search(query)
    )


def _infer_operation(query: str, *, query_type: str) -> str:
    if _ORDER_QUERY_RE.search(query):
        return "order"
    if _DATE_DELTA_QUERY_RE.search(query):
        return "date_delta"
    if _DIFFERENCE_QUERY_RE.search(query):
        return "difference"
    if _MAX_QUERY_RE.search(query):
        return "max"
    if _COUNT_QUERY_RE.search(query):
        return "count"
    if _SUM_QUERY_RE.search(query):
        return "sum"
    if query_type in _LATEST_QUERY_TYPES:
        return "latest"
    if query_type == "aggregate":
        return "count"
    return "timeline"


def _matching_events(
    index: MemoryIndex,
    query: str,
    *,
    inferred_categories: tuple[str, ...],
    query_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> list[MemoryEventRecord]:
    q_tokens = _specific_query_tokens(query, inferred_categories)
    focus_phrases = _event_focus_phrases(query, inferred_categories)
    out: list[MemoryEventRecord] = []
    for event in index.events:
        if event.metadata.get("claim_status") == "superseded":
            continue
        if event.metadata.get("polarity") == "negative":
            continue
        if query_categories and not all(category in event.category_ids for category in query_categories):
            continue
        if query_predicate != "state" and not _event_predicate_matches(
            query_predicate,
            event.predicate,
            query_categories=query_categories,
        ):
            continue
        if not time_window.contains(event):
            continue
        if q_tokens and not _query_tokens_match(q_tokens, significant_tokens(_event_match_text(index, event))):
            continue
        out.append(event)
    if _is_generic_event_query(query_categories) and focus_phrases:
        phrase_matched = [
            event
            for event in out
            if any(phrase in _normalized_token_sequence(_event_match_text(index, event)) for phrase in focus_phrases)
        ]
        if phrase_matched:
            out = phrase_matched
    return sorted(out, key=lambda event: (event.event_start or event.observed_at or "", event.sort_key, event.event_id))


def _event_predicate_matches(
    query_predicate: str,
    event_predicate: str,
    *,
    query_categories: tuple[str, ...],
) -> bool:
    if predicates_compatible(query_predicate, event_predicate):
        return True
    if "charity" in query_categories and query_predicate in {"attend", "donate"}:
        return event_predicate in {"attend", "donate"}
    if "health" in query_categories and query_predicate == "schedule":
        return event_predicate in {"schedule", "attend", "visit", "state"}
    return False


def _dedupe_events(events: Iterable[MemoryEventRecord]) -> list[MemoryEventRecord]:
    seen: set[str] = set()
    out: list[MemoryEventRecord] = []
    for event in events:
        key = str(event.metadata.get("dedupe_key") or event.event_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def _count_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    events = _prefer_user_personal_events(events, query=query)
    if query_predicate == "birth":
        events = _prefer_user_events(events)
    named_count = _count_named_birth_entities(events, query=query)
    if named_count:
        names, events = named_count
        count_value = float(len(names))
        count_label = _format_number(count_value)
        answer = f"Computed count: {count_label}"
        metadata = {"count_method": "named_entity_count", "entity_names": names, "answer_confidence": 0.86}
    else:
        snapshot = _latest_count_snapshot(events, query=query, query_categories=query_categories)
        if snapshot:
            value, snapshot_event, snapshot_quantity, prior_values = snapshot
            count_label = _format_number(value)
            answer = f"Latest matching count: {count_label}"
            events = [snapshot_event]
            metadata = {
                "count_method": "latest_quantity_snapshot",
                "quantity_values": [value],
                "snapshot_values": prior_values,
                "snapshot_event_id": snapshot_event.event_id,
                "snapshot_unit": _normalize_unit(snapshot_quantity.unit),
                "answer_confidence": 0.88,
            }
        else:
            quantity_values = _count_quantity_values(events, query=query, query_categories=query_categories)
            if quantity_values:
                count_value = sum(quantity_values)
                count_label = _format_number(count_value)
                answer = f"Computed count: {count_label}"
                metadata = {
                    "count_method": "quantity_sum",
                    "quantity_values": quantity_values,
                    "answer_confidence": 0.72,
                }
            else:
                count_value = float(len(events))
                answer = f"Computed count: {int(count_value)}"
                metadata = {"count_method": "event_count", "answer_confidence": 0.58}
    return _result(
        index,
        events,
        operation="aggregate/count",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
        metadata=metadata,
    )


def _latest_count_snapshot(
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
) -> tuple[float, MemoryEventRecord, QuantityClaim, list[float]] | None:
    if not _looks_like_count_snapshot_query(query):
        return None

    strong_query_cue = bool(_COUNT_STRONG_SNAPSHOT_QUERY_RE.search(query))
    candidates: list[tuple[tuple, float, MemoryEventRecord, QuantityClaim]] = []
    for event in events:
        event_cue = _event_has_snapshot_cue(event)
        if not strong_query_cue and not event_cue and event.predicate != "state":
            continue
        for quantity in event.quantity_claims:
            if quantity.value is None:
                continue
            if not _quantity_matches_count_query(quantity, query=query, query_categories=query_categories):
                continue
            score_key = _snapshot_sort_key(event, quantity, event_cue=event_cue)
            candidates.append((score_key, float(quantity.value), event, quantity))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, value, event, quantity = candidates[-1]
    prior_values = [candidate_value for _, candidate_value, _, _ in candidates]
    return value, event, quantity, prior_values


def _looks_like_count_snapshot_query(query: str) -> bool:
    if not _COUNT_QUERY_RE.search(query):
        return False
    if _COUNT_STRONG_SNAPSHOT_QUERY_RE.search(query):
        return True
    return bool(_COUNT_SNAPSHOT_QUERY_RE.search(query) and not _COUNT_ADDITIVE_QUERY_RE.search(query))


def _quantity_matches_count_query(
    quantity: QuantityClaim,
    *,
    query: str,
    query_categories: tuple[str, ...],
) -> bool:
    unit = _normalize_unit(quantity.unit)
    if not unit:
        return False
    query_tokens = _query_count_unit_tokens(query)
    category_units = _category_count_units(query_categories)
    return unit in query_tokens or unit in category_units


def _event_has_snapshot_cue(event: MemoryEventRecord) -> bool:
    return bool(_EVENT_SNAPSHOT_TEXT_RE.search(event.text))


def _snapshot_sort_key(
    event: MemoryEventRecord,
    quantity: QuantityClaim,
    *,
    event_cue: bool,
) -> tuple:
    role_priority = 2 if event.metadata.get("role") == "user" else 1
    predicate_priority = 2 if event.predicate == "state" else 1
    confidence = float(quantity.confidence or 0.0)
    return (
        event.event_start or event.observed_at or "",
        event.sort_key,
        role_priority,
        predicate_priority,
        1 if event_cue else 0,
        confidence,
        event.event_id,
    )


def _count_named_birth_entities(
    events: list[MemoryEventRecord],
    *,
    query: str,
) -> tuple[list[str], list[MemoryEventRecord]] | None:
    query_tokens = significant_tokens(query)
    if not ({"baby", "babie", "babi", "child", "children"} & query_tokens and {"born", "birth"} & query_tokens):
        return None

    names: list[str] = []
    selected_events: list[MemoryEventRecord] = []
    for event in events:
        if event.metadata.get("role") != "user":
            continue
        if event.predicate != "birth":
            continue
        event_names = _birth_entity_names(event.text)
        if not event_names:
            continue
        added = False
        for name in event_names:
            if name not in names:
                names.append(name)
                added = True
        if added:
            selected_events.append(event)
    if not names:
        return None
    return names, selected_events


def _birth_entity_names(text: str) -> list[str]:
    names: list[str] = []
    patterns = (
        r"\b(?:baby|boy|girl|son|daughter|child|children|twins?|girls?)\s+named\s+"
        r"(?P<names>[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?)",
        r"\bnamed\s+(?P<names>[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?)\s+who\s+were\s+born\b",
        r"\bnamed\s+(?P<names>[A-Z][a-z]+)\s+who\s+was\s+born\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            for name in re.split(r"\s+and\s+", match.group("names")):
                if name not in names:
                    names.append(name)
    return names


def _sum_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    unit = "usd" if _MONEY_QUERY_RE.search(query) else _requested_unit(query)
    events = _prefer_user_value_events(events, query=query, unit=unit)
    snapshot = _latest_value_snapshot(events, query=query, unit=unit)
    if snapshot:
        value, snapshot_event, snapshot_quantity, prior_values = snapshot
        unit_label = _normalize_unit(snapshot_quantity.unit) or unit or "quantity"
        answer = f"Latest matching value: {_format_quantity_value(value, unit_label)}"
        return _result(
            index,
            [snapshot_event],
            operation="aggregate/sum",
            answer_hint=answer,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
            unit=unit_label,
            metadata={
                "sum_method": "latest_quantity_snapshot",
                "sum_values": [value],
                "snapshot_values": prior_values,
                "snapshot_event_id": snapshot_event.event_id,
                "answer_confidence": 0.86,
            },
        )

    total = 0.0
    values: list[float] = []
    for event in events:
        event_values = _sum_values_for_event(event, unit=unit)
        total += sum(event_values)
        values.extend(event_values)
    unit_label = unit or "quantity"
    if values:
        answer = f"Computed sum: {_format_quantity_value(total, unit_label)}"
    else:
        answer = "No matching quantity values found for sum."
    return _result(
        index,
        events,
        operation="aggregate/sum",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
        unit=unit_label,
        metadata={"sum_values": values, "answer_confidence": 0.72 if values else 0.3},
    )


def _latest_value_snapshot(
    events: list[MemoryEventRecord],
    *,
    query: str,
    unit: str | None,
) -> tuple[float, MemoryEventRecord, QuantityClaim, list[float]] | None:
    if not _looks_like_value_snapshot_query(query):
        return None

    candidates: list[tuple[tuple, float, MemoryEventRecord, QuantityClaim]] = []
    for event in events:
        event_cue = _event_has_snapshot_cue(event)
        for quantity in event.quantity_claims:
            if quantity.value is None:
                continue
            normalized_unit = _normalize_unit(quantity.unit)
            if unit and normalized_unit != unit:
                continue
            score_key = _snapshot_sort_key(event, quantity, event_cue=event_cue)
            candidates.append((score_key, float(quantity.value), event, quantity))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, value, event, quantity = candidates[-1]
    prior_values = [candidate_value for _, candidate_value, _, _ in candidates]
    return value, event, quantity, prior_values


def _looks_like_value_snapshot_query(query: str) -> bool:
    if not _SUM_QUERY_RE.search(query):
        return False
    if _VALUE_ADDITIVE_QUERY_RE.search(query):
        return False
    return bool(_VALUE_SNAPSHOT_QUERY_RE.search(query))


def _sum_values_for_event(event: MemoryEventRecord, *, unit: str | None) -> list[float]:
    if unit == "usd":
        revenue_value = _per_unit_money_total(event)
        if revenue_value is not None:
            return [revenue_value]

    values: list[float] = []
    for quantity in event.quantity_claims:
        normalized_unit = _normalize_unit(quantity.unit)
        if unit and normalized_unit != unit:
            continue
        if quantity.value is None:
            continue
        values.append(float(quantity.value))
    return values


def _per_unit_money_total(event: MemoryEventRecord) -> float | None:
    if not re.search(r"\b(each|apiece|per)\b", event.text, re.IGNORECASE):
        return None
    money_values = [
        float(quantity.value)
        for quantity in event.quantity_claims
        if quantity.value is not None and _normalize_unit(quantity.unit) == "usd"
    ]
    count_values = [
        float(quantity.value)
        for quantity in event.quantity_claims
        if quantity.value is not None
        and _normalize_unit(quantity.unit) not in {None, "usd", "%"}
        and float(quantity.value) > 1
    ]
    if len(money_values) != 1 or not count_values:
        return None
    return money_values[0] * max(count_values)


def _difference_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    values: list[float] = []
    for event in events:
        for quantity in event.quantity_claims:
            if quantity.value is None or _normalize_unit(quantity.unit) != "usd":
                continue
            values.append(float(quantity.value))
    if len(values) >= 2:
        high = max(values)
        low = min(values)
        diff = high - low
        answer = (
            f"Computed difference: {_format_quantity_value(diff, 'usd')} "
            f"({_format_quantity_value(high, 'usd')} - {_format_quantity_value(low, 'usd')})"
        )
    else:
        answer = "No matching money values found for difference."
    return _result(
        index,
        events,
        operation="aggregate/difference",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
        unit="usd",
        metadata={"difference_values": values},
    )


def _max_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    unit = "usd" if _MONEY_QUERY_RE.search(query) else _requested_unit(query)
    events = _prefer_user_value_events(events, query=query, unit=unit)
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        values = [
            float(quantity.value)
            for quantity in event.quantity_claims
            if quantity.value is not None and (not unit or _normalize_unit(quantity.unit) == unit)
        ]
        if not values:
            continue
        label = _event_group_label(event.text, query=query)
        if not label:
            continue
        bucket = grouped.setdefault(label, {"total": 0.0, "events": []})
        bucket["total"] += sum(values)
        bucket["events"].append(event)

    unit_label = unit or "quantity"
    metadata: dict[str, Any] = {"group_values": {label: data["total"] for label, data in grouped.items()}}
    selected_events: list[MemoryEventRecord] = []
    if grouped:
        label, data = max(grouped.items(), key=lambda item: (item[1]["total"], item[0]))
        answer = f"Maximum matching spend: {label} ({_format_quantity_value(data['total'], unit_label)})"
        metadata.update({"max_group": label, "max_value": data["total"]})
        for _, group_data in sorted(grouped.items(), key=lambda item: (-item[1]["total"], item[0])):
            selected_events.extend(group_data["events"])
    else:
        answer = "No matching grouped quantity values found."

    return _result(
        index,
        selected_events,
        operation="aggregate/max",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
        unit=unit_label,
        metadata=metadata,
    )


def _prefer_user_value_events(
    events: list[MemoryEventRecord],
    *,
    query: str,
    unit: str | None,
) -> list[MemoryEventRecord]:
    """For personal value questions, avoid summing assistant advice as user history."""

    if not _PERSONAL_QUERY_RE.search(query):
        return events
    user_events = [
        event
        for event in events
        if event.metadata.get("role") == "user"
        and any(
            quantity.value is not None and (not unit or _normalize_unit(quantity.unit) == unit)
            for quantity in event.quantity_claims
        )
    ]
    return user_events or events


def _prefer_user_personal_events(events: list[MemoryEventRecord], *, query: str) -> list[MemoryEventRecord]:
    if not _PERSONAL_QUERY_RE.search(query):
        return events
    user_events = [event for event in events if event.metadata.get("role") == "user"]
    return user_events or events


def _prefer_user_events(events: list[MemoryEventRecord]) -> list[MemoryEventRecord]:
    user_events = [event for event in events if event.metadata.get("role") == "user"]
    return user_events or events


def _order_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    operands = _comparison_operands(query)
    if len(operands) < 2:
        return _timeline_result(
            index,
            events,
            query_categories=query_categories,
            inferred_categories=inferred_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )

    candidates_by_operand: dict[str, list[tuple[date, MemoryEventRecord]]] = {}
    for operand in operands:
        candidates = _dated_operand_events(
            index,
            operand,
            query=query,
            query_categories=query_categories,
            query_predicate=query_predicate,
            time_window=time_window,
        )
        if candidates:
            candidates_by_operand[operand] = candidates

    found_operands = list(candidates_by_operand)
    missing_operands = [operand for operand in operands if operand not in candidates_by_operand]
    selected: list[tuple[str, date, MemoryEventRecord]] = [
        (operand, candidates[0][0], candidates[0][1])
        for operand, candidates in candidates_by_operand.items()
    ]
    selected.sort(key=lambda item: (item[1], item[2].sort_key, item[0]))
    selected_events = [event for _, _, event in selected]

    metadata: dict[str, Any] = {
        "operands": operands,
        "found_operands": found_operands,
        "missing_operands": missing_operands,
        "comparison_complete": not missing_operands,
        "answer_confidence": 0.86 if not missing_operands else 0.9,
    }
    if missing_operands:
        answer = (
            "Insufficient comparison evidence: "
            f"found dated evidence for {_join_labels(found_operands) or 'none'}, "
            f"but not for {_join_labels(missing_operands)}."
        )
    else:
        ordered = [f"{operand} ({dt.isoformat()})" for operand, dt, _ in selected]
        first = selected[0][0]
        answer = f"Computed temporal order: {' -> '.join(ordered)}. First: {first}."
    return _result(
        index,
        selected_events,
        operation="temporal/order",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
        metadata=metadata,
    )


def _comparison_operands(query: str) -> list[str]:
    quoted = [_clean_operand(item) for item in _QUOTED_OPERAND_RE.findall(query)]
    if len(quoted) >= 2:
        return _dedupe(item for item in quoted if item)

    among = _AMONG_RE.search(query)
    if among:
        return _split_operand_list(among.group("items"))

    tail_match = _COMPARISON_TAIL_RE.search(query)
    tail = tail_match.group("items") if tail_match else query
    if re.search(r"\bor\b|\band\b", tail, re.IGNORECASE):
        operands = _split_operand_list(tail)
        if len(operands) >= 2:
            return operands
    return []


def _split_operand_list(text: str) -> list[str]:
    cleaned = re.sub(r"\b(first|second|third|fourth|fifth)\b", "", text, flags=re.IGNORECASE)
    parts = re.split(r"\s*,\s*|\s+\bor\b\s+|\s+\band\b\s+", cleaned)
    return _dedupe(_clean_operand(part) for part in parts if _clean_operand(part))


def _clean_operand(text: str) -> str:
    text = re.sub(r"[?.!]+$", "", text.strip())
    text = re.sub(r"^(?:the|a|an|my|our|your)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:attendance at|attending|start of|beginning of|completion of|finishing|"
        r"day i|day you|time i|time you)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:the|a|an|my|our|your)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:first|earliest|latest)\b", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    if not text:
        return ""
    tokens = text.lower().split()
    if len(tokens) == 1 and tokens[0] in _ORDINAL_WORDS:
        return ""
    return text


def _dated_operand_events(
    index: MemoryIndex,
    operand: str,
    *,
    query: str,
    query_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> list[tuple[date, MemoryEventRecord]]:
    operand_tokens = significant_tokens(operand)
    out: list[tuple[date, MemoryEventRecord]] = []
    for event in index.events:
        if event.metadata.get("claim_status") == "superseded":
            continue
        if event.metadata.get("polarity") == "negative":
            continue
        if query_categories and not any(category in event.category_ids for category in query_categories):
            continue
        if not time_window.contains(event):
            continue
        event_text = _event_match_text(index, event)
        event_tokens = significant_tokens(event_text)
        if operand_tokens and not operand_tokens <= event_tokens:
            phrase = _normalized_token_sequence(operand)
            if phrase not in _normalized_token_sequence(event_text):
                continue
        event_date = _date_for_operand(event, operand=operand, query=query)
        if event_date is None:
            continue
        out.append((event_date, event))
    out.sort(key=lambda item: (item[0], item[1].sort_key, item[1].event_id))
    return out


def _date_for_operand(event: MemoryEventRecord, *, operand: str, query: str) -> date | None:
    refs = [
        ref for ref in event.temporal_refs
        if _parse_date(ref.resolved_start or ref.resolved_end) is not None
    ]
    if not refs:
        return _parse_date(event.event_start or event.observed_at)

    text = event.text
    operand_pos = text.lower().find(operand.lower())
    query_tokens = significant_tokens(query)
    scored: list[tuple[float, date, str]] = []
    for ref in refs:
        ref_date = _parse_date(ref.resolved_start or ref.resolved_end)
        if ref_date is None:
            continue
        ref_pos = text.lower().find(ref.text.lower())
        score = 0.0
        if operand_pos >= 0 and ref_pos >= 0:
            score -= min(60, abs(ref_pos - operand_pos)) / 60
        if ref_pos >= 0 and query_tokens & _ORDER_ACTION_CUES:
            score += _cue_proximity(text, ref_pos, _ORDER_ACTION_CUES, radius=56)
        if ref_pos >= 0:
            score -= 1.25 * _cue_proximity(text, ref_pos, _ORDER_NEGATIVE_CUES, radius=56)
        scored.append((score, ref_date, ref.text))
    if not scored:
        return _parse_date(event.event_start or event.observed_at)
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return scored[0][1]


def _cue_proximity(text: str, ref_pos: int, cues: set[str], *, radius: int) -> float:
    lower = text.lower()
    best = 0.0
    for cue in cues:
        for match in re.finditer(re.escape(cue), lower):
            distance = abs(match.start() - ref_pos)
            if distance <= radius:
                best = max(best, 1.0 - (distance / radius))
    return best


def _join_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _date_delta_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
    anchor_time: str | None,
) -> GraphQueryResult:
    dated_events = [
        event
        for event in events
        if _parse_date(event.event_start or event.observed_at) is not None
    ]
    dated_events.sort(key=lambda event: (event.event_start or event.observed_at or "", event.sort_key, event.event_id))
    unit = _requested_delta_unit(query)
    anchor = _parse_anchor(anchor_time)
    answer = "No matching dated events found for date difference."
    metadata: dict[str, Any] = {"delta_unit": unit}
    if len(dated_events) >= 2:
        start = _parse_date(dated_events[0].event_start or dated_events[0].observed_at)
        end = _parse_date(dated_events[-1].event_start or dated_events[-1].observed_at)
        if start and end:
            days = abs((end - start).days)
            answer = (
                f"Computed date difference: {_format_delta(days, unit)} "
                f"({start.isoformat()} to {end.isoformat()})"
            )
            metadata.update({"delta_days": days, "delta_start": start.isoformat(), "delta_end": end.isoformat()})
    elif dated_events and anchor:
        event_date = _parse_date(dated_events[0].event_start or dated_events[0].observed_at)
        if event_date:
            days = abs((anchor - event_date).days)
            suffix = " ago" if event_date <= anchor else " from anchor"
            answer = (
                f"Computed date difference: {_format_delta(days, unit)}{suffix} "
                f"({event_date.isoformat()} to {anchor.isoformat()})"
            )
            metadata.update({"delta_days": days, "delta_start": event_date.isoformat(), "delta_end": anchor.isoformat()})
    return _result(
        index,
        dated_events,
        operation="temporal/date-delta",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
        unit=unit,
        metadata=metadata,
    )


def _latest_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    events = sorted(
        events,
        key=lambda event: (
            event.event_start or event.observed_at or "",
            _latest_role_priority(event),
            _latest_predicate_priority(event),
            event.sort_key,
            event.event_id,
        ),
        reverse=True,
    )
    if events:
        latest = events[0]
        latest_date = latest.event_start or latest.observed_at or "unknown-date"
        answer = f"Latest matching evidence: {latest.text} ({latest_date})"
    else:
        answer = "No matching latest/current evidence found."
    return _result(
        index,
        events,
        operation="temporal/latest",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
    )


def _latest_role_priority(event: MemoryEventRecord) -> int:
    role = str(event.metadata.get("role") or "")
    if role == "user":
        return 2
    if role == "assistant":
        return 1
    return 0


def _latest_predicate_priority(event: MemoryEventRecord) -> int:
    if event.predicate == "state":
        return 2
    return 1


def _timeline_result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> GraphQueryResult:
    events = sorted(events, key=lambda event: (event.event_start or event.observed_at or "", event.sort_key, event.event_id))
    if events:
        answer = f"Computed timeline rows: {len(events)}"
    else:
        answer = "Computed timeline rows: 0"
    return _result(
        index,
        events,
        operation="temporal/timeline",
        answer_hint=answer,
        query_categories=query_categories,
        inferred_categories=inferred_categories,
        query_predicate=query_predicate,
        time_window=time_window,
    )


def _result(
    index: MemoryIndex,
    events: list[MemoryEventRecord],
    *,
    operation: str,
    answer_hint: str,
    query_categories: tuple[str, ...],
    inferred_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
    unit: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraphQueryResult:
    rows = [_event_row(index, event) for event in events]
    claim_ids = _dedupe(
        event.source_claim_id for event in events if event.source_claim_id
    )
    turn_ids = _dedupe(
        turn_id for event in events for turn_id in event.source_turn_ids
    )
    filter_metadata: dict[str, Any] = {
        "categories": [category_label(category_id) for category_id in query_categories],
        "inferred_categories": [category_label(category_id) for category_id in inferred_categories],
        "predicate": query_predicate if query_predicate != "state" else "",
        "time_window": time_window.label,
    }
    if unit:
        filter_metadata["unit"] = unit
    return GraphQueryResult(
        operation=operation,
        answer_hint=answer_hint,
        selected_event_ids=[event.event_id for event in events],
        selected_claim_ids=claim_ids,
        selected_turn_ids=turn_ids,
        evidence_rows=rows,
        metadata={
            "matched_events": len(events),
            "filters": filter_metadata,
            **(metadata or {}),
        },
    )


def _event_row(index: MemoryIndex, event: MemoryEventRecord) -> dict[str, Any]:
    entity_labels = [
        index.by_entity_id[entity_id].label
        for entity_id in event.entity_refs
        if entity_id in index.by_entity_id
    ]
    return {
        "event_id": event.event_id,
        "date": event.event_start or event.observed_at,
        "event": event.text,
        "predicate": event.predicate,
        "entities": entity_labels,
        "categories": [category_label(category_id) for category_id in event.category_ids],
        "quantities": [_format_quantity(quantity) for quantity in event.quantity_claims],
        "claim_id": event.source_claim_id,
        "turn_ids": list(event.source_turn_ids),
    }


def _event_match_text(index: MemoryIndex, event: MemoryEventRecord) -> str:
    entity_text = " ".join(
        index.by_entity_id[entity_id].label
        for entity_id in event.entity_refs
        if entity_id in index.by_entity_id
    )
    category_text = " ".join(category_label(category_id) for category_id in event.category_ids)
    return f"{event.text} {event.predicate} {entity_text} {category_text}"


def _event_group_label(text: str, *, query: str) -> str | None:
    merchant = _KNOWN_MERCHANT_RE.search(text)
    if merchant:
        return _KNOWN_MERCHANT_LABELS[merchant.group(1).lower()]

    if re.search(r"\b(store|market|merchant|vendor|restaurant|airline|hotel)\b", query, re.IGNORECASE):
        label = _PREPOSITION_GROUP_RE.search(text)
        if label:
            return _clean_group_label(label.group("label"))
    return None


def _clean_group_label(label: str) -> str:
    label = re.split(
        r"\b(?:last|this|next|during|for|on|with|and|where|which|that|who)\b",
        label,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(label.strip(" .,'\"").split())


def _required_query_categories(category_ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(category_ids) <= 1:
        return category_ids
    required = tuple(
        category_id
        for category_id in category_ids
        if category_id not in _AUXILIARY_CATEGORIES
    )
    return required or category_ids


def _specific_query_tokens(query: str, inferred_categories: tuple[str, ...]) -> set[str]:
    category_tokens = {
        token
        for category_id in inferred_categories
        for token in significant_tokens(category_label(category_id))
    }
    if "health" in inferred_categories:
        category_tokens.update(
            {
                "appointment",
                "checkup",
                "doctor",
                "follow",
                "health",
                "physician",
                "surgeon",
            }
        )
    return {
        token
        for token in significant_tokens(query) - _GENERIC_MATCH_TOKENS - category_tokens
        if not token.isdigit()
    }


def _query_tokens_match(query_tokens: set[str], event_tokens: set[str]) -> bool:
    if query_tokens & event_tokens:
        return True
    for token in query_tokens:
        if _CONCEPT_TOKEN_EXPANSIONS.get(token, set()) & event_tokens:
            return True
    return False


def _is_generic_event_query(query_categories: tuple[str, ...]) -> bool:
    return query_categories == ("event",)


def _event_focus_phrases(query: str, inferred_categories: tuple[str, ...]) -> tuple[str, ...]:
    if "event" not in inferred_categories:
        return ()
    category_tokens = {
        token
        for category_id in inferred_categories
        for token in significant_tokens(category_label(category_id))
    }
    tokens = [
        token
        for token in _ordered_normalized_tokens(query)
        if token not in _GENERIC_MATCH_TOKENS
        and token not in category_tokens
        and not token.isdigit()
    ]
    phrases: list[str] = []
    for left, right in zip(tokens, tokens[1:]):
        if left in _EVENT_GENERIC_MATCH_TOKENS and right in _EVENT_GENERIC_MATCH_TOKENS:
            continue
        phrase = f"{left} {right}"
        if phrase not in phrases:
            phrases.append(phrase)
    return tuple(phrases)


def _normalized_token_sequence(text: str) -> str:
    return " ".join(_ordered_normalized_tokens(text))


def _ordered_normalized_tokens(text: str) -> list[str]:
    return [
        normalize_token(match.group(0).lower())
        for match in re.finditer(r"[A-Za-z0-9]+", text)
        if len(match.group(0)) >= 3
    ]


def _count_quantity_values(
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    query_tokens = _query_count_unit_tokens(query)
    category_units = _category_count_units(query_categories)
    for event in events:
        for quantity in event.quantity_claims:
            if quantity.value is None:
                continue
            unit = _normalize_unit(quantity.unit)
            if not unit:
                continue
            if unit in category_units or unit in query_tokens:
                values.append(float(quantity.value))
    return values


def _query_count_unit_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z&'.-]*", query.lower()):
        raw = match.group(0).strip(" .'\"")
        if len(raw) < 2:
            continue
        tokens.add(normalize_token(raw))
        normalized_unit = _normalize_unit(raw)
        if normalized_unit:
            tokens.add(normalized_unit)
    return tokens


def _category_count_units(category_ids: tuple[str, ...]) -> set[str]:
    units: set[str] = set()
    for category_id in category_ids:
        units.add(category_id)
        if category_id == "plant":
            units.update({"plant", "plants"})
        if category_id == "clothing":
            units.update({"top", "tops", "shirt", "shirts", "dress", "dresses", "shoe", "shoes"})
        if category_id == "event":
            units.update({"event", "wedding", "museum", "class", "workshop"})
        if category_id == "family":
            units.update({"baby", "babies", "wedding"})
        if category_id == "food":
            units.update({"restaurant", "restaurants"})
        if category_id == "gaming":
            units.update({"hour", "hours"})
    return {_normalize_unit(unit) or unit for unit in units}


def _requested_unit(query: str) -> str | None:
    tokens = significant_tokens(query)
    for token in tokens:
        unit = _normalize_unit(token)
        if unit in {"usd", "day", "week", "month", "year", "hour", "minute", "page", "plant"}:
            return unit
    return None


def _requested_delta_unit(query: str) -> str:
    tokens = significant_tokens(query)
    if "week" in tokens:
        return "week"
    if "month" in tokens:
        return "month"
    if "year" in tokens:
        return "year"
    return "day"


def _format_delta(days: int, unit: str) -> str:
    if unit == "week":
        value = days / 7
        return f"{_format_number(value)} weeks"
    if unit == "month":
        value = days / 30
        return f"{_format_number(value)} months"
    if unit == "year":
        value = days / 365
        return f"{_format_number(value)} years"
    return f"{days} days"


def _format_quantity(quantity: QuantityClaim) -> str:
    if quantity.value is None:
        return quantity.source_text
    unit = _normalize_unit(quantity.unit)
    return _format_quantity_value(float(quantity.value), unit)


def _format_quantity_value(value: float, unit: str | None) -> str:
    number = _format_number(value)
    if unit == "usd":
        return f"${number}"
    if unit:
        return f"{number} {unit}"
    return number


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    normalized = unit.lower().strip()
    if normalized in {"$", "dollar", "dollars", "usd"}:
        return "usd"
    if normalized == "percent":
        return "%"
    if normalized == "babies":
        return "baby"
    if normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]
    return normalized


def _parse_anchor(anchor_time: str | None) -> date | None:
    if not anchor_time:
        return None
    parsed = _parse_date(anchor_time)
    if parsed:
        return parsed
    normalized = anchor_time.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    match = re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value)
    if not match:
        return None
    year, month, day = re.split(r"[-/]", match.group(0))
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _window_label(start: date | None, end: date | None) -> str:
    if start and end and start != end:
        return f"{start.isoformat()}..{end.isoformat()}"
    if start:
        return start.isoformat()
    if end:
        return f"..{end.isoformat()}"
    return ""


def _dedupe(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
