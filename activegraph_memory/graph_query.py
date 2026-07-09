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
    predicates_compatible,
    significant_tokens,
)
from .temporal import extract_temporal_refs


_MONEY_QUERY_RE = re.compile(
    r"\b(money|spend|spent|cost|costs|expense|expenses|paid|price|total)\b|\$",
    re.IGNORECASE,
)
_COUNT_QUERY_RE = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
_SUM_QUERY_RE = re.compile(r"\b(how much|total|sum|spent|cost|expenses?)\b|\$", re.IGNORECASE)
_TIMELINE_QUERY_RE = re.compile(
    r"\b(first|earliest|latest|order|ordered|timeline|history|chronological|when)\b",
    re.IGNORECASE,
)
_AUXILIARY_CATEGORIES = {"expense"}
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

    def render(self, *, max_rows: int = 24) -> str:
        """Render a compact, source-oriented block for answer synthesis."""

        lines = [f"[graph-query: {self.operation}]"]
        if self.answer_hint:
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
    if query_type not in {"aggregate", "temporal"} and not _looks_like_graph_query(query):
        return None

    operation = _infer_operation(query, query_type=query_type)
    inferred_categories = infer_category_ids(query)
    query_categories = _required_query_categories(inferred_categories)
    query_predicate = infer_predicate(query)
    time_window = infer_query_time_window(query, anchor_time=anchor_time)
    matched = _matching_events(
        index,
        query,
        query_categories=query_categories,
        query_predicate=query_predicate,
        time_window=time_window,
    )
    matched = _dedupe_events(matched)

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
    return bool(_COUNT_QUERY_RE.search(query) or _SUM_QUERY_RE.search(query) or _TIMELINE_QUERY_RE.search(query))


def _infer_operation(query: str, *, query_type: str) -> str:
    if _COUNT_QUERY_RE.search(query):
        return "count"
    if _SUM_QUERY_RE.search(query):
        return "sum"
    if query_type == "aggregate":
        return "count"
    return "timeline"


def _matching_events(
    index: MemoryIndex,
    query: str,
    *,
    query_categories: tuple[str, ...],
    query_predicate: str,
    time_window: QueryTimeWindow,
) -> list[MemoryEventRecord]:
    q_tokens = significant_tokens(query)
    out: list[MemoryEventRecord] = []
    for event in index.events:
        if event.metadata.get("claim_status") == "superseded":
            continue
        if event.metadata.get("polarity") == "negative":
            continue
        if query_categories and not all(category in event.category_ids for category in query_categories):
            continue
        if query_predicate != "state" and not predicates_compatible(query_predicate, event.predicate):
            continue
        if not time_window.contains(event):
            continue
        if not query_categories and query_predicate == "state":
            event_text = _event_match_text(index, event)
            if not (q_tokens & significant_tokens(event_text)):
                continue
        out.append(event)
    return sorted(out, key=lambda event: (event.event_start or event.observed_at or "", event.sort_key, event.event_id))


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
    quantity_values = _count_quantity_values(events, query=query, query_categories=query_categories)
    if quantity_values:
        count_value = sum(quantity_values)
        count_label = _format_number(count_value)
        answer = f"Computed count: {count_label}"
        metadata = {"count_method": "quantity_sum", "quantity_values": quantity_values}
    else:
        count_value = float(len(events))
        answer = f"Computed count: {int(count_value)}"
        metadata = {"count_method": "event_count"}
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
    total = 0.0
    values: list[float] = []
    for event in events:
        for quantity in event.quantity_claims:
            normalized_unit = _normalize_unit(quantity.unit)
            if unit and normalized_unit != unit:
                continue
            if quantity.value is None:
                continue
            total += float(quantity.value)
            values.append(float(quantity.value))
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
        metadata={"sum_values": values},
    )


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


def _required_query_categories(category_ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(category_ids) <= 1:
        return category_ids
    required = tuple(
        category_id
        for category_id in category_ids
        if category_id not in _AUXILIARY_CATEGORIES
    )
    return required or category_ids


def _count_quantity_values(
    events: list[MemoryEventRecord],
    *,
    query: str,
    query_categories: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    query_tokens = significant_tokens(query)
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


def _category_count_units(category_ids: tuple[str, ...]) -> set[str]:
    units: set[str] = set()
    for category_id in category_ids:
        units.add(category_id)
        if category_id == "plant":
            units.update({"plant", "plants"})
        if category_id == "event":
            units.update({"event", "wedding", "museum", "class", "workshop"})
        if category_id == "family":
            units.update({"baby", "babies", "wedding"})
    return {_normalize_unit(unit) or unit for unit in units}


def _requested_unit(query: str) -> str | None:
    tokens = significant_tokens(query)
    for token in tokens:
        unit = _normalize_unit(token)
        if unit in {"usd", "day", "week", "month", "year", "hour", "minute", "page", "plant"}:
            return unit
    return None


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
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


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
