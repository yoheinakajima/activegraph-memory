"""Small deterministic temporal helpers."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .object_types import TemporalRef


_DURATION_RE = re.compile(
    r"\bfor\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)\s+(?P<unit>day|days|week|weeks|month|months|year|years)"
    r"\s+(?:now|so far|already)?",
    re.IGNORECASE,
)
_PAST_DURATION_RE = re.compile(
    r"\b(?:for\s+)?(?:the\s+)?(?:past|last)\s+(?P<count>\d+|one|two|three|"
    r"four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?P<unit>day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)
_RELATIVE_AGO_RE = re.compile(
    r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)\s+(?P<unit>day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)
_EXPLICIT_DATE_RE = re.compile(
    r"\b(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b"
)
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?P<month>January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|"
    r"July|Jul|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)"
    r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>20\d{2}))?\b",
    re.IGNORECASE,
)
_RELATIVE_DAY_RE = re.compile(r"\b(today|yesterday|day before yesterday|tomorrow)\b", re.IGNORECASE)
_RELATIVE_WEEKDAY_RE = re.compile(
    r"\b(?P<direction>last|previous|this|next)\s+"
    r"(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
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
}
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_anchor(anchor: str | None) -> date:
    if not anchor:
        return datetime.now().date()
    match = _EXPLICIT_DATE_RE.search(anchor)
    if match:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    normalized = anchor.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return date.fromisoformat(anchor[:10].replace("/", "-"))


def detect_relative_duration(text: str) -> bool:
    """Return True when text contains a simple 'for N units now' phrase."""

    return bool(_DURATION_RE.search(text) or _PAST_DURATION_RE.search(text))


def _delta_for_count_unit(count: int, unit: str) -> timedelta | None:
    try:
        if unit.startswith("day"):
            return timedelta(days=count)
        if unit.startswith("week"):
            return timedelta(weeks=count)
        if unit.startswith("month"):
            return timedelta(days=30 * count)
        return timedelta(days=365 * count)
    except OverflowError:
        return None


def _unresolved_out_of_range(
    text: str,
    *,
    anchor: date,
    count: int,
    unit: str,
) -> TemporalRef:
    return TemporalRef(
        text=text,
        anchor_time=anchor.isoformat(),
        resolution_method="unresolved",
        confidence=0.0,
        metadata={
            "count": count,
            "unit": unit,
            "reason": "relative_date_out_of_range",
        },
    )


def resolve_relative_duration(text: str, *, anchor_time: str | None = None) -> TemporalRef:
    """Resolve a simple duration phrase into an approximate start date."""

    match = _DURATION_RE.search(text) or _PAST_DURATION_RE.search(text)
    if not match:
        return TemporalRef(
            text=text,
            anchor_time=anchor_time,
            resolution_method="unresolved",
            confidence=0.0,
        )

    raw_count = match.group("count").lower()
    count = int(raw_count) if raw_count.isdigit() else _WORD_NUMBERS.get(raw_count, 0)
    unit = match.group("unit").lower()
    anchor = _parse_anchor(anchor_time)
    delta = _delta_for_count_unit(count, unit)
    if delta is None:
        return _unresolved_out_of_range(
            match.group(0), anchor=anchor, count=count, unit=unit
        )

    try:
        start = anchor - delta
    except OverflowError:
        return _unresolved_out_of_range(
            match.group(0), anchor=anchor, count=count, unit=unit
        )
    return TemporalRef(
        text=match.group(0),
        resolved_start=start.isoformat(),
        resolved_end=anchor.isoformat(),
        anchor_time=anchor.isoformat(),
        resolution_method="duration_start",
        confidence=0.75 if unit.startswith(("month", "year")) else 0.9,
        metadata={
            "count": count,
            "unit": unit,
            "approximate": unit.startswith(("month", "year")),
        },
    )


def resolve_relative_ago(text: str, *, anchor_time: str | None = None) -> TemporalRef:
    """Resolve a simple ``N days/weeks/months/years ago`` phrase."""

    match = _RELATIVE_AGO_RE.search(text)
    if not match:
        return TemporalRef(
            text=text,
            anchor_time=anchor_time,
            resolution_method="unresolved",
            confidence=0.0,
        )

    raw_count = match.group("count").lower()
    count = int(raw_count) if raw_count.isdigit() else _WORD_NUMBERS.get(raw_count, 0)
    unit = match.group("unit").lower()
    anchor = _parse_anchor(anchor_time)
    delta = _delta_for_count_unit(count, unit)
    if delta is None:
        return _unresolved_out_of_range(
            match.group(0), anchor=anchor, count=count, unit=unit
        )
    try:
        target = anchor - delta
    except OverflowError:
        return _unresolved_out_of_range(
            match.group(0), anchor=anchor, count=count, unit=unit
        )
    return TemporalRef(
        text=match.group(0),
        resolved_start=target.isoformat(),
        resolved_end=target.isoformat(),
        anchor_time=anchor.isoformat(),
        resolution_method="relative_to_query",
        confidence=0.75 if unit.startswith(("month", "year")) else 0.9,
        metadata={
            "count": count,
            "unit": unit,
            "approximate": unit.startswith(("month", "year")),
        },
    )


def extract_temporal_refs(text: str, *, anchor_time: str | None = None) -> list[TemporalRef]:
    """Extract small deterministic temporal references from text."""

    refs: list[TemporalRef] = []
    duration = resolve_relative_duration(text, anchor_time=anchor_time)
    if duration.resolution_method != "unresolved":
        refs.append(duration)
    ago = resolve_relative_ago(text, anchor_time=anchor_time)
    if ago.resolution_method != "unresolved":
        refs.append(ago)
    for match in _EXPLICIT_DATE_RE.finditer(text):
        dt = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        refs.append(
            TemporalRef(
                text=match.group(0),
                resolved_start=dt.isoformat(),
                resolved_end=dt.isoformat(),
                anchor_time=anchor_time,
                resolution_method="explicit",
                confidence=0.95,
            )
        )
    anchor = _parse_anchor(anchor_time)
    for match in _MONTH_NAME_DATE_RE.finditer(text):
        month = _MONTHS.get(match.group("month").lower())
        if not month:
            continue
        year = int(match.group("year") or anchor.year)
        try:
            dt = date(year, month, int(match.group("day")))
        except ValueError:
            continue
        refs.append(
            TemporalRef(
                text=match.group(0),
                resolved_start=dt.isoformat(),
                resolved_end=dt.isoformat(),
                anchor_time=anchor_time,
                resolution_method="explicit",
                confidence=0.9 if match.group("year") else 0.82,
                metadata={"year_inferred_from_anchor": not bool(match.group("year"))},
            )
        )

    anchor = _parse_anchor(anchor_time)
    for match in _RELATIVE_DAY_RE.finditer(text):
        label = match.group(1).lower()
        delta = {"day before yesterday": -2, "yesterday": -1, "today": 0, "tomorrow": 1}[label]
        resolved = anchor + timedelta(days=delta)
        refs.append(
            TemporalRef(
                text=match.group(0),
                resolved_start=resolved.isoformat(),
                resolved_end=resolved.isoformat(),
                anchor_time=anchor.isoformat(),
                resolution_method="relative_to_source",
                confidence=0.92,
                metadata={"relative_day": label},
            )
        )
    for match in _RELATIVE_WEEKDAY_RE.finditer(text):
        direction = match.group("direction").lower()
        weekday = match.group("weekday").lower()
        if weekday == "weekend":
            start, end = _resolve_weekend(anchor, direction)
        else:
            start = _resolve_weekday(anchor, _WEEKDAYS[weekday], direction)
            end = start
        refs.append(
            TemporalRef(
                text=match.group(0),
                resolved_start=start.isoformat(),
                resolved_end=end.isoformat(),
                anchor_time=anchor.isoformat(),
                resolution_method="relative_to_source",
                confidence=0.84 if direction == "this" else 0.9,
                metadata={"direction": direction, "weekday": weekday},
            )
        )
    return refs


def _resolve_weekday(anchor: date, weekday: int, direction: str) -> date:
    if direction in {"last", "previous"}:
        days = (anchor.weekday() - weekday) % 7 or 7
        return anchor - timedelta(days=days)
    if direction == "next":
        days = (weekday - anchor.weekday()) % 7 or 7
        return anchor + timedelta(days=days)
    return anchor + timedelta(days=weekday - anchor.weekday())


def _resolve_weekend(anchor: date, direction: str) -> tuple[date, date]:
    this_saturday = anchor + timedelta(days=5 - anchor.weekday())
    if direction in {"last", "previous"}:
        this_saturday -= timedelta(days=7)
    elif direction == "next":
        this_saturday += timedelta(days=7)
    return this_saturday, this_saturday + timedelta(days=1)
