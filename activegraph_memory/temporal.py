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
    if unit.startswith("day"):
        delta = timedelta(days=count)
    elif unit.startswith("week"):
        delta = timedelta(weeks=count)
    elif unit.startswith("month"):
        delta = timedelta(days=30 * count)
    else:
        delta = timedelta(days=365 * count)

    start = anchor - delta
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
    if unit.startswith("day"):
        delta = timedelta(days=count)
    elif unit.startswith("week"):
        delta = timedelta(weeks=count)
    elif unit.startswith("month"):
        delta = timedelta(days=30 * count)
    else:
        delta = timedelta(days=365 * count)
    target = anchor - delta
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
    return refs
