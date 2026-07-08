"""Small deterministic temporal helpers."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .object_types import TemporalRef


_DURATION_RE = re.compile(
    r"\bfor\s+(?P<count>\d+)\s+(?P<unit>day|days|week|weeks|month|months|year|years)\s+(?:now|so far|already)?",
    re.IGNORECASE,
)


def _parse_anchor(anchor: str | None) -> date:
    if not anchor:
        return datetime.now().date()
    normalized = anchor.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return date.fromisoformat(anchor[:10])


def detect_relative_duration(text: str) -> bool:
    """Return True when text contains a simple 'for N units now' phrase."""

    return bool(_DURATION_RE.search(text))


def resolve_relative_duration(text: str, *, anchor_time: str | None = None) -> TemporalRef:
    """Resolve a simple duration phrase into an approximate start date."""

    match = _DURATION_RE.search(text)
    if not match:
        return TemporalRef(
            text=text,
            anchor_time=anchor_time,
            resolution_method="unresolved",
            confidence=0.0,
        )

    count = int(match.group("count"))
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
