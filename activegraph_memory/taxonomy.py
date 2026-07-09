"""Lightweight deterministic taxonomy helpers for compiled memory events."""

from __future__ import annotations

import re


_WORD_RE = re.compile(r"[a-z0-9]+")
_NEGATED_EVENT_RE = re.compile(
    r"\b(?:did not|didn't|do not|don't|never|not|no)\s+"
    r"(?:buy|bought|purchase|purchased|order|ordered|pay|paid|spend|spent|"
    r"acquire|acquired|get|got|receive|received|attend|attended|visit|visited|"
    r"go|went|schedule|scheduled|book|booked|donate|donated)\b",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "art": ("art", "gallery", "painting", "museum", "exhibit", "mural", "studio"),
    "bike": ("bike", "bicycle", "cycling", "helmet", "pedal", "chain", "brake", "tire", "lights"),
    "book": ("book", "novel", "paperback", "chapter", "library", "read"),
    "charity": ("charity", "donation", "donated", "fundraiser", "raised"),
    "clothing": ("clothing", "shirt", "jacket", "blazer", "boots", "dress", "coat", "zara"),
    "event": ("event", "festival", "concert", "wedding", "workshop", "conference", "class"),
    "expense": (
        "$",
        "cost",
        "costs",
        "expense",
        "expenses",
        "paid",
        "spent",
        "bought",
        "purchased",
        "ordered",
    ),
    "family": ("family", "mom", "mother", "dad", "father", "sister", "brother", "baby", "wedding"),
    "food": ("food", "coffee", "tea", "cocktail", "dinner", "lunch", "restaurant", "citrus"),
    "health": ("doctor", "dentist", "medication", "therapy", "health", "appointment"),
    "home": ("home", "apartment", "kitchen", "room", "furniture", "smoker", "sofa"),
    "music": ("music", "album", "song", "concert", "playlist", "guitar", "piano"),
    "plant": (
        "plant",
        "plants",
        "succulent",
        "monstera",
        "fern",
        "orchid",
        "cactus",
        "peace lily",
        "snake plant",
        "pothos",
    ),
    "project": ("project", "repo", "repository", "launch", "prototype", "roadmap", "milestone"),
    "travel": ("travel", "trip", "flight", "hotel", "airbnb", "museum", "train", "airport"),
    "vehicle": ("car", "bike", "bicycle", "scooter", "vehicle", "tire"),
}

_PREDICATE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cancel", ("cancel", "cancelled", "canceled", "dropped", "skipped")),
    ("repair", ("repair", "repaired", "fixed", "serviced", "replaced", "maintenance")),
    ("purchase", ("buy", "bought", "purchase", "purchased", "ordered", "paid", "spend", "spent", "cost")),
    ("acquire", ("acquire", "acquired", "got", "received", "picked up", "adopted")),
    ("attend", ("attend", "attended", "went to", "joined", "participated")),
    ("visit", ("visit", "visited", "stopped by", "went to", "toured")),
    ("recommend", ("recommend", "recommended", "suggested", "advised")),
    ("decide", ("decide", "decided", "agreed", "approved", "chose", "selected")),
    ("prefer", ("prefer", "prefers", "favorite", "likes")),
    ("schedule", ("schedule", "scheduled", "booked", "planned", "appointment")),
    ("donate", ("donate", "donated", "gave", "raised", "fundraiser")),
)

_PREDICATE_GROUPS: tuple[set[str], ...] = (
    {"purchase", "acquire"},
    {"attend", "visit"},
    {"repair", "purchase"},
    {"donate", "expense"},
)


def normalize_token(token: str) -> str:
    """Normalize a token for rough lexical matching."""

    token = token.lower()
    for ending in ("ing", "ed", "es", "s"):
        if len(token) > len(ending) + 3 and token.endswith(ending):
            return token[: -len(ending)]
    return token


def significant_tokens(text: str) -> set[str]:
    """Return normalized tokens that are useful for broad matching."""

    return {
        normalize_token(match.group(0))
        for match in _WORD_RE.finditer(text.lower())
        if len(match.group(0)) >= 3
    }


def infer_category_ids(text: str) -> tuple[str, ...]:
    """Infer coarse category ids from text with deterministic keyword rules."""

    lower = f" {text.lower()} "
    tokens = significant_tokens(text)
    out: list[str] = []
    for category_id, keywords in _CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if _contains_keyword(lower, tokens, keyword):
                out.append(category_id)
                break
    return tuple(out)


def category_label(category_id: str) -> str:
    """Human label for a category id."""

    return category_id.replace("_", " ")


def infer_predicate(text: str) -> str:
    """Infer a coarse event predicate from text."""

    lower = f" {text.lower()} "
    tokens = significant_tokens(text)
    for predicate, patterns in _PREDICATE_PATTERNS:
        if any(_contains_keyword(lower, tokens, pattern) for pattern in patterns):
            return predicate
    return "state"


def infer_polarity(text: str) -> str:
    """Infer whether an event-like claim is affirmative or negated."""

    if _NEGATED_EVENT_RE.search(text):
        return "negative"
    return "affirmative"


def predicates_compatible(query_predicate: str, event_predicate: str) -> bool:
    """Return True when two coarse predicates should match for retrieval."""

    if query_predicate in {"state", "unknown"}:
        return True
    if query_predicate == event_predicate:
        return True
    for group in _PREDICATE_GROUPS:
        if query_predicate in group and event_predicate in group:
            return True
    return False


def _contains_keyword(lower_text: str, tokens: set[str], keyword: str) -> bool:
    if " " in keyword:
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in keyword.lower().split()) + r"\b"
        return bool(re.search(pattern, lower_text))
    if not keyword.isalnum():
        return keyword in lower_text
    return normalize_token(keyword) in tokens
