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
    "art": ("art", "artwork", "gallery", "painting", "museum", "exhibit", "mural", "studio"),
    "bike": ("bike", "bicycle", "cycling", "helmet", "pedal", "chain", "brake", "tire", "lights"),
    "book": ("book", "novel", "paperback", "chapter", "library", "read"),
    "charity": (
        "charity",
        "donation",
        "donated",
        "fundraiser",
        "fundraising",
        "sponsor",
        "sponsored",
        "pledge",
        "pledged",
        "benefit",
        "nonprofit",
        "animal shelter",
        "cancer research",
    ),
    "clothing": (
        "clothing",
        "shirt",
        "top",
        "tops",
        "jacket",
        "blazer",
        "boots",
        "dress",
        "coat",
        "gown",
        "heels",
        "shoes",
        "handbag",
        "bag",
        "designer",
        "fashion",
        "gucci",
        "high-end",
        "jimmy choo",
        "luxury",
        "tk maxx",
        "outlet",
        "zara",
        "h&m",
        "hm",
    ),
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
    "family": (
        "family",
        "mom",
        "mother",
        "dad",
        "father",
        "sister",
        "brother",
        "aunt",
        "cousin",
        "friend",
        "friends",
        "son",
        "daughter",
        "child",
        "children",
        "baby",
        "babies",
        "twin",
        "twins",
        "wedding",
    ),
    "food": ("food", "coffee", "tea", "cocktail", "dinner", "lunch", "restaurant", "citrus"),
    "gaming": (
        "game",
        "games",
        "gaming",
        "video game",
        "assassin's creed",
        "odyssey",
        "witcher",
        "dragon age",
        "hyper light drifter",
        "celeste",
        "last of us",
        "god of war",
        "horizon zero dawn",
        "red dead redemption",
        "ori and the blind forest",
        "gris",
        "spelunky",
        "cuphead",
        "switch",
        "ps5",
        "xbox",
    ),
    "grocery": (
        "grocery",
        "groceries",
        "supermarket",
        "publix",
        "walmart",
        "instacart",
        "trader joe",
        "trader joe's",
        "thrive market",
        "whole foods",
        "costco",
        "aldi",
    ),
    "health": (
        "doctor",
        "dr.",
        "dr ",
        "physician",
        "surgeon",
        "dentist",
        "medication",
        "therapy",
        "health",
        "appointment",
        "follow-up",
        "checkup",
    ),
    "home": ("home", "apartment", "kitchen", "room", "furniture", "appliance", "smoker", "sofa"),
    "luxury": ("luxury", "high-end", "designer", "gucci", "prada", "chanel", "versace"),
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
    "project": (
        "project",
        "repo",
        "repository",
        "launch",
        "prototype",
        "roadmap",
        "milestone",
        "contract",
        "client",
        "deal",
    ),
    "travel": ("travel", "trip", "flight", "hotel", "airbnb", "museum", "train", "airport"),
    "vehicle": (
        "car",
        "bike",
        "bicycle",
        "scooter",
        "vehicle",
        "tire",
        "truck",
        "pickup",
        "ford",
        "mustang",
        "f-150",
    ),
}

_CATEGORY_PARENTS = {
    "luxury": "clothing",
}

_QUERY_CONCEPT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "milestone": (
        "accomplishment",
        "achievement",
        "achieved",
        "award",
        "client",
        "completed",
        "contract",
        "deal",
        "finished",
        "funding",
        "launched",
        "opened",
        "partnership",
        "promotion",
        "reached",
        "sale",
        "signed",
        "started",
    ),
}

_PREDICATE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("birth", ("born", "birth", "gave birth", "welcomed")),
    ("cancel", ("cancel", "cancelled", "canceled", "dropped", "skipped")),
    ("graduate", ("graduate", "graduated", "graduation")),
    ("start", ("start", "started", "begin", "began")),
    ("finish", ("finish", "finished", "complete", "completed")),
    ("move", ("move", "moved", "relocated")),
    ("repair", ("repair", "repaired", "fixed", "serviced", "replaced", "maintenance")),
    ("install", ("install", "installed", "fitted", "mounted")),
    ("purchase", ("buy", "bought", "purchase", "purchased", "ordered", "paid", "spend", "spent", "cost")),
    ("acquire", ("acquire", "acquired", "got", "received", "picked up", "adopted", "arrive", "arrived")),
    ("attend", ("attend", "attended", "went to", "joined", "participated")),
    ("visit", ("visit", "visited", "stopped by", "went to", "toured")),
    ("recommend", ("recommend", "recommended", "suggested", "advised")),
    ("decide", ("decide", "decided", "agreed", "approved", "chose", "selected")),
    ("prefer", ("prefer", "prefers", "favorite", "likes")),
    ("schedule", ("schedule", "scheduled", "booked", "planned", "appointment", "follow-up", "checkup")),
    ("donate", ("donate", "donated", "gave", "raise", "raised", "fundraiser", "sponsor", "sponsored", "pledge", "pledged")),
)

_PREDICATE_GROUPS: tuple[set[str], ...] = (
    {"purchase", "acquire"},
    {"start", "acquire"},
    {"finish", "graduate"},
    {"attend", "visit"},
    {"repair", "purchase", "install"},
    {"donate", "expense"},
)


def normalize_token(token: str) -> str:
    """Normalize a token for rough lexical matching."""

    token = token.lower()
    if token in {"raise", "raises", "raised", "raising"}:
        return "raise"
    if token in {"organize", "organizes", "organized", "organizing"}:
        return "organize"
    if token in {"wedding", "weddings"}:
        return "wedding"
    for ending in ("ing", "ed", "es"):
        if len(token) > len(ending) + 3 and token.endswith(ending):
            return token[: -len(ending)]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
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
    if "charity" not in out and _looks_like_fundraising(lower):
        out.append("charity")
    return tuple(out)


def category_label(category_id: str) -> str:
    """Human label for a category id."""

    return category_id.replace("_", " ")


def category_mentions(text: str, category_id: str) -> tuple[str, ...]:
    """Return distinct taxonomy terms from one category mentioned in text."""

    lower = text.lower()
    tokens = significant_tokens(text)
    matches = [
        keyword
        for keyword in _CATEGORY_KEYWORDS.get(category_id, ())
        if _contains_keyword(lower, tokens, keyword)
    ]
    label = category_label(category_id)
    specific = [match for match in matches if normalize_token(match) != normalize_token(label)]
    values = specific or matches
    return tuple(dict.fromkeys(values))


def most_specific_categories(category_ids) -> set[str]:
    """Remove a broad parent when a matched child category is available."""

    values = set(category_ids)
    return {
        category_id
        for category_id in values
        if not any(_CATEGORY_PARENTS.get(child) == category_id for child in values)
    }


def expanded_query_variants(text: str) -> tuple[str, ...]:
    """Return deterministic semantic bridge queries for broad concepts."""

    tokens = significant_tokens(text)
    expansions = [
        expansion
        for concept, values in _QUERY_CONCEPT_EXPANSIONS.items()
        if normalize_token(concept) in tokens
        for expansion in values
    ]
    if not expansions:
        return ()
    return (f"{text} {' '.join(expansions)}",)


def infer_predicate(text: str) -> str:
    """Infer a coarse event predicate from text."""

    lower = f" {text.lower()} "
    tokens = significant_tokens(text)
    if re.search(r"\b(?:got|came|went|returned)\s+back\s+from\b", lower):
        return "visit"
    if _looks_like_birth_event(lower):
        return "birth"
    for predicate, patterns in _PREDICATE_PATTERNS:
        if predicate == "birth" and re.search(r"\bborn\s+and\s+raised\b", lower):
            continue
        if predicate == "donate" and re.search(r"\bborn\s+and\s+raised\b", lower):
            continue
        if predicate == "start" and re.search(r"\bstart of (?:the )?(?:year|month|week|day)\b", lower):
            continue
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
    if keyword == "ordered":
        return bool(re.search(r"\bordered\b", lower_text))
    return normalize_token(keyword) in tokens


def _looks_like_fundraising(lower_text: str) -> bool:
    return bool(
        re.search(r"\brais(?:e|ed|ing)\b[^.]{0,80}\b(?:money|funds?|donations?|sponsors?)\b", lower_text)
        or re.search(r"\brais(?:e|ed|ing)\b[^.]{0,80}\$", lower_text)
        or re.search(
            r"\brais(?:e|ed|ing)\b[^.]{0,80}\b(?:cancer|hospital|shelter|research|nonprofit|foundation|school)\b",
            lower_text,
        )
    )


def _looks_like_birth_event(lower_text: str) -> bool:
    if re.search(r"\bborn\s+and\s+raised\b", lower_text):
        return False
    return bool(
        re.search(r"\b(?:was|were)\s+born\b", lower_text)
        or re.search(r"\bgave\s+birth\b", lower_text)
        or re.search(r"\bwelcomed\b[^.]{0,80}\b(?:baby|child|son|daughter|twins?)\b", lower_text)
        or re.search(r"\bhad\b[^.]{0,60}\b(?:baby|child|son|daughter|twins?)\b", lower_text)
        or re.search(r"\bnew\s+twin\s+girls?\b", lower_text)
    )
