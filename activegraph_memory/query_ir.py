"""Multi-operator query analysis for memory retrieval and execution."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .compiler import claim_tokens
from .graph_query import infer_query_time_window
from .object_types import MemoryQuery
from .planner import infer_query_type


MemoryOperator = Literal[
    "lookup",
    "count",
    "sum",
    "max",
    "order",
    "date_delta",
    "latest",
    "current",
    "previous",
    "recommend",
    "negative_existence",
    "ordinal",
]


class QueryAnalysis(BaseModel):
    """Executable, provider-neutral interpretation of a memory query."""

    model_config = ConfigDict(extra="forbid")

    query: str
    query_type: str
    operators: list[MemoryOperator] = Field(default_factory=list)
    entity_terms: list[str] = Field(default_factory=list)
    category_terms: list[str] = Field(default_factory=list)
    operands: list[str] = Field(default_factory=list)
    source_roles: list[str] = Field(default_factory=lambda: ["user", "assistant"])
    time_start: str | None = None
    time_end: str | None = None
    time_label: str = "unbounded"
    completed_only: bool = False
    expected_answer_type: str = "text"
    proof_requirements: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list)
    deterministic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_operator(self) -> MemoryOperator:
        return self.operators[0] if self.operators else "lookup"

    @property
    def requires_exhaustive_coverage(self) -> bool:
        return any(operator in {"count", "sum", "max", "negative_existence"} for operator in self.operators)

    @property
    def requires_reasoning(self) -> bool:
        return bool(
            len(self.operators) > 1
            or self.operands
            or any(operator in {"order", "date_delta", "recommend"} for operator in self.operators)
        )


_GENERIC_TERMS = {
    "answer",
    "amount",
    "current",
    "currently",
    "first",
    "how",
    "latest",
    "many",
    "most",
    "much",
    "previous",
    "previously",
    "recent",
    "recommend",
    "second",
    "suggest",
    "third",
    "time",
    "times",
    "total",
    "what",
    "when",
    "where",
    "which",
    "who",
}
_COUNT_RE = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
_SUM_RE = re.compile(r"\b(how much|total money|total cost|sum|spent|expenses?)\b", re.IGNORECASE)
_MAX_RE = re.compile(r"\b(most|highest|largest|maximum|max)\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\b(order|earliest to latest|latest to earliest|which .+ first|what .+ first)\b", re.IGNORECASE)
_DELTA_RE = re.compile(r"\b(how (?:many|much) (?:days?|weeks?|months?|years?).*(?:between|passed|elapsed)|how long.*between)\b", re.IGNORECASE)
_LATEST_RE = re.compile(r"\b(latest|most recent|newest|last one)\b", re.IGNORECASE)
_CURRENT_RE = re.compile(r"\b(current|currently|as of now|now keep|now have)\b", re.IGNORECASE)
_PREVIOUS_RE = re.compile(r"\b(previous|previously|before (?:i|we|the)|prior)\b", re.IGNORECASE)
_RECOMMEND_RE = re.compile(r"\b(recommend|suggest|advice|tips?|what should i)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\b(never|did not|didn't|no record|have i not|hasn't|has not)\b", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b(?P<value>\d{1,3})(?:st|nd|rd|th)\b", re.IGNORECASE)
_ASSISTANT_SOURCE_RE = re.compile(
    r"\b(?:you|assistant)\b.*\b(?:say|said|mention|mentioned|tell|told|provide|provided|"
    r"list|listed|recommend|recommended|suggest|suggested|give|gave|name|named|call|called|"
    r"assign|assigned|write|wrote|send|sent|show|showed|share|shared|compute|computed|"
    r"calculate|calculated)\b|"
    r"\b(?:say|said|mention|mentioned|tell|told|provide|provided|list|listed|recommend|"
    r"recommended|suggest|suggested|give|gave|name|named|call|called|assign|assigned|write|"
    r"wrote|send|sent|show|showed|share|shared)\b.*\b(?:you|assistant)\b|"
    r"\blist (?:you|the assistant) provided\b|"
    r"\bremind me\b.*\b(?:our|we|you|your|previous|earlier|prior|last time)\b|"
    r"\b(?:our|previous|earlier|prior)\b.*\b(?:chat|conversation|discussion)\b.*\bremind me\b",
    re.IGNORECASE,
)
_USER_SOURCE_RE = re.compile(
    r"\b(?:i|we)\s+(?:(?:am|are|was|were|have been|had been)\s+)?(?:currently\s+)?"
    r"(?:attended|bought|purchased|spent|used|visited|went|flew|finished|completed|fixed|"
    r"serviced|received|redeemed|downloaded|worked|working|signed|started|launched|said|"
    r"mentioned|told|met|spoke|talked|discussed|preferred|liked|disliked|owned|have|had)\b|"
    r"\b(?:did|do|have|had|am|was|were)\s+(?:i|we)\s+(?:attend|buy|purchase|spend|use|"
    r"visit|go|finish|complete|fix|service|receive|redeem|download|work|sign|start|launch|"
    r"say|mention|tell|meet|speak|talk|discuss|prefer|like|dislike|own|have)\b",
    re.IGNORECASE,
)
_ACTUAL_RE = re.compile(
    r"\b(attended|bought|purchased|spent|used|visited|went|flew|finished|completed|"
    r"fixed|serviced|received|redeemed|downloaded|worked on|added)\b",
    re.IGNORECASE,
)
_COMPARISON_TAIL_RE = re.compile(r"\b(?:between|first,?|earlier,?)\s+(?P<tail>.+?)(?:\?|$)", re.IGNORECASE)


def analyze_query(
    query: MemoryQuery | str,
    *,
    question_date: str | None = None,
) -> QueryAnalysis:
    """Build a deterministic multi-operator query representation."""

    memory_query = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query)
    text = memory_query.query.strip()
    anchor = memory_query.time_anchor or question_date
    query_type = memory_query.query_type
    if query_type == "unknown":
        query_type = infer_query_type(text)

    operators: list[MemoryOperator] = []
    for pattern, operator in (
        (_ORDINAL_RE, "ordinal"),
        (_DELTA_RE, "date_delta"),
        (_ORDER_RE, "order"),
        (_COUNT_RE, "count"),
        (_SUM_RE, "sum"),
        (_MAX_RE, "max"),
        (_CURRENT_RE, "current"),
        (_LATEST_RE, "latest"),
        (_PREVIOUS_RE, "previous"),
        (_RECOMMEND_RE, "recommend"),
        (_NEGATIVE_RE, "negative_existence"),
    ):
        if pattern.search(text) and operator not in operators:
            operators.append(operator)  # type: ignore[arg-type]
    if "date_delta" in operators:
        operators = [operator for operator in operators if operator not in {"count", "sum"}]
    if "order" in operators:
        operators = [operator for operator in operators if operator not in {"latest", "current", "previous"}]
    if not operators:
        operators.append("lookup")

    tokens = sorted(
        token
        for token in claim_tokens(text)
        if token not in _GENERIC_TERMS and not token.isdigit()
    )
    operands = _comparison_operands(text) if any(op in {"order", "date_delta"} for op in operators) else []
    assistant_source = bool(_ASSISTANT_SOURCE_RE.search(text))
    user_source = bool(_USER_SOURCE_RE.search(text))
    if "ordinal" in operators:
        source_roles = ["assistant"]
    elif assistant_source and not user_source:
        source_roles = ["assistant"]
    elif user_source and not assistant_source:
        source_roles = ["user"]
    else:
        source_roles = ["user", "assistant"]
    time_window = infer_query_time_window(text, anchor_time=anchor)
    expected = _expected_answer_type(operators)
    requirements = _proof_requirements(operators, operands)
    confidence = 0.92
    if query_type == "semantic_lookup" and operators == ["lookup"]:
        confidence = 0.78
    if len(operators) > 2:
        confidence -= 0.08
    if any(op in {"order", "date_delta"} for op in operators) and len(operands) < 2:
        confidence -= 0.22

    variants = [text]
    variants.extend(operand for operand in operands if operand.lower() not in text.lower())
    return QueryAnalysis(
        query=text,
        query_type=str(query_type),
        operators=operators,
        entity_terms=tokens,
        category_terms=tokens,
        operands=operands,
        source_roles=source_roles,
        time_start=time_window.start.isoformat() if time_window.start else None,
        time_end=time_window.end.isoformat() if time_window.end else None,
        time_label=time_window.label,
        completed_only=bool(_ACTUAL_RE.search(text) or any(op in {"count", "sum", "order", "date_delta"} for op in operators)),
        expected_answer_type=expected,
        proof_requirements=requirements,
        query_variants=_dedupe(variants),
        deterministic_confidence=max(0.0, min(1.0, confidence)),
        metadata={
            "time_anchor": anchor,
            "ordinal": int(_ORDINAL_RE.search(text).group("value")) if _ORDINAL_RE.search(text) else None,
            "source_role_signals": {
                "assistant": assistant_source,
                "user": user_source,
                "ambiguous": assistant_source == user_source,
            },
        },
    )


def _expected_answer_type(operators: list[MemoryOperator]) -> str:
    if any(op in {"count", "ordinal"} for op in operators):
        return "integer_or_item"
    if "sum" in operators:
        return "quantity"
    if "date_delta" in operators:
        return "duration"
    if "order" in operators:
        return "ordered_list"
    if "recommend" in operators:
        return "personalized_recommendation"
    return "text"


def _proof_requirements(operators: list[MemoryOperator], operands: list[str]) -> list[str]:
    requirements = ["source_provenance", "entity_compatibility"]
    if any(op in {"count", "sum", "max"} for op in operators):
        requirements.extend(
            [
                "bounded_candidate_set",
                "canonical_event_deduplication",
                "source_coverage",
            ]
        )
    if any(op in {"current", "latest", "previous"} for op in operators):
        requirements.extend(["state_history", "supersession_check"])
    if any(op in {"order", "date_delta"} for op in operators):
        requirements.extend(["event_time_resolution", "operand_coverage"])
    if "recommend" in operators:
        requirements.extend(["preference_scope", "constraint_coverage"])
    if "ordinal" in operators:
        requirements.extend(["list_identity", "ordinal_position"])
    if operands:
        requirements.append("all_operands_found")
    return _dedupe(requirements)


def _comparison_operands(query: str) -> list[str]:
    quoted = [item.strip() for item in re.findall(r"['\"]([^'\"]+)['\"]", query)]
    if len(quoted) >= 2:
        return _dedupe(quoted)
    match = _COMPARISON_TAIL_RE.search(query)
    tail = match.group("tail") if match else query
    parts = re.split(r"\s*,\s*|\s+\bor\b\s+|\s+\band\b\s+", tail, flags=re.IGNORECASE)
    cleaned = []
    for part in parts:
        value = re.sub(r"^(?:the|a|an|my|our)\s+", "", part.strip(), flags=re.IGNORECASE)
        value = re.sub(r"[?.!]+$", "", value).strip()
        if 1 <= len(value.split()) <= 10:
            cleaned.append(value)
    return _dedupe(cleaned) if len(cleaned) >= 2 else []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(value.strip())
    return out
