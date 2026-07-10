"""Proof-oriented execution over the typed compiled-memory projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import re
from typing import Any

from .compiler import MemoryIndex, claim_tokens, extract_quantity_claims
from .query_ir import QueryAnalysis
from .ranking import RetrievalSignals
from .taxonomy import (
    category_mentions,
    infer_category_ids,
    infer_predicate,
    predicates_compatible,
)


_SNAPSHOT_COUNT_RE = re.compile(
    r"\b(?:how many|number of|count)\b.*\b(?:do i have|do we have|have i|have we|i have|we have|"
    r"currently|current|as of now|so far|already|worn|need)\b|"
    r"\bhow many\b.+\b(?:are|is)\b\s+(?:currently\s+)?(?:on|in|at|with|part of)\b",
    re.IGNORECASE,
)
_PROFESSIONAL_IDENTITY_RE = re.compile(
    r"\b(?:works? in (?:the )?(?:field|area|industry)|field of|"
    r"specializ(?:es|ing)|specialized in|professional expertise)\b",
    re.IGNORECASE,
)
_PROFESSIONAL_ACTIVITY_RE = re.compile(
    r"\b(?:research(?:es|ing)|conduct(?:s|ed|ing) research|"
    r"research (?:area|focus|interest|work))\b",
    re.IGNORECASE,
)
_INTEREST_PROFILE_RE = re.compile(
    r"\b(?:interested in|curious about|stud(?:y|ies|ying)|learning about|follows?)\b",
    re.IGNORECASE,
)
_DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "knowledge": ("article", "conference", "course", "journal", "paper", "publication", "research", "resource"),
    "travel": ("airbnb", "flight", "hotel", "resort", "travel", "trip", "vacation"),
    "media": ("documentary", "film", "movie", "netflix", "podcast", "show", "television", "watch"),
    "reading": ("author", "book", "novel", "read"),
    "food": ("cafe", "cuisine", "dinner", "food", "meal", "restaurant"),
    "technology": ("ai", "artificial intelligence", "computer", "deep learning", "machine learning", "software", "technology"),
    "health": ("health", "healthcare", "medical", "medicine", "wellness"),
    "career": ("career", "job", "profession", "work"),
    "product": ("accessory", "device", "equipment", "gear", "product", "setup"),
}


@dataclass
class CompiledEvidence:
    operation: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    candidate_answer: str | None = None
    proof_complete: bool = False
    proof_requirements: list[str] = field(default_factory=list)
    satisfied_requirements: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    selected_claim_ids: list[str] = field(default_factory=list)
    selected_turn_ids: list[str] = field(default_factory=list)
    selected_event_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, *, max_rows: int = 16) -> str:
        if not self.rows and not self.candidate_answer:
            return ""
        lines = [f"[compiled-memory: {self.operation}]", "Evidence is source-grounded; verify any incomplete proof against raw sources."]
        if self.candidate_answer:
            label = "Verified candidate" if self.proof_complete else "Tentative candidate"
            lines.append(f"{label}: {self.candidate_answer}")
        lines.append(
            "Proof: "
            + ("complete" if self.proof_complete else "incomplete")
            + f" | confidence={self.confidence:.2f}"
        )
        if self.missing_requirements:
            lines.append("Missing: " + ", ".join(self.missing_requirements))
        if self.rows:
            lines.append("Rows:")
            for row in self.rows[:max_rows]:
                fields = [f"{key}={value}" for key, value in row.items() if value not in (None, "", [], ())]
                lines.append("- " + " | ".join(fields))
        return "\n".join(lines)


def execute_query(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    signals: RetrievalSignals | None = None,
) -> CompiledEvidence:
    """Run the operator-specific deterministic compiler/executor."""

    signals = signals or RetrievalSignals()
    operator = analysis.primary_operator
    if operator == "ordinal":
        evidence = _execute_ordinal(index, analysis, signals)
    elif operator in {"current", "latest", "previous"}:
        evidence = _execute_state(index, analysis, signals)
    elif operator == "recommend":
        evidence = _execute_preferences(index, analysis, signals)
    elif operator in {"count", "sum", "max"}:
        evidence = _execute_aggregate(index, analysis, signals)
    elif operator in {"order", "date_delta"}:
        evidence = _execute_temporal(index, analysis, signals)
    else:
        evidence = _execute_lookup(index, analysis, signals)
    evidence.proof_requirements = list(analysis.proof_requirements)
    evidence.selected_claim_ids = list(dict.fromkeys(evidence.selected_claim_ids))
    evidence.selected_turn_ids = list(dict.fromkeys(evidence.selected_turn_ids))
    evidence.selected_event_ids = list(dict.fromkeys(evidence.selected_event_ids))
    evidence.missing_requirements = [
        requirement
        for requirement in evidence.proof_requirements
        if requirement not in set(evidence.satisfied_requirements)
    ]
    evidence.proof_complete = not evidence.missing_requirements and bool(evidence.rows)
    return evidence


def _execute_ordinal(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    position = int(analysis.metadata.get("ordinal") or 0)
    candidates = [item for item in index.compiled.list_items if item.position == position and item.role in analysis.source_roles]
    ranked = sorted(
        candidates,
        key=lambda item: (-signals.turn_scores.get(item.source_turn_id, 0.0), item.observed_at or "", item.item_id),
    )
    list_scores: dict[str, float] = {}
    for item in candidates:
        list_scores[item.list_id] = max(
            list_scores.get(item.list_id, 0.0),
            signals.turn_scores.get(item.source_turn_id, 0.0),
        )
    ranked_lists = sorted(list_scores, key=lambda list_id: (-list_scores[list_id], list_id))
    best_list = ranked_lists[0] if ranked_lists else None
    list_margin = (
        list_scores[ranked_lists[0]] - list_scores[ranked_lists[1]]
        if len(ranked_lists) > 1
        else (list_scores[ranked_lists[0]] if ranked_lists else 0.0)
    )
    ranked = [item for item in ranked if item.list_id == best_list] + [
        item for item in ranked if item.list_id != best_list
    ]
    selected = [item for item in ranked if item.list_id == best_list][:1]
    identity_resolved = bool(selected and list_scores.get(best_list or "", 0.0) > 0 and list_margin >= 0.02)
    displayed = selected if identity_resolved else ranked[:4]
    rows = [
        {
            "position": item.position,
            "item": item.text,
            "date": item.observed_at,
            "source": item.source_turn_id,
        }
        for item in displayed
    ]
    confidence = 0.96 if selected and signals.turn_scores.get(selected[0].source_turn_id, 0.0) > 0 else (0.72 if selected else 0.0)
    return CompiledEvidence(
        operation="ordinal",
        rows=rows,
        candidate_answer=selected[0].text if len(selected) == 1 else None,
        satisfied_requirements=(
            ["source_provenance", "ordinal_position", "entity_compatibility"]
            + (["list_identity"] if identity_resolved else [])
            if selected
            else []
        ),
        selected_turn_ids=[item.source_turn_id for item in selected],
        confidence=confidence,
        metadata={
            "position": position,
            "candidate_lists": len(ranked_lists),
            "best_list_id": best_list,
            "list_score_margin": list_margin,
        },
    )


def _execute_state(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    scored = []
    q_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    for state in index.compiled.state_versions:
        if state.metadata.get("role") not in analysis.source_roles:
            continue
        if not _not_after_anchor(state.observed_at, analysis):
            continue
        lexical = _overlap(q_tokens, claim_tokens(state.value_text))
        dense = signals.state_scores.get(state.state_id, 0.0)
        score = (0.62 * dense) + (0.38 * lexical)
        if score <= 0:
            continue
        scored.append((score, state.observed_at or "", state))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].state_id))
    top_keys = []
    for _, _, state in scored:
        if state.state_key not in top_keys:
            top_keys.append(state.state_key)
        if len(top_keys) >= 3:
            break
    versions = [
        state
        for state in index.compiled.state_versions
        if state.state_key in set(top_keys)
        and state.metadata.get("role") in analysis.source_roles
        and _not_after_anchor(state.observed_at, analysis)
    ]
    versions.sort(key=lambda state: (state.state_key, state.observed_at or "", state.state_id))
    rows = [
        {
            "state": state.value_text,
            "status": state.status,
            "valid_from": state.valid_from,
            "observed": state.observed_at,
            "source": state.source_turn_ids[0] if state.source_turn_ids else None,
        }
        for state in versions[-12:]
    ]
    best_versions = [state for state in versions if top_keys and state.state_key == top_keys[0]]
    best_versions.sort(key=lambda state: (state.observed_at or "", state.state_id))
    selected = best_versions[-1:]
    if analysis.primary_operator == "previous" and len(best_versions) >= 2:
        selected = best_versions[-2:-1]
    candidate = selected[0].value_text if selected else None
    confidence = min(0.92, scored[0][0] + 0.15) if scored else 0.0
    satisfied = ["source_provenance", "entity_compatibility"]
    if versions:
        satisfied.extend(["state_history", "supersession_check"])
    return CompiledEvidence(
        operation=f"state/{analysis.primary_operator}",
        rows=rows,
        candidate_answer=candidate,
        satisfied_requirements=satisfied,
        selected_claim_ids=[state.source_claim_id for state in selected if state.source_claim_id],
        selected_turn_ids=[turn_id for state in selected for turn_id in state.source_turn_ids],
        confidence=confidence,
        metadata={"matched_state_keys": top_keys, "versions": len(versions)},
    )


def _execute_preferences(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    q_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    query_domains = _profile_domains(analysis.query)
    scored = []
    for preference in index.compiled.preferences:
        if not _not_after_anchor(preference.observed_at, analysis):
            continue
        lexical = _overlap(q_tokens, claim_tokens(preference.text))
        dense = signals.preference_scores.get(preference.preference_id, 0.0)
        preference_domains = _profile_domains(preference.text)
        profile_kind = _profile_kind(preference.text)
        domain_match = _profile_domain_match(query_domains, preference_domains, profile_kind)
        score = (
            (0.52 * dense)
            + (0.23 * lexical)
            + (0.24 * domain_match)
            + (0.44 if profile_kind == "professional_identity" and "knowledge" in query_domains else 0.0)
            + (0.17 if profile_kind == "professional_activity" and "knowledge" in query_domains else 0.0)
            + (0.05 if profile_kind == "interest" and "knowledge" in query_domains else 0.0)
            + (0.05 if preference.explicit else 0.0)
        )
        if score > 0.0:
            scored.append((score, domain_match, profile_kind, preference.observed_at or "", preference))
    scored.sort(key=lambda item: (-item[0], -item[1], item[3], item[4].preference_id))
    picked = [item[4] for item in scored[:10]]
    rows = [
        {
            "preference": preference.text,
            "polarity": preference.polarity,
            "explicit": preference.explicit,
            "profile_kind": _profile_kind(preference.text),
            "domains": sorted(_profile_domains(preference.text)),
            "observed": preference.observed_at,
            "source": preference.source_turn_ids[0] if preference.source_turn_ids else None,
        }
        for preference in picked
    ]
    top_domain_match = scored[0][1] if scored else 0.0
    satisfied = ["source_provenance", "entity_compatibility"] if picked else []
    if picked:
        if not query_domains or top_domain_match > 0:
            satisfied.append("preference_scope")
        if not query_domains or any(item[1] > 0 for item in scored[:5]):
            satisfied.append("constraint_coverage")
    return CompiledEvidence(
        operation="preference/profile",
        rows=rows,
        satisfied_requirements=satisfied,
        selected_claim_ids=[preference.source_claim_id for preference in picked if preference.source_claim_id],
        selected_turn_ids=[turn_id for preference in picked for turn_id in preference.source_turn_ids],
        confidence=min(0.9, scored[0][0] + 0.1) if scored else 0.0,
        metadata={
            "query_domains": sorted(query_domains),
            "top_profile_kinds": [item[2] for item in scored[:10]],
        },
    )


def _execute_aggregate(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    if analysis.primary_operator == "count" and _SNAPSHOT_COUNT_RE.search(analysis.query):
        return _execute_state_snapshot_count(index, analysis, signals)
    q_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    category_constraints = _aggregate_category_constraints(analysis.query)
    query_predicate = infer_predicate(analysis.query)
    candidates = []
    for event in index.compiled.canonical_events:
        if analysis.completed_only and event.modality != "actual":
            continue
        if not set(event.metadata.get("roles") or ()) & set(analysis.source_roles):
            continue
        if not _not_after_anchor(event.metadata.get("observed_at"), analysis):
            continue
        if not _in_window(event.event_start, analysis.time_start, analysis.time_end):
            continue
        if query_predicate != "state" and not predicates_compatible(query_predicate, event.predicate):
            continue
        category_match = bool(category_constraints & set(event.category_ids))
        if category_constraints and not category_match:
            continue
        lexical = _overlap(q_tokens, claim_tokens(event.summary))
        dense = signals.event_scores.get(event.event_id, 0.0)
        score = (0.52 * dense) + (0.28 * lexical) + (0.35 if category_match else 0.0)
        if score <= 0.0:
            continue
        candidates.append((score, event.event_start or "", event))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2].event_id))
    if candidates and category_constraints:
        selected = [event for _, _, event in candidates][:200]
    elif candidates:
        threshold = max(0.18, candidates[0][0] * 0.58)
        selected = [event for score, _, event in candidates if score >= threshold][:40]
    else:
        selected = []
    rows = [
        {
            "event": event.summary,
            "date": event.event_start,
            "quantities": [
                f"{quantity.get('value')} {quantity.get('unit') or quantity.get('property_name')}"
                for quantity in event.quantities
            ],
            "sources": ",".join(event.source_turn_ids[:3]),
            "count_contribution": _event_cardinality(event, category_constraints, q_tokens),
            "matched_items": _event_category_items(event, category_constraints),
        }
        for event in sorted(selected, key=lambda event: (event.event_start or "", event.event_id))
    ]
    candidate_answer = None
    if analysis.primary_operator == "count" and selected:
        candidate_answer = str(
            sum(_event_cardinality(event, category_constraints, q_tokens) for event in selected)
        )
    elif analysis.primary_operator == "sum":
        values = [
            float(quantity["value"])
            for event in selected
            for quantity in event.quantities
            if quantity.get("value") is not None and quantity.get("unit") == "usd"
        ]
        if values:
            candidate_answer = f"${sum(values):g}"
    elif analysis.primary_operator == "max" and selected:
        numeric = [
            (float(quantity["value"]), event)
            for event in selected
            for quantity in event.quantities
            if quantity.get("value") is not None
        ]
        if numeric:
            value, event = max(numeric, key=lambda item: item[0])
            candidate_answer = f"{value:g}: {event.summary}"
    core_covered = bool(selected and category_constraints) or _core_terms_covered(q_tokens, selected)
    satisfied = ["source_provenance"]
    if core_covered:
        satisfied.append("entity_compatibility")
    if selected and core_covered:
        satisfied.extend(["bounded_candidate_set", "canonical_event_deduplication"])
    source_coverage, missing_source_ids = _aggregate_source_coverage(
        index,
        analysis,
        selected,
        category_constraints=category_constraints,
        query_predicate=query_predicate,
    )
    if source_coverage:
        satisfied.append("source_coverage")
    confidence = min(0.9, candidates[0][0] + 0.12) if selected else 0.0
    if not core_covered:
        candidate_answer = None
        confidence *= 0.5
    return CompiledEvidence(
        operation=f"aggregate/{analysis.primary_operator}",
        rows=rows,
        candidate_answer=candidate_answer,
        satisfied_requirements=satisfied,
        selected_claim_ids=[claim_id for event in selected for claim_id in event.claim_ids],
        selected_turn_ids=[turn_id for event in selected for turn_id in event.source_turn_ids],
        selected_event_ids=[event.event_id for event in selected],
        confidence=confidence,
        metadata={
            "bounded_index_scan": True,
            "category_constraints": sorted(category_constraints),
            "query_predicate": query_predicate,
            "source_coverage": source_coverage,
            "missing_source_ids": missing_source_ids,
            "matched_events": len(selected),
        },
    )


def _execute_state_snapshot_count(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    signals: RetrievalSignals,
) -> CompiledEvidence:
    q_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    candidates = []
    for state in index.compiled.state_versions:
        quantity = state.quantity or {}
        if quantity.get("value") is None:
            continue
        if state.metadata.get("role") not in analysis.source_roles:
            continue
        if not _not_after_anchor(state.observed_at, analysis):
            continue
        lexical = _overlap(q_tokens, claim_tokens(state.value_text))
        dense = signals.state_scores.get(state.state_id, 0.0)
        quantity_terms = claim_tokens(
            " ".join(
                str(value)
                for value in (
                    quantity.get("unit"),
                    quantity.get("property_name"),
                )
                if value
            )
        )
        measure_match = _overlap(q_tokens, quantity_terms)
        score = (
            (0.52 * dense)
            + (0.28 * lexical)
            + (0.28 * measure_match)
            + (0.04 if state.metadata.get("role") == "user" else 0.0)
        )
        if score > 0:
            candidates.append((score, state.observed_at or "", state))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2].state_id))
    selected = candidates[0][2] if candidates else None
    rows = []
    if selected:
        related = [state for state in index.compiled.state_versions if state.state_key == selected.state_key]
        related.sort(key=lambda state: (state.observed_at or "", state.state_id))
        rows = [
            {
                "state": state.value_text,
                "value": (state.quantity or {}).get("value"),
                "unit": (state.quantity or {}).get("unit"),
                "status": state.status,
                "observed": state.observed_at,
                "source": state.source_turn_ids[0] if state.source_turn_ids else None,
            }
            for state in related[-8:]
        ]
        related = [state for state in related if _not_after_anchor(state.observed_at, analysis)]
        selected = related[-1]
    quantity = selected.quantity if selected else None
    candidate = None
    if quantity and quantity.get("value") is not None:
        value = float(quantity["value"])
        candidate = str(int(value)) if value.is_integer() else str(value)
    satisfied = ["source_provenance", "entity_compatibility"] if selected else []
    if selected:
        satisfied.extend(
            [
                "bounded_candidate_set",
                "canonical_event_deduplication",
                "source_coverage",
                "state_history",
                "supersession_check",
            ]
        )
    return CompiledEvidence(
        operation="state/snapshot-count",
        rows=rows,
        candidate_answer=candidate,
        satisfied_requirements=satisfied,
        selected_claim_ids=[selected.source_claim_id] if selected and selected.source_claim_id else [],
        selected_turn_ids=list(selected.source_turn_ids) if selected else [],
        confidence=min(0.94, candidates[0][0] + 0.14) if candidates else 0.0,
        metadata={"snapshot": True, "state_key": selected.state_key if selected else None},
    )


def _execute_temporal(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    rows = []
    found_operands: list[str] = []
    selected_events = []
    operands = analysis.operands or [analysis.query]
    query_predicate = infer_predicate(analysis.query)
    predicate_matches: list[bool] = []
    for operand in operands:
        operand_tokens = claim_tokens(operand)
        candidates = []
        for event in index.compiled.canonical_events:
            if not event.event_start or event.modality != "actual":
                continue
            if not set(event.metadata.get("roles") or ()) & set(analysis.source_roles):
                continue
            if not _not_after_anchor(event.metadata.get("observed_at"), analysis):
                continue
            lexical = _overlap(operand_tokens, claim_tokens(event.summary))
            if lexical <= 0.0:
                continue
            dense = signals.event_scores.get(event.event_id, 0.0)
            score = (0.58 * dense) + (0.42 * lexical)
            compatible = query_predicate == "state" or predicates_compatible(query_predicate, event.predicate)
            if score > 0:
                candidates.append((compatible, lexical, event.event_start, score, event))
        candidates.sort(key=lambda item: (-int(item[0]), -item[1], item[2], -item[3], item[4].event_id))
        if not candidates:
            continue
        compatible, _, _, _, event = candidates[0]
        found_operands.append(operand)
        selected_events.append((operand, event))
        predicate_matches.append(compatible)
    selected_events.sort(key=lambda item: (item[1].event_start or "", item[1].event_id))
    for operand, event in selected_events:
        rows.append({"operand": operand, "date": event.event_start, "event": event.summary, "source": event.source_turn_ids[0] if event.source_turn_ids else None})
    candidate = None
    if analysis.primary_operator == "order" and len(selected_events) == len(operands) and len(operands) >= 2:
        candidate = " -> ".join(operand for operand, _ in selected_events)
    elif analysis.primary_operator == "date_delta" and len(selected_events) >= 2:
        first = _parse_date(selected_events[0][1].event_start)
        last = _parse_date(selected_events[-1][1].event_start)
        if first and last:
            candidate = f"{abs((last - first).days)} days"
    satisfied = ["source_provenance"]
    if selected_events:
        satisfied.extend(["entity_compatibility", "event_time_resolution"])
    if len(found_operands) == len(operands) and len(operands) >= 2:
        satisfied.extend(["operand_coverage", "all_operands_found"])
    return CompiledEvidence(
        operation=f"temporal/{analysis.primary_operator}",
        rows=rows,
        candidate_answer=candidate,
        satisfied_requirements=satisfied,
        selected_claim_ids=[claim_id for _, event in selected_events for claim_id in event.claim_ids],
        selected_turn_ids=[turn_id for _, event in selected_events for turn_id in event.source_turn_ids],
        selected_event_ids=[event.event_id for _, event in selected_events],
        confidence=0.88 if candidate else (0.58 if rows else 0.0),
        metadata={
            "operands": operands,
            "found_operands": found_operands,
            "query_predicate": query_predicate,
            "predicate_matches": predicate_matches,
        },
    )


def _execute_lookup(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    q_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    scored = []
    for record in index.claims:
        if record.claim.metadata.get("role") not in analysis.source_roles:
            continue
        if not _not_after_anchor(record.claim.observed_at, analysis):
            continue
        lexical = _overlap(q_tokens, claim_tokens(record.text))
        dense = signals.claim_scores.get(record.claim_id, 0.0)
        score = (0.7 * dense) + (0.3 * lexical)
        if score > 0:
            scored.append((score, record.sort_key, record))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].claim_id))
    picked = [item[2] for item in scored[:6]]
    rows = [{"claim": record.text, "date": record.claim.observed_at, "sources": ",".join(record.source_turn_ids)} for record in picked]
    return CompiledEvidence(
        operation="lookup",
        rows=rows,
        satisfied_requirements=["source_provenance", "entity_compatibility"] if picked else [],
        selected_claim_ids=[record.claim_id for record in picked],
        selected_turn_ids=[turn_id for record in picked for turn_id in record.source_turn_ids],
        confidence=min(0.9, scored[0][0] + 0.1) if scored else 0.0,
    )


def _core_terms_covered(query_tokens: set[str], events) -> bool:
    if not query_tokens:
        return True
    event_tokens = set().union(*(claim_tokens(event.summary) for event in events)) if events else set()
    return len(query_tokens & event_tokens) >= min(2, len(query_tokens))


def _in_window(value: str | None, start: str | None, end: str | None) -> bool:
    if not start and not end:
        return True
    parsed = _parse_date(value)
    if parsed is None:
        return False
    lower = _parse_date(start)
    upper = _parse_date(end)
    return (lower is None or parsed >= lower) and (upper is None or parsed <= upper)


def _not_after_anchor(value: str | None, analysis: QueryAnalysis) -> bool:
    anchor = _parse_date(str(analysis.metadata.get("time_anchor") or ""))
    observed = _parse_date(value)
    return anchor is None or observed is None or observed <= anchor


def _aggregate_category_constraints(query: str) -> set[str]:
    categories = set(infer_category_ids(query))
    specific = categories - {"expense", "event", "vehicle"}
    return specific or categories


def _event_category_items(event, category_constraints: set[str]) -> list[str]:
    return sorted(
        {
            mention
            for category_id in category_constraints
            for mention in category_mentions(event.summary, category_id)
        }
    )


def _event_cardinality(event, category_constraints: set[str], query_tokens: set[str]) -> int:
    for quantity in event.quantities:
        value = quantity.get("value")
        if value is None or quantity.get("unit") == "usd":
            continue
        unit_tokens = claim_tokens(
            f"{quantity.get('unit') or ''} {quantity.get('property_name') or ''}"
        )
        if unit_tokens & query_tokens:
            return max(1, int(float(value)))
    items = _event_category_items(event, category_constraints)
    return max(1, len(items))


def _aggregate_source_coverage(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    selected,
    *,
    category_constraints: set[str],
    query_predicate: str,
) -> tuple[bool, list[str]]:
    covered = {turn_id for event in selected for turn_id in event.source_turn_ids}
    candidates: set[str] = set()
    for turn in index.turns:
        if turn.role not in analysis.source_roles:
            continue
        if not _in_window(turn.session_date, analysis.time_start, analysis.time_end):
            continue
        if category_constraints and not category_constraints & set(infer_category_ids(turn.content)):
            continue
        if analysis.primary_operator == "sum":
            if not any(quantity.unit == "usd" for quantity in extract_quantity_claims(turn.content)):
                continue
        elif query_predicate != "state" and not predicates_compatible(
            query_predicate,
            infer_predicate(turn.content),
        ):
            continue
        candidates.add(turn.turn_id)
    missing = sorted(candidates - covered)
    return bool(selected) and not missing, missing


def _profile_domains(text: str) -> set[str]:
    lower = text.lower()
    tokens = claim_tokens(text)
    return {
        domain
        for domain, terms in _DOMAIN_TERMS.items()
        if any(
            (claim_tokens(term) and claim_tokens(term) <= tokens)
            or (term == "ai" and re.search(r"\bai\b", lower))
            for term in terms
        )
    }


def _profile_kind(text: str) -> str:
    if _PROFESSIONAL_IDENTITY_RE.search(text):
        return "professional_identity"
    if _PROFESSIONAL_ACTIVITY_RE.search(text):
        return "professional_activity"
    if _INTEREST_PROFILE_RE.search(text):
        return "interest"
    return "taste_or_constraint"


def _profile_domain_match(
    query_domains: set[str],
    preference_domains: set[str],
    profile_kind: str,
) -> float:
    if not query_domains:
        return 0.0
    overlap = query_domains & preference_domains
    if overlap:
        return min(1.0, 0.6 + (0.2 * len(overlap)))
    if "knowledge" in query_domains and profile_kind in {
        "professional_identity",
        "professional_activity",
        "interest",
    }:
        return 0.7
    return 0.0


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10].replace("/", "-"))
    except ValueError:
        return None


def _overlap(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)
