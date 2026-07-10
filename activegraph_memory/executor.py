"""Proof-oriented execution over the typed compiled-memory projection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import re
from typing import Any

from .compiler import MemoryIndex, SourceTurn, claim_tokens, extract_quantity_claims
from .coverage_audit import SourceCoverageAudit, audit_source_coverage
from .query_ir import QueryAnalysis
from .ranking import RetrievalSignals
from .temporal import extract_temporal_refs
from .taxonomy import (
    category_mentions,
    expanded_query_variants,
    infer_category_ids,
    infer_predicate,
    most_specific_categories,
    predicates_compatible,
    significant_tokens,
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
_PREFERENCE_EXPRESSION_RE = re.compile(
    r"\b(?:prefer|prefers|preference|favorite|like|likes|love|loves|enjoy|enjoys|"
    r"interested in|want|wants|need|needs|avoid|avoids|without|do not|don't|"
    r"dislike|dislikes|hate|hates|important to me|works? (?:best|well) for me)\b",
    re.IGNORECASE,
)
_NEGATIVE_PREFERENCE_RE = re.compile(
    r"\b(?:avoid|avoids|without|do not|don't|dislike|dislikes|hate|hates|never|not want|no [a-z])\b",
    re.IGNORECASE,
)
_NON_ACTUAL_SOURCE_RE = re.compile(
    r"\b(?:plan|plans|planned|planning|intend|intends|intended|hope|hopes|"
    r"might|may|could|would|will|hypothetical|if i|if we)\b",
    re.IGNORECASE,
)
_AGGREGATE_GENERIC_TERMS = {
    "amount", "buy", "bought", "cost", "expense", "money", "paid", "pay",
    "purchase", "spend", "spent",
}
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
_NEGATIVE_SCAN_STOPWORDS = {
    "did",
    "do",
    "does",
    "ever",
    "have",
    "had",
    "has",
    "never",
    "not",
    "buy",
    "bought",
    "purchase",
    "purchased",
    "get",
    "got",
    "make",
    "made",
    "use",
    "used",
    "visit",
    "visited",
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
    evidence_slots: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, *, max_rows: int = 16, include_candidate: bool = True) -> str:
        if not self.rows and not self.candidate_answer:
            return ""
        lines = [
            f"[compiled-memory: {self.operation}]",
            "Evidence is source-grounded; verify any incomplete proof against raw sources.",
            "Reader contract: proof status confirms required evidence fields, not answer correctness; check every candidate against cited sources.",
        ]
        if any("temporal_distance_days" in row for row in self.rows):
            lines.append(
                "For approximate relative time, apply temporal-distance tolerance and semantic fit; calendar-day equality alone is not decisive."
            )
        if self.candidate_answer and include_candidate:
            label = "Proof-complete candidate" if self.proof_complete else "Incomplete candidate"
            lines.append(f"{label}: {self.candidate_answer}")
        lines.append(
            "Proof: "
            + ("complete" if self.proof_complete else "incomplete")
            + f" | confidence={self.confidence:.2f}"
        )
        if self.missing_requirements:
            lines.append("Missing: " + ", ".join(self.missing_requirements))
        if self.evidence_slots:
            lines.append(
                "Required evidence slots (preserve names, dates, quantities, and polarity in the final answer):"
            )
            for slot in self.evidence_slots[:max_rows]:
                fields = [
                    f"{key}={value}"
                    for key, value in slot.items()
                    if value not in (None, "", [], ())
                ]
                lines.append("- " + " | ".join(fields))
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
    elif operator == "negative_existence":
        evidence = _execute_negative_existence(index, analysis, signals)
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


def _execute_negative_existence(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    signals: RetrievalSignals,
) -> CompiledEvidence:
    """Produce a bounded-memory absence certificate or positive counterevidence."""

    q_tokens = (set(analysis.entity_terms) or claim_tokens(analysis.query)) - _NEGATIVE_SCAN_STOPWORDS
    matches = []
    scanned = 0
    for record in index.claims:
        role = str(record.claim.metadata.get("role") or "unknown")
        if role not in analysis.source_roles:
            continue
        observed = record.claim.observed_at
        if not _in_window(observed, analysis.time_start, analysis.time_end):
            continue
        scanned += 1
        overlap = _overlap(q_tokens, claim_tokens(record.text))
        dense = signals.claim_scores.get(record.claim_id, 0.0)
        if overlap >= 0.45 or (overlap >= 0.2 and dense > 0.02):
            matches.append((max(overlap, dense), record))
    matches.sort(key=lambda item: (-item[0], item[1].sort_key, item[1].claim_id))
    selected = [record for _, record in matches[:12]]
    if selected:
        rows = [
            {
                "counterevidence": record.text,
                "observed_at": record.claim.observed_at,
                "sources": ",".join(record.source_turn_ids),
            }
            for record in selected
        ]
        candidate = "Matching evidence exists in memory."
    else:
        rows = [
            {
                "absence_scope": "compiled accepted memory",
                "claims_scanned": scanned,
                "source_roles": ",".join(analysis.source_roles),
                "time_window": analysis.time_label,
            }
        ]
        candidate = "No matching evidence was found in the bounded memory scope."
    return CompiledEvidence(
        operation="negative-existence/bounded-scan",
        rows=rows,
        candidate_answer=candidate,
        satisfied_requirements=[
            "entity_compatibility",
            "bounded_candidate_set",
            "source_coverage",
            "absence_certificate",
        ],
        selected_claim_ids=[record.claim_id for record in selected],
        selected_turn_ids=[turn_id for record in selected for turn_id in record.source_turn_ids],
        confidence=0.9 if selected else 0.82,
        metadata={
            "bounded_scope": "compiled accepted memory",
            "claims_scanned": scanned,
            "matches_found": len(matches),
            "world_level_absence_claim": False,
        },
    )


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
    eligible_dates = [
        parsed
        for state in index.compiled.state_versions
        if state.metadata.get("role") in analysis.source_roles
        and _not_after_anchor(state.observed_at, analysis)
        for parsed in (_parse_date(state.observed_at),)
        if parsed is not None
    ]
    newest_observation = max(eligible_dates) if eligible_dates else None
    for state in index.compiled.state_versions:
        if state.metadata.get("role") not in analysis.source_roles:
            continue
        if not _not_after_anchor(state.observed_at, analysis):
            continue
        lexical = _overlap(q_tokens, claim_tokens(state.value_text))
        dense = signals.state_scores.get(state.state_id, 0.0)
        transition = _state_transition_score(index, state)
        recency = _state_recency_score(state.observed_at, newest_observation)
        score = (0.52 * dense) + (0.3 * lexical) + (0.32 * transition) + (0.18 * recency)
        if score <= 0:
            continue
        scored.append((score, state.observed_at or "", state, transition, recency))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].state_id))
    top_keys = []
    for _, _, state, _, _ in scored:
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
    confidence = min(0.92, scored[0][0] + 0.12) if scored else 0.0
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
        metadata={
            "matched_state_keys": top_keys,
            "versions": len(versions),
            "top_transition_score": scored[0][3] if scored else 0.0,
            "top_recency_score": scored[0][4] if scored else 0.0,
        },
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
        scope_match = _overlap(q_tokens, set(preference.scope_terms))
        score = (
            (0.38 * dense)
            + (0.2 * lexical)
            + (0.24 * domain_match)
            + (0.3 * scope_match)
            + (0.44 if profile_kind == "professional_identity" and "knowledge" in query_domains else 0.0)
            + (0.17 if profile_kind == "professional_activity" and "knowledge" in query_domains else 0.0)
            + (0.05 if profile_kind == "interest" and "knowledge" in query_domains else 0.0)
            + (0.08 if preference.explicit else 0.0)
        )
        if score > 0.0:
            scored.append(
                (score, domain_match, scope_match, profile_kind, preference.observed_at or "", preference)
            )
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[4], item[5].preference_id))
    direct = [item[5] for item in scored if item[1] > 0 or item[2] > 0]
    negative = [item[5] for item in scored if item[5].polarity == "negative"]
    positive = [item[5] for item in scored if item[5].polarity != "negative"]
    picked = _dedupe_records([*direct[:8], *negative[:4], *positive[:4], *[item[5] for item in scored[:6]]])[:12]
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
    raw_candidates = _preference_source_candidates(index, analysis, signals, q_tokens, query_domains)
    selected_ids = {turn_id for preference in picked for turn_id in preference.source_turn_ids}
    extracted_ids = {
        turn_id
        for record in index.claims
        if _PREFERENCE_EXPRESSION_RE.search(record.text)
        for turn_id in record.source_turn_ids
    }
    compiled_ids = {
        turn_id
        for preference in index.compiled.preferences
        for turn_id in preference.source_turn_ids
    }
    recovery = [turn_id for _, turn_id in raw_candidates if turn_id not in selected_ids][:8]
    coverage = audit_source_coverage(
        [turn_id for _, turn_id in raw_candidates],
        extracted_source_ids=extracted_ids,
        compiled_source_ids=compiled_ids,
        selected_source_ids=selected_ids,
        recovery_source_ids=recovery,
        minimum_ratio=0.85,
        metadata={"operator": "recommend"},
    )
    top_domain_match = scored[0][1] if scored else 0.0
    top_scope_match = scored[0][2] if scored else 0.0
    satisfied = ["source_provenance", "entity_compatibility"] if picked else []
    if picked:
        if top_domain_match > 0 or top_scope_match > 0:
            satisfied.append("preference_scope")
        negative_candidates = {
            turn_id
            for _, turn_id in raw_candidates
            if _NEGATIVE_PREFERENCE_RE.search(index.by_turn_id[turn_id].content)
        }
        if negative_candidates <= (selected_ids | set(recovery)):
            satisfied.append("constraint_coverage")
        if coverage.complete:
            satisfied.append("preference_coverage")
    slots = [
        {
            "slot": "negative_constraint" if preference.polarity == "negative" else "positive_preference",
            "status": "selected",
            "scope": list(preference.scope_terms),
            "evidence": preference.text,
            "source": preference.source_turn_ids[0] if preference.source_turn_ids else None,
        }
        for preference in picked
    ]
    slots.extend(
        {
            "slot": (
                "negative_constraint_recovery"
                if _NEGATIVE_PREFERENCE_RE.search(index.by_turn_id[turn_id].content)
                else "positive_preference_recovery"
            ),
            "status": "raw_source_not_in_compiled_result",
            "evidence": _evidence_excerpt(index.by_turn_id[turn_id].content),
            "source": turn_id,
        }
        for turn_id in recovery
    )
    return CompiledEvidence(
        operation="preference/profile",
        rows=rows,
        satisfied_requirements=satisfied,
        selected_claim_ids=[preference.source_claim_id for preference in picked if preference.source_claim_id],
        selected_turn_ids=[
            *[turn_id for preference in picked for turn_id in preference.source_turn_ids],
            *recovery,
        ],
        evidence_slots=slots,
        confidence=min(0.9, scored[0][0] + 0.1) if scored else 0.0,
        metadata={
            "query_domains": sorted(query_domains),
            "top_profile_kinds": [item[3] for item in scored[:10]],
            "coverage_audit": coverage.model_dump(),
        },
    )


def _preference_source_candidates(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    signals: RetrievalSignals,
    q_tokens: set[str],
    query_domains: set[str],
) -> list[tuple[float, str]]:
    candidates = []
    dense_floor = max(signals.turn_scores.values(), default=0.0) * 0.7
    for turn in index.turns:
        if turn.role != "user" or not _not_after_anchor(turn.session_date, analysis):
            continue
        if not _PREFERENCE_EXPRESSION_RE.search(turn.content):
            continue
        domains = _profile_domains(turn.content)
        domain_match = 1.0 if query_domains & domains else 0.0
        lexical = _overlap(q_tokens, claim_tokens(turn.content))
        dense = signals.turn_scores.get(turn.turn_id, 0.0)
        if query_domains and not domain_match and lexical <= 0.0:
            continue
        if not query_domains and lexical <= 0.0 and (dense_floor <= 0.0 or dense < dense_floor):
            continue
        candidates.append(((0.45 * dense) + (0.35 * lexical) + (0.3 * domain_match), turn.turn_id))
    return sorted(candidates, key=lambda item: (-item[0], item[1]))


def _dedupe_records(values):
    seen = set()
    out = []
    for value in values:
        key = getattr(value, "preference_id", id(value))
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _state_transition_score(index: MemoryIndex, state) -> float:
    source_text = " ".join(
        index.by_turn_id[turn_id].content
        for turn_id in state.source_turn_ids
        if turn_id in index.by_turn_id
    )
    if re.search(
        r"\b(?:switched to|changed to|moved to|replaced .+ with|no longer|wrapped up|"
        r"just (?:started|finished|completed|changed|switched)|now (?:working|using|building|driving|owning))\b",
        source_text,
        re.IGNORECASE,
    ):
        return 1.0
    if re.search(r"\b(?:currently|now|latest|new|current)\b", source_text, re.IGNORECASE):
        return 0.55
    return 0.0


def _state_recency_score(observed_at: str | None, newest: date | None) -> float:
    observed = _parse_date(observed_at)
    if observed is None or newest is None:
        return 0.0
    return 1.0 / (1.0 + max(0, (newest - observed).days))


def _execute_aggregate(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    if (
        analysis.primary_operator == "count"
        and _SNAPSHOT_COUNT_RE.search(analysis.query)
        and infer_predicate(analysis.query) == "state"
    ):
        return _execute_state_snapshot_count(index, analysis, signals)
    q_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    category_constraints = _aggregate_category_constraints(analysis.query)
    query_predicate = infer_predicate(analysis.query)
    candidates = []
    for canonical_event in index.compiled.canonical_events:
        event = _event_for_source_roles(index, canonical_event, set(analysis.source_roles))
        if event is None:
            continue
        if analysis.completed_only and event.modality != "actual":
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
            "count_contribution": _event_cardinality(
                event,
                category_constraints,
                q_tokens,
                query_predicate=query_predicate,
            ),
            "matched_items": _event_category_items(event, category_constraints),
        }
        for event in sorted(selected, key=lambda event: (event.event_start or "", event.event_id))
    ]
    candidate_answer = None
    if analysis.primary_operator == "count" and selected:
        candidate_answer = str(
            sum(
                _event_cardinality(
                    event,
                    category_constraints,
                    q_tokens,
                    query_predicate=query_predicate,
                )
                for event in selected
            )
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
    coverage = _aggregate_source_coverage(
        index,
        analysis,
        selected,
        signals,
        q_tokens=q_tokens,
        category_constraints=category_constraints,
        query_predicate=query_predicate,
    )
    if coverage.complete:
        satisfied.append("source_coverage")
    recovery_rows = [
        {
            "slot": "coverage_recovery",
            "status": "raw_source_not_in_compiled_result",
            "source": turn_id,
            "evidence": _evidence_excerpt(index.by_turn_id[turn_id].content),
        }
        for turn_id in coverage.recovery_source_ids[:6]
        if turn_id in index.by_turn_id
    ]
    event_slots = [
        {
            "slot": f"matched_event_{position}",
            "status": "selected",
            "event": event.summary,
            "date": event.event_start,
            "quantity": [
                f"{quantity.get('value')} {quantity.get('unit') or quantity.get('property_name')}"
                for quantity in event.quantities
                if quantity.get("value") is not None
            ],
            "count_contribution": _event_cardinality(
                event,
                category_constraints,
                q_tokens,
                query_predicate=query_predicate,
            ),
            "sources": list(event.source_turn_ids),
        }
        for position, event in enumerate(
            sorted(selected, key=lambda event: (event.event_start or "", event.event_id)),
            start=1,
        )
    ]
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
        selected_turn_ids=[
            *[turn_id for event in selected for turn_id in event.source_turn_ids],
            *coverage.recovery_source_ids,
        ],
        selected_event_ids=[event.event_id for event in selected],
        evidence_slots=[*event_slots, *recovery_rows],
        confidence=confidence,
        metadata={
            "bounded_index_scan": True,
            "category_constraints": sorted(category_constraints),
            "query_predicate": query_predicate,
            "source_coverage": coverage.complete,
            "coverage_audit": coverage.model_dump(),
            "missing_source_ids": sorted(
                set(coverage.missing_extraction_ids)
                | set(coverage.missing_compilation_ids)
                | set(coverage.missing_selection_ids)
            ),
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
    if analysis.primary_operator == "order" and not analysis.operands:
        timeline = _execute_source_timeline(index, analysis, signals)
        if timeline.rows:
            return timeline
    rows = []
    found_operands: list[str] = []
    selected_events = []
    selected_source_turn_ids: list[str] = []
    selected_dates: list[tuple[str, str]] = []
    operands = analysis.operands or [analysis.query]
    query_predicate = infer_predicate(analysis.query)
    predicate_matches: list[bool] = []
    for operand in operands:
        source_candidate = _source_operand_candidate(index, operand, analysis, signals)
        if source_candidate is not None:
            source_date, turn = source_candidate
            found_operands.append(operand)
            selected_source_turn_ids.append(turn.turn_id)
            selected_dates.append((operand, source_date))
            predicate_matches.append(True)
            rows.append(
                {
                    "operand": operand,
                    "date": source_date,
                    "event": turn.content,
                    "source": turn.turn_id,
                    "evidence_kind": "source_turn",
                }
            )
            continue
        operand_tokens = claim_tokens(operand)
        candidates = []
        for canonical_event in index.compiled.canonical_events:
            event = _event_for_source_roles(index, canonical_event, set(analysis.source_roles))
            if event is None:
                continue
            if not event.event_start or event.modality != "actual":
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
        selected_dates.append((operand, event.event_start))
        predicate_matches.append(compatible)
    rows.sort(key=lambda row: (row.get("date") or "", row.get("source") or ""))
    for operand, event in selected_events:
        rows.append({"operand": operand, "date": event.event_start, "event": event.summary, "source": event.source_turn_ids[0] if event.source_turn_ids else None})
    rows.sort(key=lambda row: (row.get("date") or "", row.get("source") or ""))
    selected_dates.sort(key=lambda item: (item[1], item[0]))
    candidate = None
    if analysis.primary_operator == "order" and len(selected_dates) == len(operands) and len(operands) >= 2:
        candidate = " -> ".join(operand for operand, _ in selected_dates)
    elif analysis.primary_operator == "date_delta" and len(selected_dates) >= 2:
        first = _parse_date(selected_dates[0][1])
        last = _parse_date(selected_dates[-1][1])
        if first and last:
            candidate = f"{abs((last - first).days)} days"
    satisfied = ["source_provenance"]
    if selected_events or selected_source_turn_ids:
        satisfied.extend(["entity_compatibility", "event_time_resolution"])
    if len(found_operands) == len(operands) and len(operands) >= 2:
        satisfied.extend(["operand_coverage", "all_operands_found"])
    slots = []
    for operand in operands:
        matching = [row for row in rows if row.get("operand") == operand]
        if matching:
            row = matching[0]
            slots.append(
                {
                    "slot": f"operand:{operand}",
                    "status": "found",
                    "date": row.get("date"),
                    "event": row.get("event"),
                    "source": row.get("source"),
                }
            )
        else:
            slots.append({"slot": f"operand:{operand}", "status": "missing"})
    return CompiledEvidence(
        operation=f"temporal/{analysis.primary_operator}",
        rows=rows,
        candidate_answer=candidate,
        satisfied_requirements=satisfied,
        selected_claim_ids=[claim_id for _, event in selected_events for claim_id in event.claim_ids],
        selected_turn_ids=selected_source_turn_ids + [turn_id for _, event in selected_events for turn_id in event.source_turn_ids],
        selected_event_ids=[event.event_id for _, event in selected_events],
        evidence_slots=slots,
        confidence=0.88 if candidate else (0.58 if rows else 0.0),
        metadata={
            "operands": operands,
            "found_operands": found_operands,
            "query_predicate": query_predicate,
            "predicate_matches": predicate_matches,
        },
    )


_SOURCE_EVENT_ACTION_RE = re.compile(
    r"\b(?:i|we)\s+(?:actually\s+|recently\s+|just\s+)?(?:visited|attended|participated|"
    r"took|went|saw|finished|completed|started|began|signed|bought|purchased|received|"
    r"returned|came back)\b",
    re.IGNORECASE,
)
_TEMPORAL_OPERAND_GENERIC = {
    "attendance",
    "begin",
    "beginning",
    "event",
    "happen",
    "happened",
    "start",
    "started",
}
_COUNT_WORDS = {
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
_VENUE_PATTERNS = {
    "museum": re.compile(
        r"\b(?:(?:[A-Z][A-Za-z'’-]*\s+){0,4}Museum(?:'s|\s+of(?:\s+[A-Z][A-Za-z'’-]*){1,4})?)\b"
    ),
    "gallery": re.compile(
        r"\b(?:(?:[A-Z][A-Za-z'’-]*\s+){0,4}Gallery(?:'s|\s+of(?:\s+[A-Z][A-Za-z'’-]*){1,4})?)\b"
    ),
}


def _source_operand_candidate(
    index: MemoryIndex,
    operand: str,
    analysis: QueryAnalysis,
    signals: RetrievalSignals,
) -> tuple[str, SourceTurn] | None:
    operand_tokens = claim_tokens(operand)
    specific = operand_tokens - _TEMPORAL_OPERAND_GENERIC
    candidates: list[tuple[float, str, tuple, SourceTurn]] = []
    for turn in index.turns:
        if turn.role not in analysis.source_roles:
            continue
        if not _not_after_anchor(turn.session_date, analysis):
            continue
        turn_tokens = claim_tokens(turn.content)
        if specific and not specific & turn_tokens:
            continue
        lexical = _overlap(operand_tokens, turn_tokens)
        if lexical <= 0.0:
            continue
        event_date = _turn_event_date(turn, operand)
        if event_date is None:
            continue
        score = signals.turn_scores.get(turn.turn_id, 0.0) + lexical
        if _SOURCE_EVENT_ACTION_RE.search(turn.content):
            score += 0.2
        candidates.append((score, event_date, turn.sort_key, turn))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3].turn_id))
    _, event_date, _, turn = candidates[0]
    return event_date, turn


def _execute_source_timeline(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    signals: RetrievalSignals,
) -> CompiledEvidence:
    query_lower = analysis.query.lower()
    heads = [head for head in _VENUE_PATTERNS if re.search(rf"\b{head}s?\b", query_lower)]
    if not heads:
        return CompiledEvidence(operation="temporal/order")
    query_categories = most_specific_categories(infer_category_ids(analysis.query))
    query_tokens = claim_tokens(analysis.query)
    by_entity: dict[str, tuple[float, str, SourceTurn, str]] = {}
    for turn in index.turns:
        if turn.role not in analysis.source_roles or not _SOURCE_EVENT_ACTION_RE.search(turn.content):
            continue
        if not _not_after_anchor(turn.session_date, analysis):
            continue
        if query_categories and not query_categories & set(infer_category_ids(turn.content)):
            continue
        labels = _timeline_entity_labels(turn.content, heads)
        for label in labels:
            event_date = _turn_event_date(turn, label)
            if event_date is None:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
            score = signals.turn_scores.get(turn.turn_id, 0.0) + _overlap(query_tokens, claim_tokens(turn.content))
            prior = by_entity.get(key)
            if prior is None or event_date < prior[1] or (event_date == prior[1] and score > prior[0]):
                by_entity[key] = (score, event_date, turn, label)
    expected = _expected_timeline_count(analysis.query)
    ranked = sorted(by_entity.values(), key=lambda item: (-item[0], item[1], item[3].lower()))
    if expected:
        ranked = ranked[:expected]
    selected = sorted(ranked, key=lambda item: (item[1], item[2].sort_key, item[3].lower()))
    rows = [
        {
            "position": position,
            "date": event_date,
            "entity": label,
            "event": turn.content,
            "source": turn.turn_id,
            "evidence_kind": "source_timeline",
        }
        for position, (_, event_date, turn, label) in enumerate(selected, start=1)
    ]
    complete = bool(selected) and (expected is None or len(selected) == expected)
    satisfied = ["source_provenance", "entity_compatibility", "event_time_resolution"] if selected else []
    if complete and len(selected) >= 2:
        satisfied.extend(["operand_coverage", "all_operands_found"])
    return CompiledEvidence(
        operation="temporal/order",
        rows=rows,
        candidate_answer=" -> ".join(item[3] for item in selected) if len(selected) >= 2 else None,
        satisfied_requirements=satisfied,
        selected_turn_ids=[item[2].turn_id for item in selected],
        confidence=0.9 if complete and len(selected) >= 2 else (0.62 if selected else 0.0),
        metadata={
            "source_timeline": True,
            "entity_heads": heads,
            "expected_count": expected,
            "matched_entities": len(selected),
        },
    )


def _timeline_entity_labels(text: str, heads: list[str]) -> list[str]:
    labels: list[str] = []
    parts = re.split(r"(?<=[.!?])\s+", text)
    for position, part in enumerate(parts):
        if not _SOURCE_EVENT_ACTION_RE.search(part):
            continue
        local = _venue_labels_in_text(part, heads)
        if not local:
            for prior in reversed(parts[:position]):
                local = _venue_labels_in_text(prior, heads)
                if local:
                    break
        labels.extend(local)
    return list(dict.fromkeys(labels))


def _venue_labels_in_text(text: str, heads: list[str]) -> list[str]:
    labels = []
    for head in heads:
        for match in _VENUE_PATTERNS[head].finditer(text):
            label = re.sub(r"'s$", "", match.group(0).strip())
            if label.lower() not in {head, f"the {head}"}:
                labels.append(label)
    return labels


def _expected_timeline_count(query: str) -> int | None:
    match = re.search(
        r"\b(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
        r"\s+(?:[a-z]+\s+){0,2}(?:museum|gallery|event|place|trip|visit)s?\b",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group("count").lower()
    return int(value) if value.isdigit() else _COUNT_WORDS.get(value)


def _turn_event_date(turn: SourceTurn, subject: str) -> str | None:
    refs = extract_temporal_refs(turn.content, anchor_time=turn.session_date)
    if refs:
        subject_tokens = claim_tokens(subject) - _TEMPORAL_OPERAND_GENERIC
        start_intent = bool(re.search(r"\b(?:start|started|begin|began|beginning)\b", subject, re.IGNORECASE))
        scored = []
        for ref in refs:
            value = ref.resolved_start if start_intent else (ref.resolved_end or ref.resolved_start)
            if not value:
                continue
            sentence = next(
                (
                    part
                    for part in re.split(r"(?<=[.!?])\s+", turn.content)
                    if ref.text.lower() in part.lower()
                ),
                turn.content,
            )
            same_sentence = bool(subject_tokens & claim_tokens(sentence))
            action_sentence = bool(_SOURCE_EVENT_ACTION_RE.search(sentence))
            score = float(ref.confidence) + (0.45 if same_sentence else 0.0) + (0.2 if action_sentence else 0.0)
            scored.append((score, value))
        if scored:
            return max(scored, key=lambda item: (item[0], item[1]))[1]
    observed = _parse_date(turn.session_date)
    return observed.isoformat() if observed else None


def _execute_lookup(index: MemoryIndex, analysis: QueryAnalysis, signals: RetrievalSignals) -> CompiledEvidence:
    base_query_tokens = set(analysis.entity_terms) or claim_tokens(analysis.query)
    base_bridge_tokens = significant_tokens(analysis.query)
    q_tokens = set(base_query_tokens)
    bridge_tokens: set[str] = set()
    for variant in expanded_query_variants(analysis.query):
        variant_tokens = claim_tokens(variant)
        bridge_tokens.update(significant_tokens(variant) - base_bridge_tokens)
        q_tokens.update(variant_tokens)
    query_categories = most_specific_categories(infer_category_ids(analysis.query))
    scored = []
    temporal_query = bool(analysis.time_start or analysis.time_end)
    relative_query = bool(re.search(r"\b(?:ago|last|previous|yesterday)\b", analysis.query, re.IGNORECASE))
    for record in index.claims:
        if record.claim.metadata.get("role") not in analysis.source_roles:
            continue
        if not _not_after_anchor(record.claim.observed_at, analysis):
            continue
        lexical = _overlap(q_tokens, claim_tokens(record.text))
        dense = signals.claim_scores.get(record.claim_id, 0.0)
        temporal_distance = _claim_window_distance(record, analysis)
        if temporal_query and temporal_distance is None:
            continue
        if temporal_query and temporal_distance > (1 if relative_query else 0):
            continue
        temporal_score = (
            1.0
            if temporal_query and relative_query and temporal_distance <= 1
            else (1.0 / (1.0 + float(temporal_distance or 0)) if temporal_query else 0.0)
        )
        category_match = bool(query_categories & set(infer_category_ids(record.text)))
        bridge_hits = len(bridge_tokens & significant_tokens(record.text))
        bridge_score = min(0.36, 0.12 * bridge_hits)
        score = (
            (0.54 * dense)
            + (0.24 * lexical)
            + (0.3 * temporal_score)
            + (0.28 if category_match else 0.0)
            + bridge_score
        )
        if score > 0:
            scored.append((score, record.sort_key, record, bridge_hits, category_match, temporal_distance))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].claim_id))
    picked = [item[2] for item in scored[:6]]
    rows = [
        {
            "claim": item[2].text,
            "date": item[2].claim.observed_at,
            "sources": ",".join(item[2].source_turn_ids),
            "concept_bridge_hits": item[3],
            "temporal_distance_days": item[5] if temporal_query else None,
        }
        for item in scored[:6]
    ]
    candidate = None
    if (
        scored
        and relative_query
        and scored[0][3] >= 2
        and scored[0][4]
        and scored[0][5] is not None
        and scored[0][5] <= 1
    ):
        candidate = scored[0][2].text
    return CompiledEvidence(
        operation="lookup",
        rows=rows,
        candidate_answer=candidate,
        satisfied_requirements=["source_provenance", "entity_compatibility"] if picked else [],
        selected_claim_ids=[record.claim_id for record in picked],
        selected_turn_ids=[turn_id for record in picked for turn_id in record.source_turn_ids],
        confidence=min(0.9, scored[0][0] + 0.1) if scored else 0.0,
        metadata={
            "temporal_query": temporal_query,
            "relative_tolerance_days": 1 if relative_query else 0,
            "query_categories": sorted(query_categories),
            "bridge_tokens": sorted(bridge_tokens),
            "candidate_rule": "relative_time+category+two_concept_bridges" if candidate else None,
        },
    )


def _core_terms_covered(query_tokens: set[str], events) -> bool:
    core_tokens = query_tokens - _AGGREGATE_GENERIC_TERMS
    if not core_tokens:
        return True
    event_tokens = set().union(*(claim_tokens(event.summary) for event in events)) if events else set()
    return len(core_tokens & event_tokens) >= min(2, len(core_tokens))


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
    return most_specific_categories(specific or categories)


def _event_category_items(event, category_constraints: set[str]) -> list[str]:
    return sorted(
        {
            mention
            for category_id in category_constraints
            for mention in category_mentions(event.summary, category_id)
        }
    )


def _event_cardinality(
    event,
    category_constraints: set[str],
    query_tokens: set[str],
    *,
    query_predicate: str,
) -> int:
    for quantity in event.quantities:
        value = quantity.get("value")
        if value is None or quantity.get("unit") == "usd":
            continue
        unit_tokens = claim_tokens(
            f"{quantity.get('unit') or ''} {quantity.get('property_name') or ''}"
        )
        if unit_tokens & query_tokens:
            return max(1, int(float(value)))
    if query_predicate not in {"purchase", "acquire", "donate"}:
        return 1
    items = _event_category_items(event, category_constraints)
    return max(1, len(items))


def _event_for_source_roles(index: MemoryIndex, event, source_roles: set[str]):
    """Return a role-scoped canonical event without cross-role evidence leakage."""

    records = [
        index.by_claim_id[claim_id]
        for claim_id in event.claim_ids
        if claim_id in index.by_claim_id
        and index.by_claim_id[claim_id].claim.metadata.get("role") in source_roles
    ]
    if not records:
        return None
    quantities = []
    seen_quantities: set[tuple] = set()
    for record in records:
        for quantity in record.quantity_claims:
            signature = (quantity.value, quantity.unit, quantity.property_name)
            if signature in seen_quantities:
                continue
            seen_quantities.add(signature)
            quantities.append(quantity.model_dump())
    source_turn_ids = tuple(
        dict.fromkeys(
            turn_id
            for record in records
            for turn_id in record.source_turn_ids
            if turn_id in index.by_turn_id and index.by_turn_id[turn_id].role in source_roles
        )
    )
    categories = tuple(
        dict.fromkeys(
            category_id
            for record in records
            for category_id in infer_category_ids(record.text)
        )
    )
    return replace(
        event,
        summary=records[0].text,
        category_ids=categories,
        claim_ids=tuple(record.claim_id for record in records),
        source_turn_ids=source_turn_ids,
        quantities=tuple(quantities),
        metadata={**event.metadata, "roles": sorted(source_roles)},
    )


def _aggregate_source_coverage(
    index: MemoryIndex,
    analysis: QueryAnalysis,
    selected,
    signals: RetrievalSignals,
    *,
    q_tokens: set[str],
    category_constraints: set[str],
    query_predicate: str,
) -> SourceCoverageAudit:
    selected_ids = {turn_id for event in selected for turn_id in event.source_turn_ids}
    candidates: set[str] = set()
    dense_floor = max(signals.turn_scores.values(), default=0.0) * 0.7
    for turn in index.turns:
        if turn.role not in analysis.source_roles:
            continue
        if not _in_window(turn.session_date, analysis.time_start, analysis.time_end):
            continue
        if analysis.completed_only and _NON_ACTUAL_SOURCE_RE.search(turn.content):
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
        lexical = _overlap(q_tokens, claim_tokens(turn.content))
        dense = signals.turn_scores.get(turn.turn_id, 0.0)
        if lexical <= 0.0 and (dense_floor <= 0.0 or dense < dense_floor):
            continue
        candidates.add(turn.turn_id)
    if not candidates and selected_ids:
        candidates.update(selected_ids)
    extracted_ids = {
        turn_id
        for record in index.claims
        if record.claim.metadata.get("role") in analysis.source_roles
        and not (analysis.completed_only and _NON_ACTUAL_SOURCE_RE.search(record.text))
        and (query_predicate == "state" or predicates_compatible(query_predicate, infer_predicate(record.text)))
        and (
            not category_constraints
            or category_constraints & set(infer_category_ids(record.text))
        )
        for turn_id in record.source_turn_ids
    }
    compiled_ids = {
        turn_id
        for event in index.compiled.canonical_events
        if event.modality == "actual"
        and (query_predicate == "state" or predicates_compatible(query_predicate, event.predicate))
        for turn_id in event.source_turn_ids
    }
    recovery = sorted(
        candidates - selected_ids,
        key=lambda turn_id: (
            -signals.turn_scores.get(turn_id, 0.0),
            -_overlap(q_tokens, claim_tokens(index.by_turn_id[turn_id].content)),
            turn_id,
        ),
    )[:12]
    return audit_source_coverage(
        candidates,
        extracted_source_ids=extracted_ids,
        compiled_source_ids=compiled_ids,
        selected_source_ids=selected_ids,
        recovery_source_ids=recovery,
        minimum_ratio=0.9,
        metadata={"operator": analysis.primary_operator},
    )


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


def _claim_window_distance(record, analysis: QueryAnalysis) -> int | None:
    start = _parse_date(analysis.time_start)
    end = _parse_date(analysis.time_end)
    if start is None and end is None:
        return 0
    dates = [
        parsed
        for parsed in (
            _parse_date(record.claim.observed_at),
            *(
                _parse_date(ref.resolved_start or ref.resolved_end)
                for ref in record.temporal_refs
            ),
        )
        if parsed is not None
    ]
    if not dates:
        return None
    lower = start or end
    upper = end or start
    assert lower is not None and upper is not None
    distances = [
        0 if lower <= candidate <= upper else min(abs((candidate - lower).days), abs((candidate - upper).days))
        for candidate in dates
    ]
    return min(distances)


def _overlap(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def _evidence_excerpt(text: str, limit: int = 600) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3].rstrip() + "..."
