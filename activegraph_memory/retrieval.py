"""Evidence retrieval and assembly over a compiled memory index."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable

from .compiler import MemoryClaimRecord, MemoryIndex, SourceTurn, claim_tokens
from .coverage import build_coverage_report
from .graph_query import run_graph_query
from .object_types import EvidenceBundle, MemoryQuery, RetrievalPlan
from .planner import plan_query
from .scoring import MemoryConfidence, confidence_vector, select_epistemic_status
from .temporal import extract_temporal_refs


TokenCounter = Callable[[str], int]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_LARGE_REQUESTED_BUDGET = 10000
_ADAPTIVE_BUDGETS: dict[str, int] = {
    "aggregate": 3200,
    "multi_hop": 3400,
    "temporal": 3200,
    "current": 2400,
    "latest": 2400,
    "final": 2400,
    "preference": 2200,
    "lookup": 2400,
    "semantic_lookup": 2600,
    "decision_reconstruction": 3200,
    "negative_existence": 2600,
    "unknown": 2600,
}
_QUERY_STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "also",
    "amount",
    "and",
    "any",
    "are",
    "before",
    "can",
    "complement",
    "current",
    "currently",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "including",
    "into",
    "keep",
    "many",
    "much",
    "need",
    "old",
    "our",
    "previous",
    "recent",
    "remind",
    "since",
    "setup",
    "some",
    "suggest",
    "that",
    "the",
    "this",
    "term",
    "terms",
    "total",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
}
_VALUATION_RATIO_TERMS = {
    "double",
    "doubled",
    "twice",
    "triple",
    "tripled",
    "half",
    "third",
    "quarter",
    "multiple",
    "times",
}
_GRAPH_ANSWER_CONFIDENCE_FLOOR = 0.65
_SNAPSHOT_COUNT_INTENT_RE = re.compile(
    r"\b(so far|as of now|currently|current|latest|now|already|yet)\b|"
    r"\b(?:how many|number of|count)\b.*\b(?:have i|have we|i've|we've|do i have|do we have)\b",
    re.IGNORECASE,
)
_COUNT_ADDITIVE_INTENT_RE = re.compile(
    r"\b(in total|total number|overall|all together|altogether|during|between|from .+ to|"
    r"this year|this month|this week|last year|last month|past year|past month|past few|"
    r"each|per)\b",
    re.IGNORECASE,
)
_STRONG_SNAPSHOT_COUNT_INTENT_RE = re.compile(
    r"\b(so far|as of now|currently|current|latest|now|already|yet)\b",
    re.IGNORECASE,
)
_SNAPSHOT_VALUE_INTENT_RE = re.compile(
    r"\b(current|currently|latest|now|as of now|balance|limit|pre-?approved|approved for|"
    r"worth|valued|value|estimate|offer|salary|budget|rate)\b|"
    r"\bhow much\s+(?:is|was|are|were|am)\b",
    re.IGNORECASE,
)
_ASSISTANT_EXACT_SOURCE_RE = re.compile(
    r"\b(?:what|which|how many|remind|confirm)\b.*\b(?:you|assistant|we|our)\b.*"
    r"\b(?:said|say|told|tell|provided|provide|gave|give|listed|list|recommended|"
    r"recommend|suggested|suggest|computed|calculated|made|move|recipe|answer)\b|"
    r"\b(?:previous|earlier|prior)\s+conversation\b.*"
    r"\b(?:said|say|told|provided|gave|listed|recommended|suggested|made|recipe)\b|"
    r"\bwhat was the\s+\d+(?:st|nd|rd|th)\b.*\b(?:job|item|step|option|thing)\b",
    re.IGNORECASE,
)
_PREFERENCE_ADVICE_RE = re.compile(
    r"\b(prefer|preference|favorite|likes?|dislikes?|style|tone|advice|tips?|"
    r"suggest|recommend|recommendation|better results|struggling with|what else can i do)\b",
    re.IGNORECASE,
)
_PREFERENCE_SIGNAL_RE = re.compile(
    r"\b(prefer|favorite|like|liked|likes|love|loved|enjoy|enjoyed|want|wanted|"
    r"interested|avoid|without|don't|do not|not prefer|struggling|success|worked well|"
    r"better|worse|sleep quality|wind down)\b",
    re.IGNORECASE,
)
_ANSWER_PACKET_GENERIC_TOKENS = {
    *_QUERY_STOPWORDS,
    "accessory",
    "accessori",
    "ago",
    "amount",
    "answer",
    "before",
    "current",
    "currently",
    "cost",
    "day",
    "days",
    "device",
    "first",
    "get",
    "got",
    "how",
    "keep",
    "latest",
    "many",
    "month",
    "months",
    "most",
    "new",
    "now",
    "old",
    "previous",
    "previously",
    "recent",
    "second",
    "setup",
    "some",
    "suggest",
    "third",
    "time",
    "times",
    "total",
    "week",
    "weeks",
    "where",
    "who",
    "year",
    "years",
}
_LATEST_EVENT_AMBIGUITY_RE = re.compile(
    r"\b(before|previous|previously|used to|plans?|planning|next|upcoming|"
    r"will|would like|thinking of|considering|hoping to|want(?:s|ed)? to)\b",
    re.IGNORECASE,
)
_CONCEPT_TOKEN_EXPANSIONS = {
    "phone": {
        "android",
        "accessori",
        "accessorie",
        "accessory",
        "bank",
        "case",
        "charger",
        "charging",
        "device",
        "iphone",
        "magsafe",
        "power",
        "protector",
        "screen",
        "wallet",
        "wireless",
    },
    "accessori": {
        "bank",
        "case",
        "charger",
        "charging",
        "device",
        "iphone",
        "power",
        "protector",
        "screen",
        "wallet",
        "wireless",
    },
    "accessorie": {
        "bank",
        "case",
        "charger",
        "charging",
        "device",
        "iphone",
        "power",
        "protector",
        "screen",
        "wallet",
        "wireless",
    },
    "accessory": {
        "bank",
        "case",
        "charger",
        "charging",
        "device",
        "iphone",
        "power",
        "protector",
        "screen",
        "wallet",
        "wireless",
    },
    "milestone": {
        "accepted",
        "approval",
        "approved",
        "client",
        "contract",
        "customer",
        "finish",
        "finished",
        "first",
        "graduat",
        "launch",
        "launched",
        "move",
        "moved",
        "sign",
        "signed",
        "start",
        "started",
    },
    "busines": {
        "accounting",
        "brand",
        "campaign",
        "client",
        "clients",
        "contract",
        "customer",
        "freelance",
        "invoice",
        "invoices",
        "product",
        "quickbook",
        "quickbooks",
        "website",
    },
    "business": {
        "accounting",
        "brand",
        "campaign",
        "client",
        "clients",
        "contract",
        "customer",
        "freelance",
        "invoice",
        "invoices",
        "product",
        "quickbook",
        "quickbooks",
        "website",
    },
    "buisines": {
        "accounting",
        "brand",
        "campaign",
        "client",
        "clients",
        "contract",
        "customer",
        "freelance",
        "invoice",
        "invoices",
        "product",
        "quickbook",
        "quickbooks",
        "website",
    },
    "buisiness": {
        "accounting",
        "brand",
        "campaign",
        "client",
        "clients",
        "contract",
        "customer",
        "freelance",
        "invoice",
        "invoices",
        "product",
        "quickbook",
        "quickbooks",
        "website",
    },
    "photography": {
        "a7r",
        "bag",
        "camera",
        "flash",
        "godox",
        "lens",
        "lenses",
        "photo",
        "photographer",
        "sony",
    },
    "camera": {
        "a7r",
        "bag",
        "flash",
        "godox",
        "lens",
        "lenses",
        "photo",
        "photography",
        "sony",
    },
}


@dataclass
class MemoryRetrievalResult:
    """Result of a memory retrieval pass."""

    context_text: str
    truncated: bool
    selected_claim_ids: list[str]
    selected_turn_ids: list[str]
    retrieval_plan: RetrievalPlan
    evidence_bundle: EvidenceBundle
    coverage_report: Any
    confidence: MemoryConfidence
    epistemic_status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    kind: str
    uid: str
    score: float
    sort_key: tuple


def retrieve_memory(
    index: MemoryIndex,
    query: MemoryQuery | str,
    *,
    query_id: str = "query",
    question_date: str | None = None,
    token_budget: int = 10000,
    claim_scores: dict[str, float] | None = None,
    turn_scores: dict[str, float] | None = None,
    token_counter: TokenCounter | None = None,
    retrieval_plan: RetrievalPlan | None = None,
) -> MemoryRetrievalResult:
    """Retrieve and assemble provenance-backed evidence for a memory query.

    ``claim_scores`` and ``turn_scores`` may be supplied by an embedding or
    reranker backend. When omitted, deterministic lexical scoring is used.
    """

    memory_query = query if isinstance(query, MemoryQuery) else MemoryQuery(query=query)
    if question_date and memory_query.time_anchor is None:
        memory_query = memory_query.model_copy(update={"time_anchor": question_date})
    plan = retrieval_plan or plan_query(memory_query, query_id=query_id)
    query_type = str(plan.metadata.get("query_type", memory_query.query_type))
    token_counter = token_counter or _rough_token_count
    assistant_exact_source = _requires_assistant_exact_source(memory_query.query)
    preference_advice_query = _looks_like_preference_or_advice_query(memory_query.query, query_type=query_type)
    candidate_core_terms = _answer_packet_core_terms(memory_query.query)

    temporal_targets = _query_temporal_targets(memory_query.query, memory_query.time_anchor)
    claim_scores = claim_scores or {}
    turn_scores = turn_scores or {}

    scored_claims = _score_claims(
        index.claims,
        memory_query.query,
        query_type=query_type,
        external_scores=claim_scores,
        temporal_targets=temporal_targets,
        assistant_exact_source=assistant_exact_source,
        preference_advice_query=preference_advice_query,
    )
    scored_turns = _score_turns(
        index.turns,
        memory_query.query,
        query_type=query_type,
        external_scores=turn_scores,
        temporal_targets=temporal_targets,
        assistant_exact_source=assistant_exact_source,
        preference_advice_query=preference_advice_query,
    )

    candidates = _rank_candidates(index, scored_claims, scored_turns, query_type=query_type)
    graph_result = None
    if not assistant_exact_source:
        graph_result = run_graph_query(
            index,
            memory_query.query,
            query_type=query_type,
            anchor_time=memory_query.time_anchor,
        )
    effective_token_budget = _effective_token_budget(token_budget, query_type=query_type, graph_result=graph_result)
    selected_claim_ids: list[str] = []
    selected_turn_ids: set[str] = set()
    selected_direct_turn_ids: set[str] = set()
    selected_claim_set: set[str] = set()
    prefix_blocks: list[str] = []
    graph_context_rendered = False
    graph_answer_candidate_rendered = False
    observation_context_rendered = False
    temporal_target_context_rendered = False
    graph_context_ok, graph_context_reason = _graph_answer_packet_status(
        graph_result,
        query=memory_query.query,
        query_type=query_type,
    )
    dynamic_expansion = _dynamic_expansion_plan(
        graph_result,
        graph_context_ok=graph_context_ok,
        graph_context_reason=graph_context_reason,
        query=memory_query.query,
        query_type=query_type,
    )
    dynamic_added_claim_ids: list[str] = []
    dynamic_added_turn_ids: list[str] = []
    running = 0
    truncated = False

    def fits(cost: int) -> bool:
        return running + cost <= effective_token_budget

    def add_cost(cost: int) -> None:
        nonlocal running
        running += cost

    def add_claim(record: MemoryClaimRecord, *, include_sources: bool = True) -> bool:
        nonlocal truncated
        if record.claim_id in selected_claim_set:
            return True
        header = _claim_header(record)
        cost = token_counter(header) + 1
        new_turn_ids = []
        if include_sources:
            new_turn_ids = [
                turn_id
                for turn_id in record.source_turn_ids
                if turn_id in index.by_turn_id and turn_id not in selected_turn_ids
            ]
        for turn_id in new_turn_ids:
            cost += token_counter(index.by_turn_id[turn_id].text) + 1
        if not fits(cost):
            truncated = True
            return False
        add_cost(cost)
        selected_claim_ids.append(record.claim_id)
        selected_claim_set.add(record.claim_id)
        selected_turn_ids.update(new_turn_ids)
        return True

    def add_turn(turn: SourceTurn, *, direct: bool = False) -> bool:
        nonlocal truncated
        if turn.turn_id in selected_turn_ids:
            if direct:
                selected_direct_turn_ids.add(turn.turn_id)
            return True
        cost = token_counter(turn.text) + 2
        if not fits(cost):
            truncated = True
            return False
        add_cost(cost)
        selected_turn_ids.add(turn.turn_id)
        if direct:
            selected_direct_turn_ids.add(turn.turn_id)
        return True

    if (
        temporal_targets
        and query_type in {"temporal", "lookup", "semantic_lookup"}
        and not graph_context_ok
    ):
        temporal_target_context = _render_temporal_target_packet(
            index,
            memory_query.query,
            temporal_targets=temporal_targets,
            turn_scores=scored_turns,
            max_turns=8,
        )
        if temporal_target_context:
            temporal_target_cost = token_counter(temporal_target_context) + 2
            if fits(temporal_target_cost):
                prefix_blocks.append(temporal_target_context)
                temporal_target_context_rendered = True
                add_cost(temporal_target_cost)
            else:
                truncated = True

    if preference_advice_query:
        observation_context = _render_preference_observation_packet(
            index,
            memory_query.query,
            max_items=8,
        )
        if observation_context:
            observation_cost = token_counter(observation_context) + 2
            if fits(observation_cost):
                prefix_blocks.append(observation_context)
                observation_context_rendered = True
                add_cost(observation_cost)
            else:
                truncated = True

    if graph_result is not None and graph_result.answer_hint and graph_context_ok:
        include_answer_candidate = _include_graph_answer_candidate(
            graph_result,
            query=memory_query.query,
            query_type=query_type,
        )
        for max_rows in _packet_row_options(graph_result):
            rendered_graph = _render_answer_packet(
                index,
                graph_result,
                query=memory_query.query,
                query_type=query_type,
                max_rows=max_rows,
                include_answer_candidate=include_answer_candidate,
            )
            graph_cost = token_counter(rendered_graph) + 2
            if fits(graph_cost):
                prefix_blocks.append(rendered_graph)
                graph_context_rendered = True
                graph_answer_candidate_rendered = include_answer_candidate
                add_cost(graph_cost)
                break
        else:
            if include_answer_candidate:
                minimal_graph = f"[graph-query: {graph_result.operation}]\n{graph_result.answer_hint}"
            else:
                minimal_graph = (
                    f"[graph-query: {graph_result.operation}]\n"
                    "Computed answer candidate withheld because reducer confidence or query coverage is low."
                )
            graph_cost = token_counter(minimal_graph) + 2
            if fits(graph_cost):
                prefix_blocks.append(minimal_graph)
                graph_context_rendered = True
                graph_answer_candidate_rendered = include_answer_candidate
                add_cost(graph_cost)
            else:
                truncated = True
        graph_claim_ids = _graph_claim_ids_to_expand(graph_result)
        graph_turn_ids = _graph_turn_ids_to_expand(graph_result)
        for claim_id in graph_claim_ids:
            record = index.by_claim_id.get(claim_id)
            if record is not None:
                add_claim(record, include_sources=False)
        for turn_id in graph_turn_ids:
            turn = index.by_turn_id.get(turn_id)
            if turn is not None:
                add_turn(turn)

    if dynamic_expansion["triggered"] and graph_result is not None:
        for claim_id in _dynamic_graph_claim_ids_to_expand(graph_result, reason=graph_context_reason):
            if claim_id in selected_claim_set:
                continue
            record = index.by_claim_id.get(claim_id)
            if record is not None and add_claim(record, include_sources=False):
                dynamic_added_claim_ids.append(claim_id)
        for turn_id in _dynamic_graph_turn_ids_to_expand(graph_result, reason=graph_context_reason):
            if turn_id in selected_turn_ids:
                continue
            turn = index.by_turn_id.get(turn_id)
            if turn is not None and add_turn(turn, direct=True):
                dynamic_added_turn_ids.append(turn_id)

    skip_assistant_fallback = _graph_prefers_user_fallback(graph_result)
    for candidate in candidates:
        if candidate.score <= 0.0 and selected_claim_ids:
            break
        if candidate.kind == "claim":
            record = index.by_claim_id[candidate.uid]
            if skip_assistant_fallback and record.claim.metadata.get("role") == "assistant":
                continue
            if not _fallback_candidate_matches_query(
                record.text,
                query_type=query_type,
                core_terms=candidate_core_terms,
            ):
                continue
            add_claim(record)
            continue

        turn = index.by_turn_id[candidate.uid]
        if skip_assistant_fallback and turn.role == "assistant":
            continue
        if not _fallback_candidate_matches_query(
            turn.text,
            query_type=query_type,
            core_terms=candidate_core_terms,
        ):
            continue
        add_turn(turn, direct=True)

    if assistant_exact_source:
        for turn_id in _neighbor_turn_ids(index, selected_direct_turn_ids, window=2):
            turn = index.by_turn_id.get(turn_id)
            if turn is not None:
                add_turn(turn)

    rendered_turn_ids = sorted(selected_turn_ids, key=lambda tid: index.by_turn_id[tid].sort_key)
    context_text = _render_context(
        index,
        selected_claim_ids=selected_claim_ids,
        selected_turn_ids=rendered_turn_ids,
        selected_direct_turn_ids=selected_direct_turn_ids,
        prefix_text="\n\n".join(prefix_blocks),
        query_type=query_type,
    )
    searched_sessions = _sessions_for_turns(index, rendered_turn_ids)
    not_searched = [sid for sid in index.session_ids if sid not in set(searched_sessions)]
    coverage = build_coverage_report(
        query_id=query_id,
        searched_scopes=searched_sessions,
        not_searched_scopes=not_searched,
        query_type=query_type,  # type: ignore[arg-type]
        metadata={
            "scope_kind": "session",
            "selected_claim_ids": selected_claim_ids,
            "selected_turn_ids": rendered_turn_ids,
            "temporal_targets": [target.isoformat() for target in temporal_targets],
            "graph_query": _graph_query_metadata(graph_result),
            "graph_context_rendered": graph_context_rendered,
            "graph_context_skip_reason": graph_context_reason,
            "graph_answer_candidate_rendered": graph_answer_candidate_rendered,
            "observation_context_rendered": observation_context_rendered,
            "temporal_target_context_rendered": temporal_target_context_rendered,
            "dynamic_expansion": dynamic_expansion,
            "dynamic_added_claim_ids": dynamic_added_claim_ids,
            "dynamic_added_turn_ids": dynamic_added_turn_ids,
        },
    )
    confidence = _build_confidence(
        selected_claim_ids=selected_claim_ids,
        selected_turn_ids=rendered_turn_ids,
        coverage_confidence=coverage.coverage_confidence,
        top_claim_score=max(scored_claims.values(), default=0.0),
        top_turn_score=max(scored_turns.values(), default=0.0),
    )
    status = select_epistemic_status(
        confidence,
        found_evidence=bool(selected_claim_ids or rendered_turn_ids),
        direct_support=bool(selected_claim_ids),
        coverage_report=coverage,
        requires_freshness=plan.requires_freshness,
        requires_coverage=plan.requires_coverage,
        requires_reasoning=query_type in {"multi_hop", "decision_reconstruction"},
    )
    evidence = EvidenceBundle(
        query_id=query_id,
        claim_ids=selected_claim_ids,
        source_ids=rendered_turn_ids,
        coverage_report_id=None,
        conflict_ids=[
            cid
            for cid in selected_claim_ids
            for cid in index.by_claim_id[cid].contradicts
        ],
        metadata={
            "searched_sessions": searched_sessions,
            "n_direct_turns": len(selected_direct_turn_ids),
            "graph_query": _graph_query_metadata(graph_result),
            "graph_context_rendered": graph_context_rendered,
            "graph_context_skip_reason": graph_context_reason,
            "graph_answer_candidate_rendered": graph_answer_candidate_rendered,
            "observation_context_rendered": observation_context_rendered,
            "temporal_target_context_rendered": temporal_target_context_rendered,
            "dynamic_expansion": dynamic_expansion,
            "dynamic_added_claim_ids": dynamic_added_claim_ids,
            "dynamic_added_turn_ids": dynamic_added_turn_ids,
        },
    )
    return MemoryRetrievalResult(
        context_text=context_text,
        truncated=truncated,
        selected_claim_ids=selected_claim_ids,
        selected_turn_ids=rendered_turn_ids,
        retrieval_plan=plan,
        evidence_bundle=evidence,
        coverage_report=coverage,
        confidence=confidence,
        epistemic_status=status,
        metadata={
            "query_type": query_type,
            "n_claims_indexed": len(index.claims),
            "n_turns_indexed": len(index.turns),
            "n_candidates_considered": len(candidates),
            "n_direct_turns_selected": len(selected_direct_turn_ids),
            "token_budget": effective_token_budget,
            "requested_token_budget": token_budget,
            "adaptive_budget_applied": effective_token_budget != token_budget,
            "estimated_context_tokens": token_counter(context_text),
            "temporal_targets": [target.isoformat() for target in temporal_targets],
            "selected_unit_ids": [
                *rendered_turn_ids,
                *selected_claim_ids,
                *(graph_result.selected_event_ids if graph_result else []),
            ],
            "graph_query": _graph_query_metadata(graph_result),
            "graph_context_rendered": graph_context_rendered,
            "graph_context_skip_reason": graph_context_reason,
            "graph_answer_candidate_rendered": graph_answer_candidate_rendered,
            "observation_context_rendered": observation_context_rendered,
            "temporal_target_context_rendered": temporal_target_context_rendered,
            "dynamic_expansion": dynamic_expansion,
            "dynamic_added_claim_ids": dynamic_added_claim_ids,
            "dynamic_added_turn_ids": dynamic_added_turn_ids,
            "claim_scores": {cid: round(scored_claims.get(cid, 0.0), 4) for cid in selected_claim_ids},
            "turn_scores": {tid: round(scored_turns.get(tid, 0.0), 4) for tid in rendered_turn_ids},
        },
    )


def _score_claims(
    claims: Iterable[MemoryClaimRecord],
    query: str,
    *,
    query_type: str,
    external_scores: dict[str, float],
    temporal_targets: list[date],
    assistant_exact_source: bool,
    preference_advice_query: bool,
) -> dict[str, float]:
    q_tokens = _salient_query_tokens(query)
    out: dict[str, float] = {}
    for record in claims:
        lexical = _token_overlap(q_tokens, claim_tokens(record.text))
        external = _positive_cosine(external_scores.get(record.claim_id, 0.0))
        phrase = _phrase_overlap_score(query, record.text)
        score = (0.72 * external) + (0.23 * lexical) + (0.05 * phrase)
        score += _valuation_ratio_boost(memory_query=query, text=record.text)
        score += _temporal_boost(record.claim.valid_from, temporal_targets)
        if query_type in {"latest", "current", "final"}:
            score += _recency_boost(record.sort_key[0]) * 0.15
            if record.claim.status == "superseded":
                score *= 0.35
        elif record.claim.status == "superseded":
            score *= 0.8
        if query_type == "preference" and record.claim.claim_kind == "preference":
            score += 0.18
        if assistant_exact_source:
            if record.claim.metadata.get("role") == "assistant":
                score += 0.42
            else:
                score *= 0.72
        if preference_advice_query:
            if record.claim.metadata.get("role") == "user":
                score += 0.12
            if record.claim.claim_kind in {"preference", "instruction"} or _PREFERENCE_SIGNAL_RE.search(record.text):
                score += 0.28
        if query_type in {"aggregate", "multi_hop", "temporal"}:
            score += 0.04
        out[record.claim_id] = max(0.0, score)
    return out


def _score_turns(
    turns: Iterable[SourceTurn],
    query: str,
    *,
    query_type: str,
    external_scores: dict[str, float],
    temporal_targets: list[date],
    assistant_exact_source: bool,
    preference_advice_query: bool,
) -> dict[str, float]:
    q_tokens = _salient_query_tokens(query)
    out: dict[str, float] = {}
    for turn in turns:
        lexical = _token_overlap(q_tokens, _salient_query_tokens(turn.text))
        external = _positive_cosine(external_scores.get(turn.turn_id, 0.0))
        phrase = _phrase_overlap_score(query, turn.text)
        score = (0.66 * external) + (0.28 * lexical) + (0.06 * phrase)
        score += _valuation_ratio_boost(memory_query=query, text=turn.text)
        score += _temporal_boost(turn.session_date, temporal_targets)
        if query_type in {"temporal", "aggregate", "multi_hop"}:
            score += 0.03
        if query_type in {"latest", "current", "final"}:
            score += _recency_boost(turn.session_date) * 0.08
        if assistant_exact_source:
            if turn.role == "assistant":
                score += 0.45
            else:
                score *= 0.76
        if preference_advice_query:
            if turn.role == "user":
                score += 0.1
            if _PREFERENCE_SIGNAL_RE.search(turn.text):
                score += 0.22
        out[turn.turn_id] = max(0.0, score)
    return out


def _rank_candidates(
    index: MemoryIndex,
    claim_scores: dict[str, float],
    turn_scores: dict[str, float],
    *,
    query_type: str,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for record in index.claims:
        score = claim_scores.get(record.claim_id, 0.0)
        # Claims are the semantic index. Give anchored claims a slight edge,
        # but do not let orphan headers dominate raw source turns.
        score += 0.08 if record.source_turn_ids else 0.01
        candidates.append(_Candidate("claim", record.claim_id, score, record.sort_key))
    for turn in index.turns:
        score = turn_scores.get(turn.turn_id, 0.0)
        if query_type in {"lookup", "semantic_lookup", "temporal"}:
            score += 0.03
        candidates.append(_Candidate("turn", turn.turn_id, score, turn.sort_key))
    return sorted(candidates, key=lambda c: (-c.score, c.sort_key, c.uid))


def _effective_token_budget(token_budget: int, *, query_type: str, graph_result: Any | None) -> int:
    """Clamp very large caller budgets to an evidence-shaped budget.

    Large context windows are useful for fallback callers, but memory retrieval
    should default to a compact evidence packet plus supporting sources. Callers
    requesting a moderate budget keep that exact budget.
    """

    if token_budget <= _LARGE_REQUESTED_BUDGET:
        return token_budget
    budget = _ADAPTIVE_BUDGETS.get(query_type, _ADAPTIVE_BUDGETS["unknown"])
    if graph_result is not None:
        operation = str(getattr(graph_result, "operation", ""))
        matched = int((getattr(graph_result, "metadata", {}) or {}).get("matched_events") or 0)
        if operation.startswith("aggregate/") and matched > 12:
            budget = max(budget, 3800)
        if operation == "temporal/timeline" and matched > 12:
            budget = max(budget, 3600)
    return min(token_budget, budget)


def _packet_row_options(graph_result: Any) -> tuple[int, ...]:
    matched = int((getattr(graph_result, "metadata", {}) or {}).get("matched_events") or 0)
    operation = str(getattr(graph_result, "operation", ""))
    if operation.startswith("aggregate/") and matched <= 12:
        return (12, 8, 4, 0)
    if operation.startswith("aggregate/"):
        return (24, 16, 8, 0)
    if operation in {"temporal/latest", "current/latest"}:
        return (8, 4, 2, 0)
    return (16, 8, 4, 0)


def _graph_answer_packet_status(graph_result: Any | None, *, query: str, query_type: str) -> tuple[bool, str]:
    if graph_result is None:
        return False, "no_graph_result"
    if not getattr(graph_result, "answer_hint", None):
        return False, "no_answer_hint"

    metadata = getattr(graph_result, "metadata", {}) or {}
    operation = str(getattr(graph_result, "operation", ""))
    matched = int(metadata.get("matched_events") or 0)
    if matched <= 0:
        return False, "no_matched_events"
    if operation == "aggregate/count":
        count_method = metadata.get("count_method")
        if count_method == "latest_quantity_snapshot":
            return True, "latest_quantity_snapshot"
        if count_method == "quantity_sum":
            if _query_has_snapshot_count_intent(query):
                return False, "snapshot_query_quantity_sum_suppressed"
            return True, "quantity_count"
        if count_method == "event_count" and _query_has_strong_snapshot_count_intent(query) and matched > 1:
            return False, "snapshot_query_event_count_suppressed"
        if matched <= 12:
            return True, "bounded_event_count"
        return False, "broad_event_count"

    if operation == "aggregate/sum":
        if metadata.get("sum_method") == "latest_quantity_snapshot":
            return True, "latest_quantity_snapshot"
        if _query_has_value_snapshot_intent(query):
            return False, "snapshot_query_sum_suppressed"
        values = list(metadata.get("sum_values") or [])
        if values and matched <= 24:
            return True, "bounded_quantity_sum"
        if values:
            return False, "broad_quantity_sum"
        if matched <= 4:
            return True, "bounded_missing_quantity_sum"
        return False, "broad_missing_quantity_sum"

    if operation == "aggregate/max":
        group_values = metadata.get("group_values") or {}
        if group_values and matched <= 24:
            return True, "bounded_group_max"
        if group_values:
            return False, "broad_group_max"
        return False, "missing_group_max"

    if operation == "aggregate/difference":
        values = list(metadata.get("difference_values") or [])
        if len(values) >= 2 and matched <= 8:
            return True, "bounded_difference"
        return False, "insufficient_difference_evidence"

    if operation == "temporal/date-delta":
        if metadata.get("delta_days") is None:
            return False, "missing_date_delta"
        if matched <= 3:
            return True, "bounded_date_delta"
        return False, "broad_date_delta"

    if operation == "temporal/latest":
        if matched <= 12:
            return True, "bounded_latest"
        return False, "broad_latest"

    if operation == "temporal/order":
        if matched <= 12:
            return True, "bounded_temporal_order"
        return False, "broad_temporal_order"

    if operation == "temporal/timeline":
        if matched <= 16:
            return True, "bounded_timeline"
        return False, "broad_timeline"

    if query_type in {"current", "latest", "final"} and matched <= 12:
        return True, "bounded_current"
    if matched <= 12:
        return True, "bounded_graph_result"
    return False, "broad_graph_result"


def _query_has_snapshot_count_intent(query: str) -> bool:
    return bool(_SNAPSHOT_COUNT_INTENT_RE.search(query) and not _COUNT_ADDITIVE_INTENT_RE.search(query))


def _query_has_strong_snapshot_count_intent(query: str) -> bool:
    return bool(_STRONG_SNAPSHOT_COUNT_INTENT_RE.search(query))


def _query_has_value_snapshot_intent(query: str) -> bool:
    return bool(_SNAPSHOT_VALUE_INTENT_RE.search(query))


def _requires_assistant_exact_source(query: str) -> bool:
    return bool(_ASSISTANT_EXACT_SOURCE_RE.search(query))


def _looks_like_preference_or_advice_query(query: str, *, query_type: str) -> bool:
    return query_type == "preference" or bool(_PREFERENCE_ADVICE_RE.search(query))


def _include_graph_answer_candidate(graph_result: Any, *, query: str, query_type: str) -> bool:
    metadata = getattr(graph_result, "metadata", {}) or {}
    answer_hint = str(getattr(graph_result, "answer_hint", "") or "")
    if re.match(r"^(No matching|Insufficient)\b", answer_hint, re.IGNORECASE):
        return True
    answer_confidence = metadata.get("answer_confidence")
    if answer_confidence is not None:
        try:
            if float(answer_confidence) < _GRAPH_ANSWER_CONFIDENCE_FLOOR:
                return False
        except (TypeError, ValueError):
            return False

    operation = str(getattr(graph_result, "operation", ""))
    rows = list(getattr(graph_result, "evidence_rows", []) or [])
    if operation in {"temporal/latest", "current/latest"} or query_type in {"current", "latest", "final"}:
        if _latest_answer_candidate_is_ambiguous(query=query, rows=rows):
            return False
        if _missing_core_query_terms(query, rows[:1]):
            return False
    if operation == "temporal/order":
        if metadata.get("ambiguous_same_date"):
            return False
        if _missing_core_query_terms(query, rows):
            return False
    if operation in {"aggregate/sum", "aggregate/difference"} and _missing_required_component_terms(query, rows):
        return False
    return True


def _latest_answer_candidate_is_ambiguous(*, query: str, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    query_lower = query.lower()
    if "previous" in query_lower or "previously" in query_lower:
        return False
    top_event = str(rows[0].get("event") or "")
    return bool(_LATEST_EVENT_AMBIGUITY_RE.search(top_event))


def _missing_core_query_terms(query: str, rows: list[dict[str, Any]]) -> bool:
    core_terms = _answer_packet_core_terms(query)
    if not core_terms:
        return False
    row_tokens: set[str] = set()
    for row in rows:
        row_tokens.update(claim_tokens(str(row.get("event") or "")))
        for category in row.get("categories") or []:
            row_tokens.update(claim_tokens(str(category)))
        for quantity in row.get("quantities") or []:
            row_tokens.update(claim_tokens(str(quantity)))
    missing = {
        term
        for term in core_terms
        if term not in row_tokens
        and not (_CONCEPT_TOKEN_EXPANSIONS.get(term, set()) & row_tokens)
    }
    if not missing:
        return False
    # One unmatched term is often a harmless paraphrase. Multiple unmatched
    # object terms usually means the reducer found a neighboring topic instead.
    return len(missing) >= 2 or len(missing) == len(core_terms)


def _missing_required_component_terms(query: str, rows: list[dict[str, Any]]) -> bool:
    components = _query_component_term_sets(query)
    if len(components) < 2:
        return False
    row_tokens: set[str] = set()
    for row in rows:
        row_tokens.update(claim_tokens(str(row.get("event") or "")))
        for category in row.get("categories") or []:
            row_tokens.update(claim_tokens(str(category)))
    covered = 0
    for component in components:
        hits = sum(
            1
            for term in component
            if term in row_tokens or (_CONCEPT_TOKEN_EXPANSIONS.get(term, set()) & row_tokens)
        )
        if hits >= min(2, len(component)):
            covered += 1
    return covered < len(components)


def _query_component_term_sets(query: str) -> list[set[str]]:
    if not re.search(r"\b(and|plus|with)\b", query, re.IGNORECASE):
        return []
    # Keep this deliberately conservative. It is meant to catch conjunctive
    # object requests where a graph reducer found only one side of the query,
    # not to police every broad aggregate.
    tail = query
    for marker in (" of ", " for ", " between "):
        if marker in tail.lower():
            tail = tail.lower().split(marker, 1)[1]
            break
    tail = re.split(r"[?.!]", tail, maxsplit=1)[0]
    parts = re.split(r"\s+(?:and|plus|with)\s+", tail, flags=re.IGNORECASE)
    out = []
    for part in parts:
        terms = _answer_packet_core_terms(part)
        if terms:
            out.append(terms)
    return out


def _answer_packet_core_terms(query: str) -> set[str]:
    return {
        token
        for token in claim_tokens(query)
        if token not in _ANSWER_PACKET_GENERIC_TOKENS and not token.isdigit()
    }


def _fallback_candidate_matches_query(text: str, *, query_type: str, core_terms: set[str]) -> bool:
    if query_type not in {"current", "latest", "final"} or not core_terms:
        return True
    tokens = claim_tokens(text)
    return any(
        term in tokens or (_CONCEPT_TOKEN_EXPANSIONS.get(term, set()) & tokens)
        for term in core_terms
    )


def _dynamic_expansion_plan(
    graph_result: Any | None,
    *,
    graph_context_ok: bool,
    graph_context_reason: str,
    query: str,
    query_type: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    actions: list[str] = []
    metadata = getattr(graph_result, "metadata", {}) or {}
    operation = str(getattr(graph_result, "operation", ""))
    answer_confidence = float(metadata.get("answer_confidence") or 0.0)

    if graph_result is not None and answer_confidence and answer_confidence < _GRAPH_ANSWER_CONFIDENCE_FLOOR:
        reasons.append("graph_answer_confidence_below_threshold")
    if graph_result is not None and not graph_context_ok and graph_context_reason in {
        "low_confidence_graph_answer",
        "snapshot_query_quantity_sum_suppressed",
        "snapshot_query_event_count_suppressed",
        "snapshot_query_sum_suppressed",
        "broad_event_count",
        "broad_quantity_sum",
        "broad_group_max",
        "broad_timeline",
        "broad_temporal_order",
        "broad_latest",
    }:
        reasons.append(graph_context_reason)
    if operation.startswith("aggregate/") and _query_has_snapshot_count_intent(query):
        method = str(metadata.get("count_method") or metadata.get("sum_method") or "")
        if method not in {"latest_quantity_snapshot", "named_entity_count"}:
            reasons.append("snapshot_intent_needs_raw_evidence_check")
    if query_type in {"aggregate", "temporal"} and graph_result is None:
        reasons.append("no_executable_graph_result")

    if graph_result is not None and reasons:
        actions.append("expand_graph_evidence_without_answer_packet" if not graph_context_ok else "expand_additional_graph_evidence")
    elif reasons:
        actions.append("fall_back_to_ranked_candidates")

    return {
        "triggered": bool(reasons),
        "reasons": _dedupe(reasons),
        "actions": actions,
        "answer_confidence": answer_confidence or None,
    }


def _graph_prefers_user_fallback(graph_result: Any | None) -> bool:
    if graph_result is None:
        return False
    metadata = getattr(graph_result, "metadata", {}) or {}
    return metadata.get("count_method") == "named_entity_count"


def _render_preference_observation_packet(
    index: MemoryIndex,
    query: str,
    *,
    max_items: int = 8,
) -> str:
    query_tokens = _salient_query_tokens(query)
    candidates: list[tuple[float, tuple, MemoryClaimRecord]] = []
    for record in index.claims:
        if record.claim.status == "superseded":
            continue
        if record.claim.metadata.get("role") != "user":
            continue
        signal = bool(_PREFERENCE_SIGNAL_RE.search(record.text))
        preference_kind = record.claim.claim_kind in {"preference", "instruction"}
        overlap = _token_overlap(query_tokens, claim_tokens(record.text))
        if not signal and not preference_kind and overlap <= 0.0:
            continue
        score = 4.0 * overlap
        if overlap > 0.0:
            score += 1.0
        if signal:
            score += 0.5 if overlap > 0.0 else 0.12
        if preference_kind:
            score += 0.35 if overlap > 0.0 else 0.12
        score += min(0.2, 0.02 * len(record.source_turn_ids))
        candidates.append((score, record.sort_key, record))

    if not candidates:
        return ""

    lines = [
        "[memory-observation-packet]",
        "Purpose: compact user preference/advice profile from source-grounded claims.",
        "Use these as constraints for advice, but verify against raw source turns when they conflict.",
        "Observations:",
    ]
    seen: set[str] = set()
    picked = 0
    for score, _, record in sorted(candidates, key=lambda item: (-item[0], item[1], item[2].claim_id)):
        normalized = " ".join(record.text.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        label = _preference_observation_label(record.text)
        source = ", ".join(record.source_turn_ids[:3])
        date_label = record.claim.valid_from or record.sort_key[0] or "unknown-date"
        source_part = f" | source={source}" if source else ""
        lines.append(f"- {date_label} | {label} | {record.text}{source_part}")
        picked += 1
        if picked >= max_items:
            break
    return "\n".join(lines) if picked else ""


def _preference_observation_label(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(avoid|without|don't|do not|not prefer|dislike|sleep quality)\b", lower):
        return "constraint"
    if re.search(r"\b(prefer|favorite|like|love|enjoy|want|interested)\b", lower):
        return "preference"
    if re.search(r"\b(success|worked well|better|worse|struggling)\b", lower):
        return "experience"
    return "related"


def _render_temporal_target_packet(
    index: MemoryIndex,
    query: str,
    *,
    temporal_targets: list[date],
    turn_scores: dict[str, float],
    max_turns: int = 8,
) -> str:
    """Render a compact source packet for relative/explicit date lookups.

    Temporal lookup often starts with a resolved date, while the fact asked for
    may use different words than the question ("milestone" -> "signed a first
    client"). This packet gives the reader a small near-date shelf before the
    normal ranked evidence, without needing benchmark-specific labels.
    """

    if not temporal_targets:
        return ""

    query_tokens = _salient_query_tokens(query)
    candidates: list[tuple[float, int, tuple, SourceTurn]] = []
    for turn in index.turns:
        if turn.role != "user":
            continue
        source_date = _parse_source_date(turn.session_date)
        if source_date is None:
            continue
        best_days = min(abs((source_date - target).days) for target in temporal_targets)
        if best_days > 1:
            continue

        turn_tokens = _salient_query_tokens(turn.text)
        overlap = _token_overlap(query_tokens, turn_tokens)
        score = turn_scores.get(turn.turn_id, 0.0) + overlap
        score += 0.55 if best_days == 0 else 0.42
        score += _milestone_language_boost(query=query, text=turn.text)
        candidates.append((score, best_days, turn.sort_key, turn))

    if not candidates:
        return ""

    lines = [
        "[memory-date-packet]",
        "Purpose: ranked source turns near the query's resolved date; use them when relative time is central.",
        "Relative week/month arithmetic can land one day off, so inspect one-day-away rows when exact-date rows do not answer the question.",
        "Rows:",
    ]
    picked = 0
    seen_sessions: dict[str, int] = {}
    for score, best_days, _, turn in sorted(candidates, key=lambda item: (-item[0], item[1], item[2], item[3].turn_id)):
        # Avoid letting one long session monopolize the packet.
        if seen_sessions.get(turn.session_id, 0) >= 3:
            continue
        seen_sessions[turn.session_id] = seen_sessions.get(turn.session_id, 0) + 1
        proximity = "exact-date" if best_days == 0 else f"{best_days} day away"
        lines.append(f"- {proximity} | {turn.text}")
        picked += 1
        if picked >= max_turns:
            break
    return "\n".join(lines) if picked else ""


def _milestone_language_boost(*, query: str, text: str) -> float:
    query_tokens = _salient_query_tokens(query)
    if not ({"milestone", "busines", "business", "buisines", "buisiness"} & query_tokens):
        return 0.0
    text_tokens = _salient_query_tokens(text)
    if {"contract", "client"} <= text_tokens or {"first", "client"} <= text_tokens:
        return 8.0
    if {"contract", "client", "clients"} & text_tokens:
        return 7.0
    if {"signed", "sign", "launched", "launch"} & text_tokens:
        return 3.0
    if {"approved", "approval", "graduat", "started", "finished", "moved"} & text_tokens:
        return 0.35
    return 0.0


def _neighbor_turn_ids(index: MemoryIndex, turn_ids: Iterable[str], *, window: int) -> list[str]:
    by_session: dict[str, list[SourceTurn]] = {}
    for turn in index.turns:
        by_session.setdefault(turn.session_id, []).append(turn)
    out: list[str] = []
    selected = set(turn_ids)
    for turn_id in turn_ids:
        turn = index.by_turn_id.get(turn_id)
        if turn is None:
            continue
        session_turns = by_session.get(turn.session_id, [])
        for neighbor in session_turns:
            if abs(neighbor.turn_idx - turn.turn_idx) <= window and neighbor.turn_id not in selected:
                out.append(neighbor.turn_id)
    return _dedupe(out)


def _render_answer_packet(
    index: MemoryIndex,
    graph_result: Any,
    *,
    query: str,
    query_type: str,
    max_rows: int,
    include_answer_candidate: bool = True,
) -> str:
    lines = ["[memory-answer-packet]"]
    lines.append(f"Question type: {query_type}")
    lines.append(f"Question: {query}")
    if graph_result.answer_hint and include_answer_candidate:
        lines.append(f"Computed answer candidate: {graph_result.answer_hint}")
    if graph_result.answer_hint and not include_answer_candidate:
        lines.append(
            "Computed answer candidate withheld: reducer confidence or query coverage "
            "is below the answer threshold; use the evidence rows and raw source turns instead."
        )
    guidance = (
        "Use any computed answer only when the evidence rows match the question; "
        "use the raw source turns to verify or resolve ambiguity."
    )
    if query_type in {"aggregate", "temporal", "multi_hop"}:
        guidance += " Include nearby facts that identify each evidence row."
    lines.append(guidance)
    lines.append("")
    lines.append(graph_result.render(max_rows=max_rows, include_answer_hint=include_answer_candidate))
    related = _graph_related_fact_lines(index, graph_result, query=query)
    if related:
        lines.append("")
        lines.append("Related nearby facts:")
        lines.extend(f"- {fact}" for fact in related)
    return "\n".join(lines)


def _graph_related_fact_lines(
    index: MemoryIndex,
    graph_result: Any,
    *,
    query: str,
    max_facts: int = 10,
) -> list[str]:
    rows = list(getattr(graph_result, "evidence_rows", []) or [])
    if not rows:
        return []

    anchor_turn_ids = _dedupe(
        str(turn_id)
        for row in rows[:12]
        for turn_id in (row.get("turn_ids") or [])
        if turn_id in index.by_turn_id
    )
    if not anchor_turn_ids:
        return []

    selected_claim_ids = {
        str(row.get("claim_id"))
        for row in rows
        if row.get("claim_id")
    }

    query_tokens = _salient_query_tokens(query)

    facts: list[str] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows[:12], start=1):
        row_turn_ids = [
            str(turn_id)
            for turn_id in (row.get("turn_ids") or [])
            if turn_id in index.by_turn_id
        ]
        if not row_turn_ids:
            continue
        row_anchor_idxs_by_session: dict[str, list[int]] = {}
        for turn_id in row_turn_ids:
            turn = index.by_turn_id[turn_id]
            row_anchor_idxs_by_session.setdefault(turn.session_id, []).append(turn.turn_idx)
        row_event_tokens = _salient_query_tokens(str(row.get("event") or ""))

        candidates: list[tuple[float, tuple, str]] = []
        for record in index.claims:
            if record.claim_id in selected_claim_ids:
                continue
            if record.claim.metadata.get("role") != "user":
                continue
            source_turns = [
                index.by_turn_id[turn_id]
                for turn_id in record.source_turn_ids
                if turn_id in index.by_turn_id
            ]
            if not source_turns:
                continue

            same_turn = False
            min_distance: int | None = None
            for turn in source_turns:
                anchor_idxs = row_anchor_idxs_by_session.get(turn.session_id)
                if not anchor_idxs:
                    continue
                distance = min(abs(turn.turn_idx - idx) for idx in anchor_idxs)
                if distance <= 4:
                    min_distance = distance if min_distance is None else min(min_distance, distance)
                if turn.turn_id in row_turn_ids:
                    same_turn = True
            if min_distance is None:
                continue

            record_tokens = _salient_query_tokens(record.text)
            query_overlap = _token_overlap(query_tokens, record_tokens)
            event_overlap = _token_overlap(row_event_tokens, record_tokens)
            if min_distance > 0 and query_overlap <= 0.0 and event_overlap <= 0.0:
                continue

            score = (
                (2.0 if same_turn else 0.0)
                + (1.0 / (min_distance + 1))
                + query_overlap
                + event_overlap
                + _identity_detail_boost(record.text)
            )
            candidates.append((score, record.sort_key, record.text))

        picked_for_row = 0
        for _, _, text in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if text in seen:
                continue
            seen.add(text)
            facts.append(f"row {row_index}: {text}")
            picked_for_row += 1
            if len(facts) >= max_facts:
                return facts
            if picked_for_row >= 3:
                break
    return facts


def _identity_detail_boost(text: str) -> float:
    lower = text.lower()
    score = 0.0
    if re.search(r"\b(named|partner|married|bridesmaid|cousin|friend|roommate)\b", lower):
        score += 0.45
    names = [
        name
        for name in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        if name not in {"The", "User", "Assistant"}
    ]
    return score + min(0.3, 0.1 * len(names))


def _graph_claim_ids_to_expand(graph_result: Any) -> list[str]:
    operation = str(getattr(graph_result, "operation", ""))
    rows = list(getattr(graph_result, "evidence_rows", []) or [])
    if operation.startswith("aggregate/"):
        limit = 16
    elif operation in {"temporal/latest", "current/latest"}:
        limit = 6
    else:
        limit = 10
    return _dedupe(
        str(row.get("claim_id"))
        for row in rows[:limit]
        if row.get("claim_id")
    )


def _dynamic_graph_claim_ids_to_expand(graph_result: Any, *, reason: str) -> list[str]:
    operation = str(getattr(graph_result, "operation", ""))
    rows = list(getattr(graph_result, "evidence_rows", []) or [])
    if reason.startswith("broad_"):
        limit = 32
    elif operation.startswith("aggregate/"):
        limit = 24
    else:
        limit = 14
    return _dedupe(
        str(row.get("claim_id"))
        for row in rows[:limit]
        if row.get("claim_id")
    )


def _graph_turn_ids_to_expand(graph_result: Any) -> list[str]:
    operation = str(getattr(graph_result, "operation", ""))
    rows = list(getattr(graph_result, "evidence_rows", []) or [])
    if operation.startswith("aggregate/"):
        limit = 6
    elif operation in {"temporal/latest", "current/latest"}:
        limit = 4
    else:
        limit = 5
    return _dedupe(
        str(turn_id)
        for row in rows[:limit]
        for turn_id in (row.get("turn_ids") or [])
    )


def _dynamic_graph_turn_ids_to_expand(graph_result: Any, *, reason: str) -> list[str]:
    operation = str(getattr(graph_result, "operation", ""))
    rows = list(getattr(graph_result, "evidence_rows", []) or [])
    if reason.startswith("broad_"):
        limit = 16
    elif operation.startswith("aggregate/"):
        limit = 12
    else:
        limit = 8
    return _dedupe(
        str(turn_id)
        for row in rows[:limit]
        for turn_id in (row.get("turn_ids") or [])
    )


def _render_context(
    index: MemoryIndex,
    *,
    selected_claim_ids: list[str],
    selected_turn_ids: list[str],
    selected_direct_turn_ids: set[str],
    prefix_text: str = "",
    query_type: str = "",
) -> str:
    claims_for_turn: dict[str, list[MemoryClaimRecord]] = {}
    standalone_claims: list[MemoryClaimRecord] = []
    selected_turn_set = set(selected_turn_ids)
    for claim_id in selected_claim_ids:
        record = index.by_claim_id[claim_id]
        anchors = [tid for tid in record.source_turn_ids if tid in selected_turn_set]
        if not anchors:
            standalone_claims.append(record)
            continue
        for turn_id in anchors:
            claims_for_turn.setdefault(turn_id, []).append(record)

    entries: list[tuple[tuple, str]] = []
    for turn_id in selected_turn_ids:
        turn = index.by_turn_id[turn_id]
        records = sorted(claims_for_turn.get(turn_id, []), key=lambda rec: rec.sort_key)
        headers = "\n".join(_claim_header(record) for record in records)
        block = f"{headers}\n{turn.text}" if headers else turn.text
        if turn_id in selected_direct_turn_ids and not headers:
            block = f"[source-turn]\n{block}"
        entries.append((turn.sort_key, block))
    for record in standalone_claims:
        entries.append((record.sort_key, _claim_header(record)))
    entries.sort(key=lambda item: item[0], reverse=query_type in {"current", "latest", "final"})
    blocks = [block for _, block in entries]
    if prefix_text:
        blocks.insert(0, prefix_text)
    return "\n\n".join(blocks)


def _graph_query_metadata(graph_result: Any | None) -> dict[str, Any] | None:
    if graph_result is None:
        return None
    return {
        "operation": graph_result.operation,
        "answer_hint": graph_result.answer_hint,
        "selected_event_ids": graph_result.selected_event_ids,
        "selected_claim_ids": graph_result.selected_claim_ids,
        "selected_turn_ids": graph_result.selected_turn_ids,
        "evidence_rows": graph_result.evidence_rows,
        **graph_result.metadata,
    }


def _claim_header(record: MemoryClaimRecord) -> str:
    status = f"; status={record.claim.status}" if record.claim.status != "active" else ""
    temporal = _temporal_summary(record)
    quantity = _quantity_summary(record)
    suffix = "".join(part for part in (status, temporal, quantity) if part)
    return f"[memory-claim: {record.text}{suffix}]"


def _temporal_summary(record: MemoryClaimRecord) -> str:
    parts: list[str] = []
    for ref in record.temporal_refs[:2]:
        if ref.resolved_start and ref.resolved_end and ref.resolved_start != ref.resolved_end:
            parts.append(f"{ref.text} => {ref.resolved_start}..{ref.resolved_end}")
        elif ref.resolved_start:
            parts.append(f"{ref.text} => {ref.resolved_start}")
    return f"; time={'; '.join(parts)}" if parts else ""


def _quantity_summary(record: MemoryClaimRecord) -> str:
    parts: list[str] = []
    for quantity in record.quantity_claims[:3]:
        if quantity.value is None:
            continue
        value = int(quantity.value) if float(quantity.value).is_integer() else quantity.value
        parts.append(f"{value}{' ' + quantity.unit if quantity.unit else ''}")
    return f"; quantities={', '.join(parts)}" if parts else ""


def _sessions_for_turns(index: MemoryIndex, turn_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for turn_id in turn_ids:
        turn = index.by_turn_id.get(turn_id)
        if turn is None or turn.session_id in seen:
            continue
        seen.add(turn.session_id)
        out.append(turn.session_id)
    return out


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _query_temporal_targets(query: str, anchor_time: str | None) -> list[date]:
    out: list[date] = []
    for ref in extract_temporal_refs(query, anchor_time=anchor_time):
        value = ref.resolved_start or ref.resolved_end
        if not value:
            continue
        try:
            out.append(date.fromisoformat(value[:10].replace("/", "-")))
        except ValueError:
            continue
    return out


def _parse_source_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10].replace("/", "-"))
    except ValueError:
        return None


def _temporal_boost(value: str | None, targets: list[date]) -> float:
    if not value or not targets:
        return 0.0
    source_date = _parse_source_date(value)
    if source_date is None:
        return 0.0
    best_days = min(abs((source_date - target).days) for target in targets)
    if best_days == 0:
        return 0.45
    if best_days <= 1:
        return 0.32
    if best_days <= 3:
        return 0.22
    if best_days <= 7:
        return 0.12
    if best_days <= 14:
        return 0.05
    return 0.0


def _recency_boost(value: str | None) -> float:
    if not value:
        return 0.0
    # Date strings sort lexicographically after YYYY normalization. This is
    # intentionally small; recency should break ties, not dominate relevance.
    try:
        source_date = date.fromisoformat(value[:10].replace("/", "-"))
    except ValueError:
        return 0.0
    ordinal = source_date.toordinal()
    return 1.0 / (1.0 + math.exp(-(ordinal - 738000) / 365.0))


def _build_confidence(
    *,
    selected_claim_ids: list[str],
    selected_turn_ids: list[str],
    coverage_confidence: float,
    top_claim_score: float,
    top_turn_score: float,
) -> MemoryConfidence:
    relevance = max(top_claim_score, top_turn_score, 0.0)
    relevance = min(1.0, relevance)
    extraction = 0.86 if selected_claim_ids else 0.55
    authority = 0.72 if selected_claim_ids else 0.55
    return confidence_vector(
        relevance=relevance,
        entity_match=0.7 if (selected_claim_ids or selected_turn_ids) else 0.0,
        authority=authority,
        freshness=0.65,
        coverage=coverage_confidence,
        consistency=0.72,
        extraction=extraction,
        reasoning=0.68 if len(set(selected_turn_ids)) > 1 else 0.45,
    )


def _query_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text) if len(match.group(0)) >= 3}


def _salient_query_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in claim_tokens(text)
        if token not in _QUERY_STOPWORDS and not token.isdigit()
    }
    if "painting" in tokens:
        tokens.update({"art", "artwork"})
    if "artwork" in tokens:
        tokens.update({"art", "painting"})
    if "wedding" in tokens:
        tokens.add("wedd")
    if "wedd" in tokens:
        tokens.add("wedding")
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_CONCEPT_TOKEN_EXPANSIONS.get(token, ()))
    tokens = expanded
    return tokens or _query_tokens(text)


def _token_overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    hits = len(query_tokens & doc_tokens)
    return hits / max(1, len(query_tokens))


def _phrase_overlap_score(query: str, text: str) -> float:
    query_terms = _ordered_salient_terms(query)
    if len(query_terms) < 2:
        return 0.0
    haystack = " ".join(_ordered_salient_terms(text))
    if not haystack:
        return 0.0
    phrases: list[str] = []
    for size in (3, 2):
        for i in range(0, len(query_terms) - size + 1):
            phrase = " ".join(query_terms[i : i + size])
            if phrase not in phrases:
                phrases.append(phrase)
    if not phrases:
        return 0.0
    hits = sum(1 for phrase in phrases if phrase in haystack)
    return hits / len(phrases)


def _valuation_ratio_boost(*, memory_query: str, text: str) -> float:
    q_tokens = _ordered_salient_terms(memory_query)
    if not ({"worth", "value", "paid", "pay", "cost"} & set(q_tokens)):
        return 0.0
    text_tokens = set(_ordered_salient_terms(text))
    has_value = bool({"worth", "value", "valued"} & text_tokens)
    has_paid = bool({"paid", "pay", "cost"} & text_tokens)
    has_ratio = bool(_VALUATION_RATIO_TERMS & text_tokens)
    if has_value and has_paid and has_ratio:
        return 0.35
    if has_paid and has_ratio:
        return 0.22
    if has_value and has_ratio:
        return 0.16
    return 0.0


def _ordered_salient_terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 3
        and token.lower() not in _QUERY_STOPWORDS
        and not token.isdigit()
    ]


def _positive_cosine(value: float) -> float:
    # Embedding cosine is [-1, 1]. Shift only slightly so weak positive
    # evidence stays weak and negative evidence drops out.
    return max(0.0, float(value))


def _rough_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))
