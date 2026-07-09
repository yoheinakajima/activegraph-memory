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

    temporal_targets = _query_temporal_targets(memory_query.query, memory_query.time_anchor)
    claim_scores = claim_scores or {}
    turn_scores = turn_scores or {}

    scored_claims = _score_claims(
        index.claims,
        memory_query.query,
        query_type=query_type,
        external_scores=claim_scores,
        temporal_targets=temporal_targets,
    )
    scored_turns = _score_turns(
        index.turns,
        memory_query.query,
        query_type=query_type,
        external_scores=turn_scores,
        temporal_targets=temporal_targets,
    )

    candidates = _rank_candidates(index, scored_claims, scored_turns, query_type=query_type)
    graph_result = run_graph_query(
        index,
        memory_query.query,
        query_type=query_type,
        anchor_time=memory_query.time_anchor,
    )
    selected_claim_ids: list[str] = []
    selected_turn_ids: set[str] = set()
    selected_direct_turn_ids: set[str] = set()
    selected_claim_set: set[str] = set()
    graph_context = ""
    running = 0
    truncated = False

    def fits(cost: int) -> bool:
        return running + cost <= token_budget

    def add_cost(cost: int) -> None:
        nonlocal running
        running += cost

    def add_claim(record: MemoryClaimRecord) -> bool:
        nonlocal truncated
        if record.claim_id in selected_claim_set:
            return True
        header = _claim_header(record)
        cost = token_counter(header) + 1
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

    if graph_result is not None and graph_result.answer_hint:
        for max_rows in (24, 12, 6, 0):
            rendered_graph = graph_result.render(max_rows=max_rows)
            graph_cost = token_counter(rendered_graph) + 2
            if fits(graph_cost):
                graph_context = rendered_graph
                add_cost(graph_cost)
                break
        else:
            minimal_graph = f"[graph-query: {graph_result.operation}]\n{graph_result.answer_hint}"
            graph_cost = token_counter(minimal_graph) + 2
            if fits(graph_cost):
                graph_context = minimal_graph
                add_cost(graph_cost)
            else:
                truncated = True
        for claim_id in graph_result.selected_claim_ids:
            record = index.by_claim_id.get(claim_id)
            if record is not None:
                add_claim(record)
        for turn_id in graph_result.selected_turn_ids:
            turn = index.by_turn_id.get(turn_id)
            if turn is not None:
                add_turn(turn)

    for candidate in candidates:
        if candidate.score <= 0.0 and selected_claim_ids:
            break
        if candidate.kind == "claim":
            record = index.by_claim_id[candidate.uid]
            add_claim(record)
            continue

        turn = index.by_turn_id[candidate.uid]
        add_turn(turn, direct=True)

    rendered_turn_ids = sorted(selected_turn_ids, key=lambda tid: index.by_turn_id[tid].sort_key)
    context_text = _render_context(
        index,
        selected_claim_ids=selected_claim_ids,
        selected_turn_ids=rendered_turn_ids,
        selected_direct_turn_ids=selected_direct_turn_ids,
        prefix_text=graph_context,
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
            "token_budget": token_budget,
            "estimated_context_tokens": token_counter(context_text),
            "temporal_targets": [target.isoformat() for target in temporal_targets],
            "selected_unit_ids": [
                *rendered_turn_ids,
                *selected_claim_ids,
                *(graph_result.selected_event_ids if graph_result else []),
            ],
            "graph_query": _graph_query_metadata(graph_result),
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
) -> dict[str, float]:
    q_tokens = _query_tokens(query)
    out: dict[str, float] = {}
    for record in claims:
        lexical = _token_overlap(q_tokens, claim_tokens(record.text))
        external = _positive_cosine(external_scores.get(record.claim_id, 0.0))
        score = (0.82 * external) + (0.18 * lexical)
        score += _temporal_boost(record.claim.valid_from, temporal_targets)
        if query_type in {"latest", "current", "final"}:
            score += _recency_boost(record.sort_key[0]) * 0.15
            if record.claim.status == "superseded":
                score *= 0.35
        elif record.claim.status == "superseded":
            score *= 0.8
        if query_type == "preference" and record.claim.claim_kind == "preference":
            score += 0.18
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
) -> dict[str, float]:
    q_tokens = _query_tokens(query)
    out: dict[str, float] = {}
    for turn in turns:
        lexical = _token_overlap(q_tokens, _query_tokens(turn.text))
        external = _positive_cosine(external_scores.get(turn.turn_id, 0.0))
        score = (0.74 * external) + (0.26 * lexical)
        score += _temporal_boost(turn.session_date, temporal_targets)
        if query_type in {"temporal", "aggregate", "multi_hop"}:
            score += 0.03
        if query_type in {"latest", "current", "final"}:
            score += _recency_boost(turn.session_date) * 0.08
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


def _render_context(
    index: MemoryIndex,
    *,
    selected_claim_ids: list[str],
    selected_turn_ids: list[str],
    selected_direct_turn_ids: set[str],
    prefix_text: str = "",
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
    entries.sort(key=lambda item: item[0])
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


def _temporal_boost(value: str | None, targets: list[date]) -> float:
    if not value or not targets:
        return 0.0
    try:
        source_date = date.fromisoformat(value[:10].replace("/", "-"))
    except ValueError:
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


def _token_overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    hits = len(query_tokens & doc_tokens)
    return hits / max(1, len(query_tokens))


def _positive_cosine(value: float) -> float:
    # Embedding cosine is [-1, 1]. Shift only slightly so weak positive
    # evidence stays weak and negative evidence drops out.
    return max(0.0, float(value))


def _rough_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))
