"""Hybrid retrieval signals, embedding adapters, and rank fusion."""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from .compiler import MemoryIndex, claim_tokens
from .embedding_store import EmbeddingVectorStore


@dataclass
class RetrievalSignals:
    claim_scores: dict[str, float] = field(default_factory=dict)
    turn_scores: dict[str, float] = field(default_factory=dict)
    entity_scores: dict[str, float] = field(default_factory=dict)
    event_scores: dict[str, float] = field(default_factory=dict)
    state_scores: dict[str, float] = field(default_factory=dict)
    preference_scores: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrievalSignalProvider(Protocol):
    def score(self, query: str) -> RetrievalSignals:
        """Return retrieval scores and usage for one query variant."""

        ...


class CallableSignalProvider:
    def __init__(self, fn: Callable[[str], RetrievalSignals]):
        self._fn = fn

    def score(self, query: str) -> RetrievalSignals:
        return self._fn(query)


class EmbeddingSignalProvider:
    """Fielded dense retrieval over ActiveGraph's embedding-provider seam."""

    def __init__(
        self,
        index: MemoryIndex,
        provider,
        *,
        model: str | None = None,
        embed_entities: bool = True,
        embed_events: bool = True,
        estimate_tokens: Callable[[str], int] | None = None,
        cost_per_million_tokens: float = 0.0,
        vector_store: EmbeddingVectorStore | None = None,
    ) -> None:
        self.index = index
        self.provider = provider
        self.model = model or provider.default_model
        self.embed_entities = embed_entities
        self.embed_events = embed_events
        self.estimate_tokens = estimate_tokens or (lambda text: max(1, len(text) // 4))
        self.cost_per_million_tokens = cost_per_million_tokens
        self.vector_store = vector_store
        self._vectors: dict[str, tuple[list[str], list[list[float]]]] = {}

    def score(self, query: str) -> RetrievalSignals:
        started = time.perf_counter()
        q_vec = self.provider.embed(texts=[query], model=self.model)[0]
        input_tokens = self.estimate_tokens(query)

        claims = [(record.claim_id, record.text) for record in self.index.claims]
        turns = [(turn.turn_id, turn.text) for turn in self.index.turns]
        entities = (
            [
                (
                    record.entity_id,
                    " ".join((record.kind, record.canonical_name, *record.aliases)),
                )
                for record in self.index.compiled.entities
            ]
            if self.embed_entities
            else []
        )
        events = (
            [
                (
                    record.event_id,
                    " ".join(
                        (
                            record.predicate,
                            *record.category_ids,
                            record.summary,
                            record.event_start or "",
                        )
                    ),
                )
                for record in self.index.compiled.canonical_events
            ]
            if self.embed_events
            else []
        )
        states = [
            (
                record.state_id,
                f"{record.predicate} {record.value_text} {record.quantity or ''}",
            )
            for record in self.index.compiled.state_versions
        ]
        preferences = [(record.preference_id, record.text) for record in self.index.compiled.preferences]
        score_sets: dict[str, dict[str, float]] = {}
        for field, rows in (
            ("claim", claims),
            ("turn", turns),
            ("entity", entities),
            ("event", events),
            ("state", states),
            ("preference", preferences),
        ):
            ids, vectors, tokens = self._field_vectors(field, rows)
            input_tokens += tokens
            score_sets[field] = {
                uid: _cosine(q_vec, vector)
                for uid, vector in zip(ids, vectors)
            }
        cost = (input_tokens / 1_000_000.0) * self.cost_per_million_tokens
        return RetrievalSignals(
            claim_scores=score_sets["claim"],
            turn_scores=score_sets["turn"],
            entity_scores=score_sets["entity"],
            event_scores=score_sets["event"],
            state_scores=score_sets["state"],
            preference_scores=score_sets["preference"],
            input_tokens=input_tokens,
            cost_usd=cost,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "provider": type(self.provider).__name__,
                "model": self.model,
                "vector_store": (
                    type(self.vector_store).__name__ if self.vector_store is not None else "memory"
                ),
                "vector_store_stats": (
                    self.vector_store.stats()
                    if self.vector_store is not None and hasattr(self.vector_store, "stats")
                    else {}
                ),
            },
        )

    def _field_vectors(
        self,
        field: str,
        rows: list[tuple[str, str]],
    ) -> tuple[list[str], list[list[float]], int]:
        cached = self._vectors.get(field)
        ids = [uid for uid, _ in rows]
        if cached is not None and cached[0] == ids:
            return cached[0], cached[1], 0
        vectors: list[list[float] | None] = [None] * len(rows)
        missing_indexes: list[int] = []
        keys = [self._vector_key(field, uid, text) for uid, text in rows]
        if self.vector_store is not None:
            for index, key in enumerate(keys):
                vectors[index] = self.vector_store.get(key)
                if vectors[index] is None:
                    missing_indexes.append(index)
        else:
            missing_indexes = list(range(len(rows)))
        missing_texts = [rows[index][1] for index in missing_indexes]
        embedded = self.provider.embed(texts=missing_texts, model=self.model) if missing_texts else []
        pending_store_records = []
        for index, vector in zip(missing_indexes, embedded):
            vectors[index] = vector
            if self.vector_store is not None:
                uid, text = rows[index]
                pending_store_records.append(
                    (
                        keys[index],
                        vector,
                        {
                        "field": field,
                        "subject_id": uid,
                        "model": self.model,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        },
                    )
                )
        if self.vector_store is not None and pending_store_records:
            self.vector_store.put_many(pending_store_records)
        complete_vectors = [vector or [] for vector in vectors]
        self._vectors[field] = (ids, complete_vectors)
        return ids, complete_vectors, sum(self.estimate_tokens(text) for text in missing_texts)

    def _vector_key(self, field: str, uid: str, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = f"{self.model}\n{field}\n{uid}\n{text_hash}"
        return "embedding:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def lexical_signals(index: MemoryIndex, query: str) -> RetrievalSignals:
    q_tokens = claim_tokens(query)
    return RetrievalSignals(
        claim_scores={record.claim_id: _overlap(q_tokens, claim_tokens(record.text)) for record in index.claims},
        turn_scores={turn.turn_id: _overlap(q_tokens, claim_tokens(turn.text)) for turn in index.turns},
        entity_scores={
            entity.entity_id: _overlap(q_tokens, claim_tokens(entity.canonical_name))
            for entity in index.compiled.entities
        },
        event_scores={
            event.event_id: _overlap(q_tokens, claim_tokens(event.summary))
            for event in index.compiled.canonical_events
        },
        state_scores={
            state.state_id: _overlap(q_tokens, claim_tokens(state.value_text))
            for state in index.compiled.state_versions
        },
        preference_scores={
            preference.preference_id: _overlap(q_tokens, claim_tokens(preference.text))
            for preference in index.compiled.preferences
        },
        metadata={"provider": "deterministic_lexical"},
    )


def fuse_signals(signals: list[RetrievalSignals], *, rank_constant: int = 60) -> RetrievalSignals:
    """Fuse heterogeneous score sets with reciprocal-rank fusion."""

    if not signals:
        return RetrievalSignals()

    def fuse_field(field: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for signal in signals:
            scores = getattr(signal, field)
            ranked = sorted(
                (uid for uid, score in scores.items() if score > 0.0),
                key=lambda uid: (-scores[uid], uid),
            )[:200]
            for rank, uid in enumerate(ranked, start=1):
                out[uid] = out.get(uid, 0.0) + 1.0 / (rank_constant + rank)
        if not out:
            return {}
        peak = max(out.values())
        return {uid: value / peak for uid, value in out.items()}

    return RetrievalSignals(
        claim_scores=fuse_field("claim_scores"),
        turn_scores=fuse_field("turn_scores"),
        entity_scores=fuse_field("entity_scores"),
        event_scores=fuse_field("event_scores"),
        state_scores=fuse_field("state_scores"),
        preference_scores=fuse_field("preference_scores"),
        input_tokens=sum(signal.input_tokens for signal in signals),
        cost_usd=sum(signal.cost_usd for signal in signals),
        latency_ms=sum(signal.latency_ms for signal in signals),
        metadata={"fusion": "reciprocal_rank", "n_signal_sets": len(signals)},
    )


def merge_rounds(signals: list[RetrievalSignals]) -> RetrievalSignals:
    """Keep the best field score observed across targeted retrieval rounds."""

    if not signals:
        return RetrievalSignals()

    def merge_field(field: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for signal in signals:
            for uid, score in getattr(signal, field).items():
                out[uid] = max(out.get(uid, 0.0), score)
        return out

    return RetrievalSignals(
        claim_scores=merge_field("claim_scores"),
        turn_scores=merge_field("turn_scores"),
        entity_scores=merge_field("entity_scores"),
        event_scores=merge_field("event_scores"),
        state_scores=merge_field("state_scores"),
        preference_scores=merge_field("preference_scores"),
        input_tokens=sum(signal.input_tokens for signal in signals),
        cost_usd=sum(signal.cost_usd for signal in signals),
        latency_ms=sum(signal.latency_ms for signal in signals),
        metadata={"rounds": len(signals)},
    )


def propagate_graph_signals(index: MemoryIndex, signals: RetrievalSignals) -> RetrievalSignals:
    """Spread entity relevance through compiled provenance and adjacency edges."""

    propagated = 0
    for entity in index.compiled.entities:
        entity_score = signals.entity_scores.get(entity.entity_id, 0.0)
        if entity_score <= 0.0:
            continue
        for claim_id in entity.source_claim_ids:
            previous = signals.claim_scores.get(claim_id, 0.0)
            signals.claim_scores[claim_id] = max(previous, entity_score * 0.82)
            propagated += int(signals.claim_scores[claim_id] > previous)
        for turn_id in entity.source_turn_ids:
            previous = signals.turn_scores.get(turn_id, 0.0)
            signals.turn_scores[turn_id] = max(previous, entity_score * 0.76)
            propagated += int(signals.turn_scores[turn_id] > previous)
    for event in index.compiled.canonical_events:
        inherited = max(
            (signals.entity_scores.get(entity_id, 0.0) for entity_id in event.entity_ids),
            default=0.0,
        )
        if inherited > 0.0:
            previous = signals.event_scores.get(event.event_id, 0.0)
            signals.event_scores[event.event_id] = max(previous, inherited * 0.9)
            propagated += int(signals.event_scores[event.event_id] > previous)
    for state in index.compiled.state_versions:
        inherited = max(
            (signals.entity_scores.get(entity_id, 0.0) for entity_id in state.entity_ids),
            default=0.0,
        )
        if inherited > 0.0:
            previous = signals.state_scores.get(state.state_id, 0.0)
            signals.state_scores[state.state_id] = max(previous, inherited * 0.86)
            propagated += int(signals.state_scores[state.state_id] > previous)
    signals.metadata["graph_signal_propagation"] = {
        "entity_edges_applied": propagated,
    }
    return signals


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def _overlap(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)
