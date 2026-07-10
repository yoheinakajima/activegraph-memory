"""Materialize compiled memory and retrieval traces into an ActiveGraph."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from .compiler import ExtractedClaimInput, MemoryIndex, SourceTurn, compile_memory_index
from .object_types import (
    MemoryClaim,
    MemoryEntity,
    MemoryEvent,
    MemoryListItem,
    MemoryPreference,
    MemoryProof,
    MemoryRetrievalStage,
    MemoryState,
)
from .runtime import MemoryRuntime


@dataclass
class GraphMaterialization:
    claim_object_ids: dict[str, str] = field(default_factory=dict)
    entity_object_ids: dict[str, str] = field(default_factory=dict)
    event_object_ids: dict[str, str] = field(default_factory=dict)
    state_object_ids: dict[str, str] = field(default_factory=dict)
    preference_object_ids: dict[str, str] = field(default_factory=dict)
    list_item_object_ids: dict[str, str] = field(default_factory=dict)
    quantity_object_ids: dict[str, str] = field(default_factory=dict)
    temporal_ref_object_ids: dict[str, str] = field(default_factory=dict)


def materialize_memory_index(
    graph,
    index: MemoryIndex,
    *,
    source_object_ids: dict[str, str] | None = None,
) -> GraphMaterialization:
    """Write a compiled projection as idempotent graph-visible objects."""

    source_object_ids = source_object_ids or {}
    output = GraphMaterialization()
    for record in index.claims:
        obj = _upsert_by_key(
            graph,
            "memory_claim",
            "metadata.claim_id",
            record.claim_id,
            record.claim.model_dump(),
        )
        output.claim_object_ids[record.claim_id] = obj.id
        for turn_id in record.source_turn_ids:
            source_id = source_object_ids.get(turn_id)
            if source_id:
                _ensure_relation(graph, obj.id, source_id, "memory_grounded_in")
        for quantity_index, quantity in enumerate(record.quantity_claims):
            quantity_id = _stable_artifact_id(
                "quantity",
                f"{record.claim_id}|{quantity_index}|{quantity.model_dump_json()}",
            )
            payload = quantity.model_dump()
            payload["metadata"] = {**payload.get("metadata", {}), "quantity_id": quantity_id}
            quantity_obj = _upsert_by_key(
                graph,
                "quantity_claim",
                "metadata.quantity_id",
                quantity_id,
                payload,
            )
            output.quantity_object_ids[quantity_id] = quantity_obj.id
            _ensure_relation(graph, obj.id, quantity_obj.id, "memory_has_quantity")
        for temporal_index, temporal_ref in enumerate(record.temporal_refs):
            temporal_id = _stable_artifact_id(
                "temporal-ref",
                f"{record.claim_id}|{temporal_index}|{temporal_ref.model_dump_json()}",
            )
            payload = temporal_ref.model_dump()
            payload["metadata"] = {**payload.get("metadata", {}), "temporal_ref_id": temporal_id}
            temporal_obj = _upsert_by_key(
                graph,
                "temporal_ref",
                "metadata.temporal_ref_id",
                temporal_id,
                payload,
            )
            output.temporal_ref_object_ids[temporal_id] = temporal_obj.id
            _ensure_relation(graph, obj.id, temporal_obj.id, "memory_has_temporal_ref")

    for entity in index.compiled.entities:
        payload = MemoryEntity(
            entity_key=entity.entity_id,
            canonical_name=entity.canonical_name,
            kind=entity.kind,
            aliases=list(entity.aliases),
            source_claim_ids=list(entity.source_claim_ids),
            source_ids=list(entity.source_turn_ids),
            metadata=entity.metadata,
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_entity", "entity_key", entity.entity_id, payload)
        output.entity_object_ids[entity.entity_id] = obj.id
        _link_sources(graph, obj.id, entity.source_claim_ids, entity.source_turn_ids, output, source_object_ids)

    for event in index.compiled.canonical_events:
        payload = MemoryEvent(
            event_key=event.event_id,
            predicate=event.predicate,
            summary=event.summary,
            entity_refs=list(event.entity_ids),
            categories=list(event.category_ids),
            modality=event.modality,
            polarity=event.polarity,
            event_start=event.event_start,
            event_end=event.event_end,
            source_claim_ids=list(event.claim_ids),
            source_ids=list(event.source_turn_ids),
            confidence=event.confidence,
            metadata={**event.metadata, "quantities": list(event.quantities)},
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_event", "event_key", event.event_id, payload)
        output.event_object_ids[event.event_id] = obj.id
        _link_sources(graph, obj.id, event.claim_ids, event.source_turn_ids, output, source_object_ids)
        for entity_id in event.entity_ids:
            target = output.entity_object_ids.get(entity_id)
            if target:
                _ensure_relation(graph, obj.id, target, "memory_about")

    states_by_key: dict[str, list[Any]] = {}
    for state in index.compiled.state_versions:
        payload = MemoryState(
            state_key=state.state_key,
            value=state.value_text,
            subject_ref=state.subject_ref,
            predicate=state.predicate,
            status=state.status,
            valid_from=state.valid_from,
            valid_until=state.valid_until,
            observed_at=state.observed_at,
            source_claim_id=state.source_claim_id,
            source_ids=list(state.source_turn_ids),
            confidence=state.confidence,
            metadata={**state.metadata, "state_id": state.state_id, "quantity": state.quantity},
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_state", "metadata.state_id", state.state_id, payload)
        output.state_object_ids[state.state_id] = obj.id
        states_by_key.setdefault(state.state_key, []).append((state, obj.id))
        _link_sources(
            graph,
            obj.id,
            [state.source_claim_id] if state.source_claim_id else [],
            state.source_turn_ids,
            output,
            source_object_ids,
        )
        for entity_id in state.entity_ids:
            target = output.entity_object_ids.get(entity_id)
            if target:
                _ensure_relation(graph, obj.id, target, "memory_about")
    for versions in states_by_key.values():
        versions.sort(key=lambda item: (item[0].observed_at or "", item[0].state_id))
        for older, newer in zip(versions, versions[1:]):
            _ensure_relation(graph, newer[1], older[1], "memory_version_of")
            _ensure_relation(graph, newer[1], older[1], "memory_supersedes")

    for preference in index.compiled.preferences:
        payload = MemoryPreference(
            preference_key=preference.preference_id,
            subject_ref=preference.subject_ref,
            text=preference.text,
            polarity=preference.polarity,
            scope_terms=list(preference.scope_terms),
            explicit=preference.explicit,
            observed_at=preference.observed_at,
            source_claim_id=preference.source_claim_id,
            source_ids=list(preference.source_turn_ids),
            confidence=preference.confidence,
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_preference", "preference_key", preference.preference_id, payload)
        output.preference_object_ids[preference.preference_id] = obj.id
        _link_sources(
            graph,
            obj.id,
            [preference.source_claim_id] if preference.source_claim_id else [],
            preference.source_turn_ids,
            output,
            source_object_ids,
        )

    for item in index.compiled.list_items:
        payload = MemoryListItem(
            item_key=item.item_id,
            list_key=item.list_id,
            position=item.position,
            text=item.text,
            role=item.role,
            source_id=item.source_turn_id,
            observed_at=item.observed_at,
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_list_item", "item_key", item.item_id, payload)
        output.list_item_object_ids[item.item_id] = obj.id
        source_id = source_object_ids.get(item.source_turn_id)
        if source_id:
            _ensure_relation(graph, obj.id, source_id, "memory_grounded_in")
    return output


def materialize_retrieval_trace(
    graph,
    query_id: str,
    result,
    *,
    materialization: GraphMaterialization | None = None,
    source_object_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write proof and telemetry objects linked to an existing memory query."""

    compiled = result.metadata.get("compiled_evidence") or {}
    proof = MemoryProof(
        query_id=query_id,
        operation=str(compiled.get("operation") or "lookup"),
        complete=bool(compiled.get("proof_complete")),
        confidence=float(compiled.get("confidence") or 0.0),
        requirements=list(compiled.get("proof_requirements") or []),
        satisfied=list(compiled.get("satisfied_requirements") or []),
        missing=list(compiled.get("missing_requirements") or []),
        candidate_answer=compiled.get("candidate_answer"),
        evidence_ids=[
            *list(compiled.get("selected_claim_ids") or []),
            *list(compiled.get("selected_turn_ids") or []),
            *list(compiled.get("selected_event_ids") or []),
        ],
        metadata={
            **dict(compiled.get("metadata") or {}),
            "trace_key": f"{query_id}|proof",
        },
    )
    proof_obj = _upsert_by_key(
        graph,
        "memory_proof",
        "metadata.trace_key",
        f"{query_id}|proof",
        proof.model_dump(),
    )
    _try_relation(graph, proof_obj.id, query_id, "memory_proves")
    materialization = materialization or GraphMaterialization()
    source_object_ids = source_object_ids or {}
    for claim_id in compiled.get("selected_claim_ids") or []:
        target = materialization.claim_object_ids.get(claim_id)
        if target:
            _ensure_relation(graph, proof_obj.id, target, "memory_grounded_in")
    for event_id in compiled.get("selected_event_ids") or []:
        target = materialization.event_object_ids.get(event_id)
        if target:
            _ensure_relation(graph, proof_obj.id, target, "memory_grounded_in")
    for source_id in compiled.get("selected_turn_ids") or []:
        target = source_object_ids.get(source_id)
        if target:
            _ensure_relation(graph, proof_obj.id, target, "memory_grounded_in")
    stage_ids = []
    for index, raw in enumerate((result.metadata.get("pipeline_telemetry") or {}).get("stages", [])):
        trace_key = f"{query_id}|stage|{index}|{raw.get('stage')}|{raw.get('implementation')}"
        stage = MemoryRetrievalStage(
            query_id=query_id,
            stage=str(raw.get("stage") or "unknown"),
            implementation=str(raw.get("implementation") or "unknown"),
            duration_ms=float(raw.get("duration_ms") or 0.0),
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            cost_usd=float(raw.get("cost_usd") or 0.0),
            candidates_in=int(raw.get("candidates_in") or 0),
            candidates_out=int(raw.get("candidates_out") or 0),
            cached=bool(raw.get("cached")),
            metadata={**dict(raw.get("metadata") or {}), "trace_key": trace_key},
        )
        obj = _upsert_by_key(
            graph,
            "memory_retrieval_stage",
            "metadata.trace_key",
            trace_key,
            stage.model_dump(),
        )
        _try_relation(graph, obj.id, query_id, "memory_stage_for")
        stage_ids.append(obj.id)
    return {"proof_object_id": proof_obj.id, "stage_object_ids": stage_ids}


class GraphMemoryRepository:
    """Convenience facade that keeps compilation and retrieval graph-visible."""

    def __init__(self, graph, *, runtime: MemoryRuntime | None = None) -> None:
        self.graph = graph
        self.runtime = runtime or MemoryRuntime("balanced")
        self.index: MemoryIndex | None = None
        self.materialization: GraphMaterialization | None = None
        self.source_object_ids: dict[str, str] = {}

    def compile(
        self,
        *,
        turns: list[SourceTurn],
        claims: list[ExtractedClaimInput],
        source_object_ids: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryIndex:
        self.index = compile_memory_index(turns=turns, claims=claims, metadata=metadata)
        self.source_object_ids = dict(source_object_ids or {})
        self.materialization = materialize_memory_index(
            self.graph,
            self.index,
            source_object_ids=source_object_ids,
        )
        return self.index

    def retrieve(self, query, *, query_id: str, **kwargs):
        if self.index is None:
            raise RuntimeError("GraphMemoryRepository.compile() must run before retrieve()")
        result = self.runtime.retrieve(self.index, query, query_id=query_id, **kwargs)
        materialize_retrieval_trace(
            self.graph,
            query_id,
            result,
            materialization=self.materialization,
            source_object_ids=self.source_object_ids,
        )
        return result


def _upsert_by_key(graph, object_type: str, path: str, value: str, payload: dict[str, Any]):
    for obj in graph.objects(type=object_type):
        current: Any = obj.data
        for part in path.split("."):
            current = current.get(part) if isinstance(current, dict) else None
        if current == value:
            return obj
    return graph.add_object(object_type, payload)


def _link_sources(
    graph,
    source_id: str,
    claim_ids,
    turn_ids,
    materialization: GraphMaterialization,
    source_object_ids: dict[str, str],
) -> None:
    for claim_id in claim_ids:
        target = materialization.claim_object_ids.get(claim_id)
        if target:
            _ensure_relation(graph, source_id, target, "memory_grounded_in")
    for turn_id in turn_ids:
        target = source_object_ids.get(turn_id)
        if target:
            _ensure_relation(graph, source_id, target, "memory_grounded_in")


def _ensure_relation(graph, source: str, target: str, relation_type: str) -> None:
    if any(
        relation.source == source and relation.target == target
        for relation in graph.relations(source=source, type=relation_type)
    ):
        return
    graph.add_relation(source, target, relation_type)


def _try_relation(graph, source: str, target: str, relation_type: str) -> None:
    try:
        _ensure_relation(graph, source, target, relation_type)
    except Exception:
        # Standalone callers may use an external query key rather than a graph object id.
        pass


def _stable_artifact_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
