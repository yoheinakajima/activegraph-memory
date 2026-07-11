"""Materialize compiled memory and retrieval traces into an ActiveGraph."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from .compiler import ExtractedClaimInput, MemoryIndex, SourceTurn, compile_memory_index
from .extraction import ActiveGraphLLMMemoryExtractor, MemoryExtractor, extract_claim_inputs
from .object_types import (
    MemoryClaim,
    MemoryConflict,
    MemoryEntity,
    MemoryEvent,
    MemoryIngestionStage,
    MemoryListItem,
    MemoryPreference,
    MemoryProof,
    MemoryRetrievalStage,
    MemoryRetrievalAssessment,
    MemorySourceTurn,
    MemoryState,
)
from .runtime import MemoryRuntime


@dataclass
class GraphMaterialization:
    ingestion_stage_object_ids: dict[str, str] = field(default_factory=dict)
    source_turn_object_ids: dict[str, str] = field(default_factory=dict)
    claim_object_ids: dict[str, str] = field(default_factory=dict)
    entity_object_ids: dict[str, str] = field(default_factory=dict)
    event_object_ids: dict[str, str] = field(default_factory=dict)
    state_object_ids: dict[str, str] = field(default_factory=dict)
    preference_object_ids: dict[str, str] = field(default_factory=dict)
    list_item_object_ids: dict[str, str] = field(default_factory=dict)
    quantity_object_ids: dict[str, str] = field(default_factory=dict)
    temporal_ref_object_ids: dict[str, str] = field(default_factory=dict)
    conflict_object_ids: dict[str, str] = field(default_factory=dict)


def materialize_memory_index(
    graph,
    index: MemoryIndex,
    *,
    source_object_ids: dict[str, str] | None = None,
) -> GraphMaterialization:
    """Write a compiled projection as idempotent graph-visible objects."""

    source_object_ids = source_object_ids or {}
    output = GraphMaterialization()
    for turn in index.turns:
        payload = MemorySourceTurn(
            turn_key=turn.turn_id,
            session_id=turn.session_id,
            session_date=turn.session_date,
            session_idx=turn.session_idx,
            turn_idx=turn.turn_idx,
            role=turn.role,
            content=turn.content,
            rendered_text=turn.text,
            metadata=turn.metadata,
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_source_turn", "turn_key", turn.turn_id, payload)
        output.source_turn_object_ids[turn.turn_id] = obj.id
        external_source_id = source_object_ids.get(turn.turn_id)
        if external_source_id:
            _ensure_relation(graph, obj.id, external_source_id, "memory_grounded_in")

    for raw in index.metadata.get("ingestion_runs") or []:
        stage = MemoryIngestionStage.model_validate(raw)
        obj = _upsert_by_key(
            graph,
            "memory_ingestion_stage",
            "stage_key",
            stage.stage_key,
            stage.model_dump(),
        )
        output.ingestion_stage_object_ids[stage.stage_key] = obj.id
        for turn_id in stage.source_ids:
            target = output.source_turn_object_ids.get(turn_id)
            if target:
                _ensure_relation(graph, obj.id, target, "memory_grounded_in")

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
            turn_object_id = output.source_turn_object_ids.get(turn_id)
            if turn_object_id:
                _ensure_relation(graph, obj.id, turn_object_id, "memory_grounded_in")
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

    for record in index.claims:
        source = output.claim_object_ids.get(record.claim_id)
        if not source:
            continue
        if record.superseded_by:
            newer = output.claim_object_ids.get(record.superseded_by)
            if newer:
                _ensure_relation(graph, newer, source, "memory_supersedes")
        for supported_id in record.supports:
            target = output.claim_object_ids.get(supported_id)
            if target:
                _ensure_relation(graph, source, target, "memory_supports")
        for contradicted_id in record.contradicts:
            target = output.claim_object_ids.get(contradicted_id)
            if target:
                _ensure_relation(graph, source, target, "memory_contradicts")

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

    for conflict in index.compiled.conflicts:
        payload = MemoryConflict(
            conflict_key=conflict.conflict_id,
            claim_ids=list(conflict.claim_ids),
            state_key=conflict.state_key,
            reason=conflict.reason,
            status=conflict.status,
            confidence=conflict.confidence,
            source_ids=list(conflict.source_turn_ids),
            metadata=conflict.metadata,
        ).model_dump()
        obj = _upsert_by_key(graph, "memory_conflict", "conflict_key", conflict.conflict_id, payload)
        output.conflict_object_ids[conflict.conflict_id] = obj.id
        for claim_id in conflict.claim_ids:
            target = output.claim_object_ids.get(claim_id)
            if target:
                _ensure_relation(graph, obj.id, target, "memory_contradicts")
                for other_id in conflict.claim_ids:
                    other = output.claim_object_ids.get(other_id)
                    if other and other != target:
                        _ensure_relation(graph, target, other, "memory_contradicts")
        for turn_id in conflict.source_turn_ids:
            target = output.source_turn_object_ids.get(turn_id)
            if target:
                _ensure_relation(graph, obj.id, target, "memory_grounded_in")
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
        target = materialization.source_turn_object_ids.get(source_id) or source_object_ids.get(source_id)
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
    assessment_id = None
    raw_assessment = result.metadata.get("retrieval_assessment") or {}
    if raw_assessment:
        trace_key = f"{query_id}|assessment|{raw_assessment.get('round_index', 1)}"
        assessment = MemoryRetrievalAssessment(
            query_id=query_id,
            round_index=int(raw_assessment.get("round_index") or 1),
            sufficient=bool(raw_assessment.get("sufficient")),
            overall_confidence=float(raw_assessment.get("overall_confidence") or 0.0),
            dimensions=dict(raw_assessment.get("dimensions") or {}),
            missing_requirements=list(raw_assessment.get("missing_requirements") or []),
            conflict_ids=list(raw_assessment.get("conflict_ids") or []),
            reasons=list(raw_assessment.get("reasons") or []),
            next_queries=list(raw_assessment.get("next_queries") or []),
            metadata={**dict(raw_assessment.get("metadata") or {}), "trace_key": trace_key},
        )
        assessment_obj = _upsert_by_key(
            graph,
            "memory_retrieval_assessment",
            "metadata.trace_key",
            trace_key,
            assessment.model_dump(),
        )
        _try_relation(graph, assessment_obj.id, query_id, "memory_stage_for")
        assessment_id = assessment_obj.id
    return {
        "proof_object_id": proof_obj.id,
        "stage_object_ids": stage_ids,
        "assessment_object_id": assessment_id,
    }


def load_memory_index(
    graph,
    *,
    metadata: dict[str, Any] | None = None,
    enable_temporal_resolution: bool = True,
    enable_conflict_detection: bool = True,
) -> MemoryIndex:
    """Rebuild the in-process projection from graph-visible source turns and claims."""

    turns = []
    for obj in graph.objects(type="memory_source_turn"):
        data = MemorySourceTurn.model_validate(obj.data)
        turns.append(
            SourceTurn(
                turn_id=data.turn_key,
                session_id=data.session_id,
                session_date=data.session_date,
                session_idx=data.session_idx,
                turn_idx=data.turn_idx,
                role=data.role,
                content=data.content,
                text=data.rendered_text,
                metadata=data.metadata,
            )
        )
    by_turn_id = {turn.turn_id: turn for turn in turns}
    claims = []
    for obj in graph.objects(type="memory_claim"):
        data = MemoryClaim.model_validate(obj.data)
        source_turn_ids = tuple(source_id for source_id in data.source_ids if source_id in by_turn_id)
        anchor = min(
            (by_turn_id[source_id] for source_id in source_turn_ids),
            key=lambda turn: turn.sort_key,
            default=None,
        )
        if anchor is None:
            continue
        claim_metadata = {
            **data.metadata,
            "claim_kind": data.claim_kind,
            "subject_ref": data.subject_ref,
            "scope": data.scope,
            "authority": data.authority,
            "belief_confidence": data.belief_confidence,
            "valid_from": data.valid_from,
            "valid_until": data.valid_until,
            "observed_at": data.observed_at,
        }
        claims.append(
            ExtractedClaimInput(
                text=data.text,
                session_id=anchor.session_id,
                session_date=anchor.session_date,
                session_idx=anchor.session_idx,
                role=str(data.metadata.get("role") or anchor.role),
                confidence=data.extraction_confidence,
                source=str(data.metadata.get("source") or "graph_replay"),
                metadata=claim_metadata,
                source_turn_ids=source_turn_ids,
            )
        )
    if not turns:
        raise RuntimeError("No memory_source_turn objects are available for reconstruction")
    ingestion_runs = [
        MemoryIngestionStage.model_validate(obj.data).model_dump()
        for obj in graph.objects(type="memory_ingestion_stage")
    ]
    replay_metadata = {"loaded_from_graph": True, **(metadata or {})}
    if ingestion_runs:
        replay_metadata["ingestion_runs"] = sorted(
            ingestion_runs,
            key=lambda item: item["stage_key"],
        )
    return compile_memory_index(
        turns=turns,
        claims=claims,
        metadata=replay_metadata,
        enable_temporal_resolution=enable_temporal_resolution,
        enable_conflict_detection=enable_conflict_detection,
    )


class GraphMemoryRepository:
    """Convenience facade that keeps compilation and retrieval graph-visible."""

    def __init__(
        self,
        graph,
        *,
        runtime: MemoryRuntime | None = None,
        extractor: MemoryExtractor | None = None,
        enable_claim_extraction: bool = True,
        enable_temporal_resolution: bool = True,
        enable_conflict_detection: bool = True,
    ) -> None:
        self.graph = graph
        self.runtime = runtime or MemoryRuntime("balanced")
        self.extractor = extractor
        self.enable_claim_extraction = enable_claim_extraction
        self.enable_temporal_resolution = enable_temporal_resolution
        self.enable_conflict_detection = enable_conflict_detection
        self.index: MemoryIndex | None = None
        self.materialization: GraphMaterialization | None = None
        self.source_object_ids: dict[str, str] = {}

    @classmethod
    def from_activegraph(
        cls,
        activegraph_runtime,
        *,
        profile=None,
        settings=None,
        extraction_model: str | None = None,
        reasoning_model: str | None = None,
        embedding_model: str | None = None,
        embedding_cost_per_million_tokens: float = 0.0,
        embedding_store=None,
        token_counter=None,
    ) -> "GraphMemoryRepository":
        """Bind ingestion and retrieval to one configured ActiveGraph runtime."""

        from .profiles import profile_from_settings

        resolved_profile = (
            profile
            if profile is not None
            else profile_from_settings(settings) if settings is not None else "balanced"
        )
        if settings is not None:
            extraction_model = extraction_model or settings.extraction_model
            reasoning_model = reasoning_model or settings.reasoning_model
            embedding_model = embedding_model or settings.embedding_model
            if not embedding_cost_per_million_tokens:
                embedding_cost_per_million_tokens = settings.embedding_cost_per_million_tokens
        runtime = MemoryRuntime.from_activegraph(
            activegraph_runtime,
            resolved_profile,
            reasoning_model=reasoning_model,
            embedding_model=embedding_model,
            embedding_cost_per_million_tokens=embedding_cost_per_million_tokens,
            embedding_store=embedding_store,
            token_counter=token_counter,
        )
        extractor = None
        if activegraph_runtime.llm_provider is not None and extraction_model:
            extractor = ActiveGraphLLMMemoryExtractor(
                activegraph_runtime.llm_provider,
                model=extraction_model,
            )
        # ADR 0026: when shared extraction is configured, the standalone
        # extractor is kept only as an inert compatibility adapter — never
        # silently active beside shared extraction (zero provider calls).
        consume_shared = bool(
            settings is not None and getattr(settings, "consume_shared_extraction", False)
        )
        if consume_shared:
            from .extraction import DeterministicMemoryExtractor
            from .shared_extraction import CompatibilityMemoryExtractor

            extractor = CompatibilityMemoryExtractor(
                extractor or DeterministicMemoryExtractor(),
                shared_extraction_active=True,
            )
        return cls(
            activegraph_runtime.graph,
            runtime=runtime,
            extractor=extractor,
            enable_claim_extraction=(settings.enable_claim_extraction if settings is not None else True),
            enable_temporal_resolution=(settings.enable_temporal_resolution if settings is not None else True),
            enable_conflict_detection=(settings.enable_conflict_detection if settings is not None else True),
        )

    def compile(
        self,
        *,
        turns: list[SourceTurn],
        claims: list[ExtractedClaimInput] | None = None,
        extractor: MemoryExtractor | None = None,
        source_object_ids: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryIndex:
        extraction_result = None
        if claims is None:
            if not self.enable_claim_extraction:
                raise ValueError("Claim extraction is disabled; provide accepted claims explicitly")
            claims, extraction_result = extract_claim_inputs(turns, extractor=extractor or self.extractor)
        compile_metadata = dict(metadata or {})
        if extraction_result is not None:
            trace = _ingestion_trace(extraction_result, turns)
            compile_metadata["extraction"] = trace
            compile_metadata["ingestion_runs"] = _merge_ingestion_runs(
                compile_metadata.get("ingestion_runs") or [],
                [trace],
            )
        self.index = compile_memory_index(
            turns=turns,
            claims=claims,
            metadata=compile_metadata,
            enable_temporal_resolution=self.enable_temporal_resolution,
            enable_conflict_detection=self.enable_conflict_detection,
        )
        self.source_object_ids = dict(source_object_ids or {})
        self.materialization = materialize_memory_index(
            self.graph,
            self.index,
            source_object_ids=source_object_ids,
        )
        return self.index

    def load(self, *, metadata: dict[str, Any] | None = None) -> MemoryIndex:
        """Restore a compiled index after process restart from ActiveGraph state."""

        self.index = load_memory_index(
            self.graph,
            metadata=metadata,
            enable_temporal_resolution=self.enable_temporal_resolution,
            enable_conflict_detection=self.enable_conflict_detection,
        )
        self.materialization = graph_materialization(self.graph)
        self.source_object_ids = {}
        return self.index

    def append(
        self,
        *,
        turns: list[SourceTurn],
        claims: list[ExtractedClaimInput] | None = None,
        extractor: MemoryExtractor | None = None,
        source_object_ids: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryIndex:
        """Append source events and deterministically rebuild stable projections."""

        if self.index is None:
            try:
                self.load()
            except RuntimeError:
                return self.compile(
                    turns=turns,
                    claims=claims,
                    extractor=extractor,
                    source_object_ids=source_object_ids,
                    metadata=metadata,
                )
        assert self.index is not None
        new_claims = claims
        extraction_trace = None
        if new_claims is None:
            if not self.enable_claim_extraction:
                raise ValueError("Claim extraction is disabled; provide accepted claims explicitly")
            new_claims, extraction_result = extract_claim_inputs(turns, extractor=extractor or self.extractor)
            extraction_trace = _ingestion_trace(extraction_result, turns)
        merged_turns = {turn.turn_id: turn for turn in self.index.turns}
        merged_turns.update({turn.turn_id: turn for turn in turns})
        merged_claims = {claim.claim_id: _claim_input_from_record(claim) for claim in self.index.claims}
        for claim in new_claims:
            probe = compile_memory_index(
                turns=merged_turns.values(),
                claims=[claim],
                enable_temporal_resolution=self.enable_temporal_resolution,
                enable_conflict_detection=False,
            ).claims
            if probe:
                merged_claims[probe[0].claim_id] = claim
        self.source_object_ids.update(source_object_ids or {})
        compile_metadata = {**self.index.metadata, **(metadata or {}), "incremental_rebuild": True}
        if extraction_trace is not None:
            compile_metadata["extraction"] = extraction_trace
            compile_metadata["ingestion_runs"] = _merge_ingestion_runs(
                self.index.metadata.get("ingestion_runs") or [],
                [extraction_trace],
            )
        return self.compile(
            turns=list(merged_turns.values()),
            claims=list(merged_claims.values()),
            source_object_ids=self.source_object_ids,
            metadata=compile_metadata,
        )

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
            if obj.data != payload:
                graph.patch_object(obj.id, payload, rationale="refresh deterministic memory projection")
                return graph.get_object(obj.id)
            return obj
    return graph.add_object(object_type, payload)


def graph_materialization(graph) -> GraphMaterialization:
    """Index stable logical keys for an already materialized graph."""

    output = GraphMaterialization()
    mappings = (
        ("memory_ingestion_stage", "stage_key", output.ingestion_stage_object_ids),
        ("memory_source_turn", "turn_key", output.source_turn_object_ids),
        ("memory_claim", "metadata.claim_id", output.claim_object_ids),
        ("memory_entity", "entity_key", output.entity_object_ids),
        ("memory_event", "event_key", output.event_object_ids),
        ("memory_state", "metadata.state_id", output.state_object_ids),
        ("memory_preference", "preference_key", output.preference_object_ids),
        ("memory_list_item", "item_key", output.list_item_object_ids),
        ("memory_conflict", "conflict_key", output.conflict_object_ids),
    )
    for object_type, path, target in mappings:
        for obj in graph.objects(type=object_type):
            current: Any = obj.data
            for part in path.split("."):
                current = current.get(part) if isinstance(current, dict) else None
            if current:
                target[str(current)] = obj.id
    return output


def _ingestion_trace(result, turns: list[SourceTurn]) -> dict[str, Any]:
    source_ids = [turn.turn_id for turn in turns]
    identity = "|".join([result.extractor, result.model, *source_ids])
    return MemoryIngestionStage(
        stage_key=_stable_artifact_id("ingestion", identity),
        extractor=result.extractor,
        model=result.model,
        source_ids=source_ids,
        fact_count=len(result.facts),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        duration_ms=result.latency_ms,
        cached=result.cached,
        metadata=result.metadata,
    ).model_dump()


def _merge_ingestion_runs(*collections) -> list[dict[str, Any]]:
    by_key = {}
    for collection in collections:
        for raw in collection:
            stage = MemoryIngestionStage.model_validate(raw)
            by_key[stage.stage_key] = stage.model_dump()
    return [by_key[key] for key in sorted(by_key)]


def _claim_input_from_record(record) -> ExtractedClaimInput:
    metadata = {
        **record.claim.metadata,
        "claim_kind": record.claim.claim_kind,
        "subject_ref": record.claim.subject_ref,
        "scope": record.claim.scope,
        "authority": record.claim.authority,
        "belief_confidence": record.claim.belief_confidence,
        "valid_from": record.claim.valid_from,
        "valid_until": record.claim.valid_until,
        "observed_at": record.claim.observed_at,
        "temporal_refs": [item.model_dump() for item in record.temporal_refs],
        "quantity_claims": [item.model_dump() for item in record.quantity_claims],
    }
    return ExtractedClaimInput(
        text=record.text,
        session_id=str(record.claim.metadata.get("session_id") or "graph"),
        session_date=str(record.claim.metadata.get("session_date") or record.claim.observed_at or ""),
        session_idx=int(record.claim.metadata.get("session_idx") or 0),
        role=str(record.claim.metadata.get("role") or "unknown"),
        confidence=record.claim.extraction_confidence,
        source=str(record.claim.metadata.get("source") or "graph_replay"),
        metadata=metadata,
        source_turn_ids=tuple(record.source_turn_ids),
    )


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
        target = materialization.source_turn_object_ids.get(turn_id)
        if target:
            _ensure_relation(graph, source_id, target, "memory_grounded_in")
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
