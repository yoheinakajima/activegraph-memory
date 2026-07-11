"""Graph-native ingestion of the shared annotation layer (ADR 0026 steps 5-7).

The pack form consumes ``semantic_annotation`` records (from the shared
``semantic_extraction`` pack) and canonical ``entity`` ids (from the
``entity`` pack) instead of running memory's own extractor. Composition
is graph-object-only: this module never imports activegraph-packs; it
reacts to object types by name.

Behaviors:

* ``memory_ingest_shared_annotation`` — on each claim-bearing
  ``semantic_annotation``, mint one ``memory_claim`` grounded in the
  annotation's evidence (consuming canonical entity ids), and record the
  covering ``memory_ingestion_stage`` so proof completeness reads the
  shared extraction run's coverage. Idempotent by shared annotation id.

* ``memory_deprecate_entity`` — on each canonical ``entity`` created,
  map any matching ``memory_entity`` to it (``status="mapped"``,
  ``canonical_entity_id`` set) — never silently dropped.
"""

from __future__ import annotations

import hashlib
from typing import Any

from activegraph.packs import behavior

from .settings import ActiveGraphMemorySettings
from .shared_extraction import CLAIM_FACETS, ENTITY_FACET

_INGESTION_STAGE_EXTRACTOR_PREFIX = "shared:"


def _event_object(event) -> dict[str, Any]:
    payload = getattr(event, "payload", None) or {}
    return payload.get("object", {}) or {}


def _stable(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:32]}"


def _patch(graph, target: str, updates: dict[str, Any], *, rationale: str) -> None:
    """Patch through either graph surface (BehaviorGraph takes no rationale)."""
    try:
        graph.patch_object(target, updates, rationale=rationale)
    except TypeError:
        graph.patch_object(target, updates)


def _canonical_entity_refs(view, evidence_id: str) -> list[str]:
    """Canonical `entity` ids resolved for one evidence's mentions.

    Reads entity_mention annotations on the evidence and follows each to
    its resolved canonical id (``metadata.canonical_entity_id`` /
    ``entity_id``). Canonical resolution stays owned by the entity pack;
    memory only consumes the ids.
    """
    refs: list[str] = []
    for obj in view.objects(type="semantic_annotation"):
        data = obj.data
        if data.get("facet") != ENTITY_FACET or data.get("evidence_id") != evidence_id:
            continue
        metadata = data.get("metadata") or {}
        canonical = metadata.get("canonical_entity_id") or metadata.get("entity_id")
        if canonical and canonical not in refs:
            refs.append(str(canonical))
    return refs


@behavior(
    name="memory_ingest_shared_annotation",
    on=["object.created"],
    where={"object.type": "semantic_annotation"},
    view={
        "include_types": [
            "semantic_annotation",
            "memory_claim",
            "memory_ingestion_stage",
        ]
    },
    creates=["memory_claim", "memory_ingestion_stage"],
)
def memory_ingest_shared_annotation(
    event, graph, ctx, *, settings: ActiveGraphMemorySettings
):
    """Mint a memory_claim from one shared claim-bearing annotation."""
    if not settings.consume_shared_extraction:
        return None
    wrapper = _event_object(event)
    data = wrapper.get("data", {}) or {}
    facet = data.get("facet")
    if facet not in CLAIM_FACETS or data.get("status") not in (None, "active"):
        return None
    body = data.get("body") or {}
    text = body.get("text") or (data.get("selector") or {}).get("exact")
    evidence_id = data.get("evidence_id")
    if not text or not evidence_id:
        return None
    annotation_id = data.get("annotation_identity") or wrapper.get("id")

    # Idempotent: one memory_claim per shared annotation identity.
    for existing in ctx.view.objects(type="memory_claim"):
        if (existing.data.get("metadata") or {}).get("shared_annotation_id") == annotation_id:
            return None

    entity_refs = _canonical_entity_refs(ctx.view, evidence_id)
    polarity = "negative" if data.get("polarity") == "negative" else "affirmative"
    claim = graph.add_object(
        "memory_claim",
        {
            "text": str(text),
            "claim_kind": "preference" if facet == "preference_expression" else "unknown",
            "confidence": float(data.get("confidence") or 0.6),
            "extraction_confidence": float(data.get("confidence") or 0.6),
            "source_ids": [str(evidence_id)],
            "observation_ids": [wrapper.get("id")],
            "observed_at": data.get("observation_time"),
            "valid_from": data.get("event_time"),
            "metadata": {
                "shared_annotation_id": annotation_id,
                "shared_extractor_id": data.get("extractor_id"),
                "shared_extractor_version": data.get("extractor_version"),
                "facet": facet,
                "polarity": polarity,
                "canonical_entity_ids": entity_refs,
                "role": (data.get("author_role") or "unknown"),
            },
        },
    )
    try:
        graph.add_relation(claim.id, evidence_id, "memory_grounded_in")
    except Exception:
        pass

    _record_ingestion_stage(
        graph, ctx.view, data.get("extractor_id") or "shared", str(evidence_id)
    )
    return claim


def _record_ingestion_stage(graph, view, extractor_id: str, evidence_id: str) -> None:
    """Record/extend the shared extraction run's source coverage.

    One stage per shared extractor id; its ``source_ids`` accumulate the
    evidence the shared run covered so proof completeness (ADR 0026 step
    6) reads the real extraction-run coverage.
    """
    extractor = f"{_INGESTION_STAGE_EXTRACTOR_PREFIX}{extractor_id}"
    stage_key = _stable("ingestion", extractor)
    for obj in view.objects(type="memory_ingestion_stage"):
        if obj.data.get("stage_key") == stage_key:
            source_ids = list(obj.data.get("source_ids") or [])
            if evidence_id in source_ids:
                return
            source_ids.append(evidence_id)
            _patch(
                graph,
                obj.id,
                {"source_ids": source_ids, "fact_count": len(source_ids)},
                rationale="extend shared extraction-run coverage",
            )
            return
    stage = graph.add_object(
        "memory_ingestion_stage",
        {
            "stage_key": stage_key,
            "operation": "shared_annotation_ingest",
            "extractor": extractor,
            "model": "",
            "source_ids": [evidence_id],
            "fact_count": 1,
            "cached": True,
            "metadata": {"shared_extraction": True},
        },
    )
    try:
        graph.add_relation(stage.id, evidence_id, "memory_grounded_in")
    except Exception:
        pass


@behavior(
    name="memory_deprecate_entity",
    on=["object.created"],
    where={"object.type": "entity"},
    view={"include_types": ["entity", "memory_entity"]},
    creates=[],
)
def memory_deprecate_entity(
    event, graph, ctx, *, settings: ActiveGraphMemorySettings
):
    """Map a legacy memory_entity to a newly-resolved canonical entity.

    ADR 0026 step 5: memory_entity is deprecated; existing objects are
    mapped to canonical ids, never silently dropped. Matching is by
    normalized name/alias overlap — conservative, and only ever sets a
    mapping (it never deletes).
    """
    if not settings.deprecate_memory_entity:
        return None
    wrapper = _event_object(event)
    entity_id = wrapper.get("id")
    entity_data = wrapper.get("data", {}) or {}
    labels = {
        _normalize(entity_data.get("name", "")),
        *(_normalize(alias) for alias in entity_data.get("aliases", [])),
    }
    labels.discard("")
    if not labels:
        return None
    for obj in ctx.view.objects(type="memory_entity"):
        data = obj.data
        if data.get("status") in ("mapped", "superseded"):
            continue
        candidate_labels = {
            _normalize(data.get("canonical_name", "")),
            *(_normalize(alias) for alias in data.get("aliases", [])),
        }
        if labels & candidate_labels:
            _patch(
                graph,
                obj.id,
                {"status": "mapped", "canonical_entity_id": entity_id},
                rationale="ADR 0026: memory_entity mapped to canonical entity",
            )
    return None


def _normalize(value: str) -> str:
    return " ".join(str(value).lower().split())


SHARED_INGESTION_BEHAVIORS = [
    memory_ingest_shared_annotation,
    memory_deprecate_entity,
]

__all__ = ["SHARED_INGESTION_BEHAVIORS"]
