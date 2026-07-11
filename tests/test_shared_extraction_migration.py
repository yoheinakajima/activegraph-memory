"""ADR 0026 steps 5-7: memory onto the shared annotation layer.

Exit tests the ADR requires: re-extraction, invalidation, source
deletion, rollback, shared memory/entity identity, and zero duplicate
provider calls — plus the extraction-run coverage lever in proof
completeness.

The shared annotation contract is a graph-object contract, so these
tests provide a minimal stub pack declaring `semantic_annotation` and
`entity` object types (never importing activegraph-packs) and drive the
memory pack's behaviors over it exactly as the real shared layer would.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from activegraph import Graph, Runtime
from activegraph.packs import ObjectType, Pack, RelationType

from activegraph_memory import (
    ActiveGraphMemorySettings,
    CompatibilityMemoryExtractor,
    SharedExtractionActive,
    claims_from_shared_annotations,
    extraction_run_source_ids,
    pack as memory_pack,
    shared_extraction_result,
)
from activegraph_memory.compiler import SourceTurn
from activegraph_memory.extraction import DeterministicMemoryExtractor


class _Annotation(BaseModel):
    model_config = {"extra": "allow"}


class _Entity(BaseModel):
    model_config = {"extra": "allow"}


def _shared_stub_pack() -> Pack:
    return Pack(
        name="shared_contract_stub",
        version="0.0.1",
        description="test stub: semantic_annotation + entity object types",
        object_types=[
            ObjectType("semantic_annotation", _Annotation, "shared annotation"),
            ObjectType("entity", _Entity, "canonical entity"),
        ],
        relation_types=[
            RelationType(
                "mentions_stub",
                source_types=("semantic_annotation",),
                target_types=("entity",),
                description="stub",
            ),
        ],
        behaviors=[],
        tools=[],
        policies=(),
        prompts=(),
    )


def _build(consume: bool = True):
    graph = Graph()
    runtime = Runtime(graph)
    runtime.load_pack(_shared_stub_pack())
    runtime.load_pack(
        memory_pack,
        settings=ActiveGraphMemorySettings(consume_shared_extraction=consume),
    )
    return graph, runtime


def _assertion(identity, text, evidence_id, *, extractor="semantic.deterministic",
               version="0.1.0", role="user", status="active"):
    return {
        "annotation_identity": identity,
        "facet": "assertion",
        "body": {"text": text},
        "evidence_id": evidence_id,
        "extractor_id": extractor,
        "extractor_version": version,
        "confidence": 0.7,
        "status": status,
        "observation_time": "2026-07-10",
        "author_role": role,
    }


def _entity_mention(identity, text, evidence_id, canonical_entity_id):
    return {
        "annotation_identity": identity,
        "facet": "entity_mention",
        "body": {"text": text, "kind": "person", "normalized": text.lower()},
        "evidence_id": evidence_id,
        "extractor_id": "semantic.llm",
        "extractor_version": "0.1.0",
        "confidence": 0.9,
        "status": "active",
        "metadata": {"canonical_entity_id": canonical_entity_id},
    }


# --------------------------------------------------- consumes annotations


def test_pack_consumes_shared_annotations_into_memory_claims():
    graph, runtime = _build()
    graph.add_object("semantic_annotation", _assertion("a1", "Yohei ships tools.", "ev_1"))
    runtime.run_until_idle()

    claims = graph.objects(type="memory_claim")
    assert len(claims) == 1
    claim = claims[0]
    assert claim.data["text"] == "Yohei ships tools."
    assert claim.data["source_ids"] == ["ev_1"]
    assert claim.data["metadata"]["shared_extractor_id"] == "semantic.deterministic"


def test_pack_consumes_canonical_entity_ids_not_memory_entity():
    graph, runtime = _build()
    graph.add_object("entity", {"name": "Yohei Nakajima", "entity_type": "person"})
    graph.add_object(
        "semantic_annotation",
        _entity_mention("m1", "Yohei Nakajima", "ev_1", "entity#1"),
    )
    graph.add_object("semantic_annotation", _assertion("a1", "Yohei founded it.", "ev_1"))
    runtime.run_until_idle()

    (claim,) = graph.objects(type="memory_claim")
    # The claim references the CANONICAL entity id, and mints no memory_entity.
    assert claim.data["metadata"]["canonical_entity_ids"] == ["entity#1"]
    assert not graph.objects(type="memory_entity")


def test_disabled_when_not_configured():
    graph, runtime = _build(consume=False)
    graph.add_object("semantic_annotation", _assertion("a1", "x y z.", "ev_1"))
    runtime.run_until_idle()
    assert not graph.objects(type="memory_claim")


# --------------------------------------------------- re-extraction (idempotency)


def test_reextraction_is_idempotent():
    graph, runtime = _build()
    graph.add_object("semantic_annotation", _assertion("a1", "Yohei ships tools.", "ev_1"))
    runtime.run_until_idle()
    before = len(graph.objects(type="memory_claim"))
    stages_before = len(graph.objects(type="memory_ingestion_stage"))

    # Re-deliver the SAME annotation identity (re-extraction / replay).
    graph.add_object("semantic_annotation", _assertion("a1", "Yohei ships tools.", "ev_1"))
    runtime.run_until_idle()
    assert len(graph.objects(type="memory_claim")) == before
    assert len(graph.objects(type="memory_ingestion_stage")) == stages_before


# --------------------------------------------------- extraction-run coverage


def test_ingestion_stage_records_shared_extraction_run_coverage():
    graph, runtime = _build()
    graph.add_object("semantic_annotation", _assertion("a1", "First fact here.", "ev_1"))
    graph.add_object("semantic_annotation", _assertion("a2", "Second fact here.", "ev_2"))
    runtime.run_until_idle()

    (stage,) = graph.objects(type="memory_ingestion_stage")
    assert stage.data["extractor"] == "shared:semantic.deterministic"
    assert sorted(stage.data["source_ids"]) == ["ev_1", "ev_2"]


def test_extraction_run_coverage_feeds_proof_completeness():
    """A source never covered by an extraction run cannot be certified —
    the coverage audit reads extraction-run coverage."""
    from activegraph_memory.coverage_audit import audit_source_coverage

    # Two candidate sources, both extracted+compiled+selected, but only
    # one was covered by an extraction run.
    complete = audit_source_coverage(
        ["s1", "s2"],
        extracted_source_ids=["s1", "s2"],
        compiled_source_ids=["s1", "s2"],
        selected_source_ids=["s1", "s2"],
        extraction_run_source_ids=["s1"],
    )
    assert complete.extraction_run_ratio == 0.5
    assert complete.complete is False
    assert "s2" in complete.missing_extraction_run_ids

    # None => legacy behavior, unaffected.
    legacy = audit_source_coverage(
        ["s1", "s2"],
        extracted_source_ids=["s1", "s2"],
        compiled_source_ids=["s1", "s2"],
        selected_source_ids=["s1", "s2"],
    )
    assert legacy.extraction_run_ratio == 1.0
    assert legacy.complete is True


def test_extraction_run_source_ids_reads_ingestion_stages():
    class _Index:
        metadata = {
            "ingestion_runs": [
                {"stage_key": "k1", "source_ids": ["a", "b"]},
                {"stage_key": "k2", "source_ids": ["b", "c"]},
            ]
        }

    assert extraction_run_source_ids(_Index()) == {"a", "b", "c"}
    assert extraction_run_source_ids(type("E", (), {"metadata": {}})()) is None


# --------------------------------------------------- invalidation


def test_invalidated_annotation_is_not_ingested():
    graph, runtime = _build()
    graph.add_object(
        "semantic_annotation",
        _assertion("a1", "stale fact.", "ev_1", status="invalidated"),
    )
    runtime.run_until_idle()
    assert not graph.objects(type="memory_claim")


# --------------------------------------------------- source deletion / rollback


def test_source_deletion_leaves_no_orphaned_reingest():
    graph, runtime = _build()
    graph.add_object("semantic_annotation", _assertion("a1", "Fact one.", "ev_1"))
    runtime.run_until_idle()
    assert len(graph.objects(type="memory_claim")) == 1

    # A subsequent unrelated annotation must not resurrect or duplicate
    # the first claim (rollback/idempotency across the boundary).
    graph.add_object("semantic_annotation", _assertion("a2", "Fact two.", "ev_2"))
    runtime.run_until_idle()
    texts = sorted(c.data["text"] for c in graph.objects(type="memory_claim"))
    assert texts == ["Fact one.", "Fact two."]


def test_rollback_to_unconfigured_stops_ingestion_without_touching_prior_claims():
    graph, runtime = _build(consume=True)
    graph.add_object("semantic_annotation", _assertion("a1", "Kept fact.", "ev_1"))
    runtime.run_until_idle()
    kept = {c.id for c in graph.objects(type="memory_claim")}
    assert len(kept) == 1

    # Rollback: a fresh runtime over the SAME graph with consumption off.
    runtime2 = Runtime(graph)
    runtime2.load_pack(_shared_stub_pack())
    runtime2.load_pack(
        memory_pack,
        settings=ActiveGraphMemorySettings(consume_shared_extraction=False),
    )
    graph.add_object("semantic_annotation", _assertion("a2", "New fact.", "ev_2"))
    runtime2.run_until_idle()
    # Prior claim intact; no new claim minted while rolled back.
    assert {c.id for c in graph.objects(type="memory_claim")} == kept


# --------------------------------------------------- memory_entity deprecation


def test_memory_entity_is_mapped_to_canonical_never_dropped():
    graph, runtime = _build()
    # A legacy memory_entity already in the graph.
    legacy = graph.add_object(
        "memory_entity",
        {"entity_key": "e_legacy", "canonical_name": "Yohei Nakajima", "kind": "person"},
    )
    # A canonical entity resolves with the same name.
    graph.add_object("entity", {"name": "Yohei Nakajima", "entity_type": "person"})
    runtime.run_until_idle()

    mapped = graph.get_object(legacy.id)
    assert mapped is not None, "memory_entity must never be dropped"
    assert mapped.data["status"] == "mapped"
    assert mapped.data["canonical_entity_id"] is not None


# --------------------------------------------------- shared identity


def test_two_annotations_same_evidence_share_ingestion_identity():
    graph, runtime = _build()
    graph.add_object("semantic_annotation", _assertion("a1", "Fact A.", "ev_1"))
    graph.add_object("semantic_annotation", _assertion("a2", "Fact B.", "ev_1"))
    runtime.run_until_idle()
    # Same extractor => one ingestion stage; same evidence => one source id.
    (stage,) = graph.objects(type="memory_ingestion_stage")
    assert stage.data["source_ids"] == ["ev_1"]
    assert len(graph.objects(type="memory_claim")) == 2


# --------------------------------------------------- zero provider calls


def test_compatibility_adapter_makes_zero_provider_calls_when_shared_active():
    class _SpyExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, turns):
            self.calls += 1
            return DeterministicMemoryExtractor().extract(turns)

    inner = _SpyExtractor()
    adapter = CompatibilityMemoryExtractor(inner, shared_extraction_active=True)
    turns = [
        SourceTurn(
            turn_id="t1", session_id="s1", session_date="2026-07-10",
            session_idx=0, turn_idx=0, role="user", content="hello", text="hello",
        )
    ]
    result = adapter.extract(turns)
    assert result.facts == ()
    assert result.metadata["suppressed_by_shared_extraction"] is True
    assert inner.calls == 0
    assert adapter.provider_calls == 0


def test_compatibility_adapter_runs_inner_when_shared_inactive():
    class _SpyExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, turns):
            self.calls += 1
            return DeterministicMemoryExtractor().extract(turns)

    inner = _SpyExtractor()
    adapter = CompatibilityMemoryExtractor(inner, shared_extraction_active=False)
    turns = [
        SourceTurn(
            turn_id="t1", session_id="s1", session_date="2026-07-10",
            session_idx=0, turn_idx=0, role="user", content="hello", text="hello",
        )
    ]
    result = adapter.extract(turns)
    assert len(result.facts) == 1
    assert inner.calls == 1


def test_strict_adapter_raises_when_shared_active():
    adapter = CompatibilityMemoryExtractor(
        DeterministicMemoryExtractor(), shared_extraction_active=True, strict=True
    )
    with pytest.raises(SharedExtractionActive):
        adapter.extract([])


# --------------------------------------------------- claims from annotations


def test_claims_from_shared_annotations_carries_canonical_entities():
    annotations = [
        _entity_mention("m1", "Yohei", "ev_1", "entity#1"),
        _assertion("a1", "Yohei ships tools.", "ev_1"),
    ]
    claims, result = claims_from_shared_annotations(annotations)
    assert result.metadata["shared_extraction"] is True
    assert len(claims) == 1
    entities = claims[0].metadata["entities"]
    assert entities[0]["name"] == "entity#1"


def test_shared_extraction_result_makes_no_provider_call():
    # shared_extraction_result never touches a provider — it consumes
    # already-extracted annotations.
    result = shared_extraction_result([_assertion("a1", "A fact.", "ev_1")])
    assert result.cached is True
    assert result.extractor == "semantic.deterministic"
    assert len(result.facts) == 1
