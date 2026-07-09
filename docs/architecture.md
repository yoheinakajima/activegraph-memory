# Architecture

`activegraph-memory` is the semantic memory layer above ActiveGraph Core and `memory_gateway`.

Core remains the universal substrate:

```text
source -> observation -> memory_candidate -> evaluation
```

`memory_gateway` remains the durable memory lifecycle and backend seam:

```text
memory_candidate -> memory_item -> memory_retrieval_request -> memory_retrieval
```

This pack adds the epistemic layer:

```text
memory_query -> retrieval_plan -> evidence_bundle -> coverage_report -> memory_answer
      |               |                  |
      |               |                  -> memory_used_as_evidence
      |               -> memory_gateway_request metadata
      -> memory_has_temporal_ref
```

## Design Principles

- Vector search is a candidate generator, not proof.
- Retrieval is not the final step; it produces evidence to verify.
- Latest, current, final, negative, aggregate, and multi-hop questions need different plans.
- Answers should report coverage and uncertainty rather than sounding definitive by default.
- Supersession and contradiction are graph states, not hidden resolver side effects.

## Layers

1. Raw sources stay in Core `source` objects.
2. Observations stay in Core `observation` objects.
3. Candidate memories flow through Core and `memory_gateway`.
4. Semantic claims live in `memory_claim`.
5. Coherent histories live in `memory_episode`.
6. Retrieval work is made explicit through `memory_query` and `retrieval_plan`.
7. Evidence and uncertainty are represented by `evidence_bundle`, `coverage_report`, and `memory_answer`.

## First Release Scope

Version 0.1 is deterministic:

- Pydantic schemas
- ActiveGraph object and relation types
- deterministic query classification
- retrieval plan generation
- source-turn and extracted-claim compiler
- evidence-bundle retrieval/assembly runtime
- coverage and confidence helpers
- one graph-visible planning behavior

It intentionally does not own connector-specific extraction, automatic
conflict graph mutation, or external connector logic yet. The runtime accepts
claims produced by upstream extractors and compiles them into a source-grounded
memory index.
