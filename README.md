# activegraph-memory

`activegraph-memory` is a semantic and epistemic memory pack for [ActiveGraph](https://pypi.org/project/activegraph/).

It sits above `activegraph-packs/packs/memory_gateway`. The gateway owns the low-level lifecycle:

```text
memory_candidate -> evaluation -> memory_item -> retrieval/backend
```

This package owns the higher-level memory layer:

- claim and episode objects
- temporal validity and quantity claims
- supersession and contradiction relations
- deterministic query planning
- coverage reports
- confidence vectors
- evidence-backed memory answers

The first release is intentionally deterministic. It defines the object model, relation model, planner utilities, coverage/confidence helpers, and a graph-visible query-planning behavior. It does not perform LLM-backed extraction and does not replace mem0, Zep, pgvector, SQLite, or the existing memory backend seam.

## Install

```bash
pip install -e ".[dev]"
```

## Pack Entry Point

After installation, ActiveGraph can load the pack by entry point:

```python
from activegraph_memory import ActiveGraphMemorySettings, pack

settings = ActiveGraphMemorySettings()
```

The package registers:

```toml
[project.entry-points."activegraph.packs"]
activegraph_memory = "activegraph_memory:pack"
```

## What This Pack Adds

Objects:

- `memory_claim`
- `memory_episode`
- `memory_query`
- `retrieval_plan`
- `coverage_report`
- `evidence_bundle`
- `memory_answer`
- `temporal_ref`
- `quantity_claim`
- `memory_policy`
- `memory_evaluation`

Relations:

- `memory_supports`
- `memory_contradicts`
- `memory_supersedes`
- `memory_has_temporal_ref`
- `memory_has_quantity`
- `memory_retrieved_for`
- `memory_used_as_evidence`
- `memory_has_coverage`
- `memory_governed_by_policy`

Core provenance relations such as `derived_from` stay in Core and are not redefined here.

## Deterministic Planning

The planner turns `memory_query` data into a `retrieval_plan`:

```python
from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query

query = MemoryQuery(query="What is the latest launch plan?")
plan = plan_query(query)

assert plan.requires_freshness is True
assert "supersession_scan" in plan.strategies
```

The pack also registers a `memory_query_planner` behavior. When loaded in an ActiveGraph runtime, creating a `memory_query` can create a graph-visible `retrieval_plan`.

## Behavior Map

| Behavior | Trigger | Output | Notes |
| --- | --- | --- | --- |
| `memory_query_planner` | `memory_query.created` | `retrieval_plan` | Deterministic, offline, no API key. |

Future behavior layers should add claim extraction, temporal resolution, gateway integration, coverage checks, and answer synthesis as graph-visible steps.

## Gateway Boundary

This pack should compose with `core` and `memory_gateway`:

```text
source -> observation -> memory_candidate -> evaluation -> memory_item
                                              |
memory_query -> retrieval_plan -> memory_retrieval_request -> memory_retrieval
                         |
                         -> evidence_bundle -> coverage_report -> memory_answer
```

Retrieval remains auditable because plans and evidence are represented as graph objects instead of hidden function calls.

## Validation

```bash
pytest -q
```

Tests run offline with no API key and no live network.

## Research Posture

The design follows the idea that vector search is only one access path. Strong memory needs continuity, provenance, belief maintenance, and epistemic reporting:

- semantic recall finds candidate evidence
- ActiveGraph preserves chronology and relationships
- structured metadata handles freshness, authority, and coverage
- answer contracts report uncertainty instead of hiding it

This repo is a foundation for that direction, not a benchmark claim.
