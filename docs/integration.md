# Integration Notes

## With `memory_gateway`

This pack should request retrieval by creating or shaping data for `memory_retrieval_request`. It should not bypass the gateway backend protocol.

Use:

```python
from activegraph_memory.gateway import build_memory_retrieval_request
from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query

query = MemoryQuery(query="Did we ever agree to exclusivity?")
plan = plan_query(query, query_id="q1")
request_data = build_memory_retrieval_request(query, plan)
```

Then write `request_data` as a `memory_retrieval_request` when `memory_gateway` is loaded.

## With `core`

Do not redefine Core objects:

- `source`
- `observation`
- `memory_candidate`
- `evaluation`

Use Core provenance:

```text
source -> grounds -> observation
observation -> proposes -> memory_candidate
memory_candidate -> accepted_as -> memory_item
memory_claim -> derived_from -> observation/source/memory_item
```

`derived_from` belongs to Core.

## With Standalone Agents

The pure helper modules can be used without a full ActiveGraph runtime after package dependencies are installed:

- `activegraph_memory.planner`
- `activegraph_memory.coverage`
- `activegraph_memory.scoring`
- `activegraph_memory.temporal`
- `activegraph_memory.gateway`

The package still depends on ActiveGraph because the exported pack and object declarations use ActiveGraph pack APIs.
