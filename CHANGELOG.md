# Changelog

## 0.1.0

- Initial ActiveGraph semantic memory pack.
- Added Pydantic schemas and ActiveGraph object/relation type declarations.
- Added deterministic query classification and retrieval planning.
- Added compiled entity/category/event projection for graph-query reducers.
- Added deterministic count, sum, and chronological reducers during retrieval.
- Increased the standalone retrieval default budget to 10000 rough tokens.
- Hardened graph-query matching for negated events, phrase punctuation,
  comma/word quantities, two-date windows, repeated event counts, and tight
  token budgets.
- Added coverage and confidence helpers.
- Added graph-visible `memory_query_planner` behavior.
- Added gateway adapter helpers for creating `memory_retrieval_request` data.
- Added docs, fixtures, examples, and offline tests.
