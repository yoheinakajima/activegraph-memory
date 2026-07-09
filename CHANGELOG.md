# Changelog

## 0.1.0

- Initial ActiveGraph semantic memory pack.
- Added Pydantic schemas and ActiveGraph object/relation type declarations.
- Added deterministic query classification and retrieval planning.
- Added compiled entity/category/event projection for graph-query reducers.
- Added deterministic count, sum, and chronological reducers during retrieval.
- Increased the standalone retrieval default budget to 10000 rough tokens.
- Added temporal order comparison over named operands, including explicit
  insufficient-evidence packets when a comparison operand is missing.
- Added month-name date normalization for dates like "February 25th" anchored
  to the source/session year.
- Added compact preference/advice observation packets for recommendation-style
  queries, grounded in user claims and source ids.
- Added concept expansion for device/accessory and business-milestone queries
  so retrieval can bridge wording gaps without benchmark-specific rules.
- Added a compact near-date source packet for relative-date lookups when graph
  context is weak or unavailable.
- Added exact assistant-source recall routing for questions about what the
  assistant previously said, listed, recommended, or provided.
- Render low-confidence graph reducers as evidence rows without an authoritative
  computed answer candidate.
- Hardened graph-query matching for negated events, phrase punctuation,
  comma/word quantities, two-date windows, repeated event counts, and tight
  token budgets.
- Hardened temporal sequence planning so "earliest to latest" order questions
  are not treated as latest/current queries, and sequence "order" is not
  mistaken for purchase/order events.
- Added coverage and confidence helpers.
- Added graph-visible `memory_query_planner` behavior.
- Added gateway adapter helpers for creating `memory_retrieval_request` data.
- Added docs, fixtures, examples, and offline tests.
