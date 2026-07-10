"""Semantic and epistemic memory pack for ActiveGraph."""

from __future__ import annotations

from pathlib import Path

from activegraph.packs import Pack, load_prompts_from_dir

from .behaviors import BEHAVIORS
from .constants import PACK_NAME, PACK_VERSION
from .compiler import (
    CategoryRef,
    EntityRef,
    ExtractedClaimInput,
    MemoryEventRecord,
    MemoryIndex,
    SourceTurn,
    compile_memory_index,
)
from .benchmarking import (
    MemoryBenchmarkCase,
    MemoryBenchmarkResult,
    benchmark_profiles,
    benchmark_runtime,
    render_benchmark_markdown,
)
from .graph_runtime import (
    GraphMemoryRepository,
    materialize_memory_index,
    materialize_retrieval_trace,
)
from .embedding_store import EmbeddingVectorStore, GraphEmbeddingStore, SQLiteEmbeddingStore
from .graph_query import GraphQueryResult, run_graph_query
from .object_types import OBJECT_TYPES, RELATION_TYPES
from .profiles import (
    MemoryRuntimeProfile,
    StageReasoningPolicy,
    profile_from_settings,
    runtime_profile,
    runtime_profiles,
)
from .query_ir import QueryAnalysis, analyze_query
from .ranking import EmbeddingSignalProvider, RetrievalSignals
from .reasoning import (
    ActiveGraphLLMReasoningBackend,
    ReasoningBackend,
    ReasoningRequest,
    ReasoningResponse,
)
from .retrieval import MemoryRetrievalResult, retrieve_memory
from .runtime import MemoryRuntime
from .settings import ActiveGraphMemorySettings
from .tools import TOOLS

_PROMPTS_DIR = Path(__file__).parent / "prompts"

pack = Pack(
    name=PACK_NAME,
    version=PACK_VERSION,
    description=(
        "Semantic and epistemic memory layer for ActiveGraph. Adds claims, "
        "episodes, temporal references, retrieval plans, coverage reports, "
        "confidence vectors, and evidence-backed answers while composing "
        "with core and memory_gateway."
    ),
    object_types=OBJECT_TYPES,
    relation_types=RELATION_TYPES,
    behaviors=BEHAVIORS,
    tools=TOOLS,
    policies=(),
    prompts=load_prompts_from_dir(_PROMPTS_DIR) if _PROMPTS_DIR.exists() else (),
    settings_schema=ActiveGraphMemorySettings,
)

__all__ = [
    "pack",
    "ActiveGraphMemorySettings",
    "ActiveGraphLLMReasoningBackend",
    "CategoryRef",
    "EmbeddingSignalProvider",
    "EmbeddingVectorStore",
    "EntityRef",
    "ExtractedClaimInput",
    "GraphQueryResult",
    "GraphMemoryRepository",
    "GraphEmbeddingStore",
    "MemoryEventRecord",
    "MemoryIndex",
    "MemoryBenchmarkCase",
    "MemoryBenchmarkResult",
    "MemoryRuntime",
    "MemoryRuntimeProfile",
    "MemoryRetrievalResult",
    "SourceTurn",
    "SQLiteEmbeddingStore",
    "QueryAnalysis",
    "ReasoningBackend",
    "ReasoningRequest",
    "ReasoningResponse",
    "RetrievalSignals",
    "StageReasoningPolicy",
    "analyze_query",
    "benchmark_profiles",
    "benchmark_runtime",
    "render_benchmark_markdown",
    "compile_memory_index",
    "retrieve_memory",
    "profile_from_settings",
    "runtime_profile",
    "runtime_profiles",
    "run_graph_query",
    "materialize_memory_index",
    "materialize_retrieval_trace",
]
