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
from .graph_query import GraphQueryResult, run_graph_query
from .object_types import OBJECT_TYPES, RELATION_TYPES
from .retrieval import MemoryRetrievalResult, retrieve_memory
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
    "CategoryRef",
    "EntityRef",
    "ExtractedClaimInput",
    "GraphQueryResult",
    "MemoryEventRecord",
    "MemoryIndex",
    "MemoryRetrievalResult",
    "SourceTurn",
    "compile_memory_index",
    "retrieve_memory",
    "run_graph_query",
]
