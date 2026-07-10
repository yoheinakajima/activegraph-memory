"""Optional durable stores for compiled-memory embedding vectors."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingVectorStore(Protocol):
    """Persistence seam used by the fielded embedding retriever."""

    def get(self, key: str) -> list[float] | None: ...

    def put(self, key: str, vector: list[float], metadata: dict[str, Any]) -> None: ...


class SQLiteEmbeddingStore:
    """Small process-safe embedding cache with deterministic keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                embedding_key TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self.connection.commit()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, key: str) -> list[float] | None:
        row = self.connection.execute(
            "SELECT vector_json FROM memory_embeddings WHERE embedding_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return [float(value) for value in json.loads(row[0])]

    def put(self, key: str, vector: list[float], metadata: dict[str, Any]) -> None:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO memory_embeddings
                (embedding_key, vector_json, metadata_json)
            VALUES (?, ?, ?)
            """,
            (
                key,
                json.dumps(vector, separators=(",", ":")),
                json.dumps(metadata, ensure_ascii=True, sort_keys=True, default=str),
            ),
        )
        self.connection.commit()
        self.writes += int(cursor.rowcount > 0)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteEmbeddingStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


class GraphEmbeddingStore:
    """Persist vectors as idempotent, graph-visible ``memory_embedding`` objects."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self._objects: dict[str, Any] | None = None
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, key: str) -> list[float] | None:
        obj = self._index().get(key)
        if obj is None:
            self.misses += 1
            return None
        self.hits += 1
        return [float(value) for value in obj.data.get("vector", [])]

    def put(self, key: str, vector: list[float], metadata: dict[str, Any]) -> None:
        if key in self._index():
            return
        obj = self.graph.add_object(
            "memory_embedding",
            {
                "embedding_key": key,
                "subject_kind": str(metadata.get("field") or "unknown"),
                "subject_key": str(metadata.get("subject_id") or ""),
                "model": str(metadata.get("model") or ""),
                "text_sha256": str(metadata.get("text_sha256") or ""),
                "dimensions": len(vector),
                "vector": vector,
                "metadata": metadata,
            },
        )
        self._index()[key] = obj
        self.writes += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}

    def _index(self) -> dict[str, Any]:
        if self._objects is None:
            self._objects = {
                str(obj.data.get("embedding_key")): obj
                for obj in self.graph.objects(type="memory_embedding")
            }
        return self._objects
