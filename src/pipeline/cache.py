"""
Cache semântico para o Pro Git Q&A.
TODO 3 implementado: armazena pares (query, resposta) no Chroma com embedding local.
Hit quando distância cosseno < CACHE_THRESHOLD (padrão 0.08, equivale a ~0.92 similaridade).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("data/cache")
CACHE_COLLECTION = "semantic_cache"
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.92"))
# Chroma usa distância cosseno: threshold em distância = 1 - similaridade
_DIST_THRESHOLD = 0.35

EMBED_MODEL = "all-MiniLM-L6-v2"

_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


class CacheEmbedFn(EmbeddingFunction):
    def __init__(self):
        self._model = _get_st_model()

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), show_progress_bar=False).tolist()


_cache_client: chromadb.PersistentClient | None = None
_cache_col: chromadb.Collection | None = None

# Estatísticas em memória
_stats = {"hits": 0, "misses": 0}


def _get_cache_col() -> chromadb.Collection:
    global _cache_client, _cache_col
    if _cache_col is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_client = chromadb.PersistentClient(path=str(CACHE_DIR))
        _cache_col = _cache_client.get_or_create_collection(
            name=CACHE_COLLECTION,
            embedding_function=CacheEmbedFn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _cache_col


def cache_get(query: str) -> dict[str, Any] | None:
    # Busca resposta em cache para a query
    # Retorna dict com answer e sources se hit, None se miss
    
    col = _get_cache_col()
    if col.count() == 0:
        _stats["misses"] += 1
        return None

    result = col.query(query_texts=[query], n_results=1)
    distance = result["distances"][0][0] if result["distances"][0] else 1.0

    if distance <= _DIST_THRESHOLD:
        _stats["hits"] += 1
        meta = result["metadatas"][0][0]
        return {
            "answer": meta["answer"],
            "sources": [],
            "model": meta.get("model", "cache"),
            "cache_hit": True,
            "cache_distance": distance,
            "cached_query": result["documents"][0][0],
        }

    _stats["misses"] += 1
    return None


def cache_set(query: str, answer: str, model: str = "") -> None:
    # Armazena a resposta no cache semântico
    col = _get_cache_col()
    entry_id = f"cache-{int(time.time() * 1000)}"
    col.add(
        ids=[entry_id],
        documents=[query],
        metadatas=[{"answer": answer, "model": model, "query": query}],
    )


def cache_stats() -> dict[str, Any]:
    # Retorna estatísticas de hit/miss do cache
    total = _stats["hits"] + _stats["misses"]
    hit_rate = _stats["hits"] / total if total > 0 else 0.0
    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "total": total,
        "hit_rate": round(hit_rate, 3),
        "cache_size": _get_cache_col().count(),
        "threshold": CACHE_THRESHOLD,
    }


def cache_clear() -> None:
    # Limpa todo o cache (útil em testes)
    col = _get_cache_col()
    all_ids = col.get()["ids"]
    if all_ids:
        col.delete(ids=all_ids)
    _stats["hits"] = 0
    _stats["misses"] = 0
