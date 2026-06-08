"""
Smoke tests para o pipeline Pro Git Q&A.
Não fazem chamadas LLM reais, testam lógica local (routing, cache, ingestão mock).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# Routing 
def test_routing_simple_question():
    from src.pipeline.routing import route, LLM_MODEL_CHEAP
    model = route("O que é git?")
    assert model == LLM_MODEL_CHEAP


def test_routing_complex_question():
    from src.pipeline.routing import route, LLM_MODEL_MAIN
    model = route("Por que o git rebase é considerado mais arriscado que o merge em branches compartilhadas?")
    assert model == LLM_MODEL_MAIN


def test_routing_stats_increment():
    from src.pipeline.routing import route, routing_stats
    before = routing_stats()["total"]
    route("Como fazer commit?")
    after = routing_stats()["total"]
    assert after == before + 1


# Cache
def test_cache_miss_on_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_THRESHOLD", "0.92")
    # Forçar diretório temporário
    import src.pipeline.cache as cache_mod
    cache_mod.CACHE_DIR = tmp_path / "cache"
    cache_mod._cache_col = None
    cache_mod._cache_client = None

    result = cache_mod.cache_get("pergunta que nunca foi feita antes xyz123")
    assert result is None


def test_cache_set_and_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_THRESHOLD", "0.99")
    import src.pipeline.cache as cache_mod
    cache_mod.CACHE_DIR = tmp_path / "cache2"
    cache_mod._cache_col = None
    cache_mod._cache_client = None
    cache_mod._DIST_THRESHOLD = 0.01  # threshold bem restrito

    cache_mod.cache_set("O que é git staging area?", "É a área de preparação.")
    # Query idêntica deve dar hit
    result = cache_mod.cache_get("O que é git staging area?")
    assert result is not None
    assert result["cache_hit"] is True


def test_cache_stats_structure(tmp_path, monkeypatch):
    import src.pipeline.cache as cache_mod
    cache_mod.CACHE_DIR = tmp_path / "cache3"
    cache_mod._cache_col = None
    cache_mod._cache_client = None
    cache_mod._stats = {"hits": 3, "misses": 7}

    stats = cache_mod.cache_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 7
    assert stats["total"] == 10
    assert stats["hit_rate"] == 0.3


# RAG (mock LLM) 
def test_retrieve_returns_list(tmp_path, monkeypatch):
    """Testa que retrieve retorna lista mesmo com collection vazia."""
    import chromadb
    import src.pipeline.rag as rag_mod

    # Substituir collection por mock in-memory
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["chunk de teste"]],
        "metadatas": [[{"source": "test.pdf", "page": 1, "chapter": 1}]],
        "distances": [[0.1]],
    }
    rag_mod._collection = mock_col

    hits = rag_mod.retrieve("O que é git?", k=1)
    assert isinstance(hits, list)
    assert len(hits) == 1
    assert "text" in hits[0]
    assert "chapter" in hits[0]


def test_rag_answer_structure(monkeypatch):
    # Testa estrutura do retorno de rag_answer com LLM mockado
    import src.pipeline.rag as rag_mod

    mock_hits = [{"text": "Git é um VCS.", "source": "progit.pdf", "page": 1, "chapter": 1, "distance": 0.1}]
    monkeypatch.setattr(rag_mod, "retrieve", lambda q, k: mock_hits)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Git é um sistema de controle de versão."
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 120

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    monkeypatch.setattr(rag_mod, "get_llm_client", lambda: mock_client)

    result = rag_mod.rag_answer("O que é git?")
    assert "answer" in result
    assert "sources" in result
    assert "usage" in result
    assert result["usage"]["total_tokens"] == 120
