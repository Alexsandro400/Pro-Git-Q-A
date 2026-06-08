"""
Model routing cheap-first para o Pro Git Q&A.
TODO 5 implementado: classifica a complexidade da pergunta e roteia para
llama-3.1-8b-instant (barato) ou llama-3.3-70b-versatile (poderoso).

Critérios de routing:
- Pergunta curta + palavras-chave simples (o que é, como usar, definição) -> modelo barato
- Pergunta longa, comparativa, multi-step, "por que", "diferença entre" -> modelo caro
"""

from __future__ import annotations

import os
import re

LLM_MODEL_MAIN = os.getenv("LLM_MODEL_MAIN", "llama-3.3-70b-versatile")
LLM_MODEL_CHEAP = os.getenv("LLM_MODEL_CHEAP", "llama-3.1-8b-instant")
ROUTING_THRESHOLD = float(os.getenv("ROUTING_THRESHOLD", "0.65"))

# Estatísticas de routing
_routing_stats = {"cheap": 0, "main": 0}

# Padrões que indicam pergunta simples (modelo barato suficiente)
_SIMPLE_PATTERNS = [
    r"\bo que (é|e)\b",
    r"\bcomo (usar|instalar|configurar|criar|fazer)\b",
    r"\bdefin[ie]\b",
    r"\bsignifica\b",
    r"\bexemplo de\b",
    r"\bpara que serve\b",
    r"\bquais (são|sao) os comandos\b",
]

# Padrões que indicam pergunta complexa (modelo poderoso necessário)
_COMPLEX_PATTERNS = [
    r"\bpor que\b",
    r"\bdiferença entre\b",
    r"\bcompar[ae]\b",
    r"\bquando (devo|usar|escolher)\b",
    r"\bmelhor (forma|maneira|abordagem|estratégia)\b",
    r"\bvantagens? e desvantagens?\b",
    r"\bcomo (funciona internamente|é implementado)\b",
    r"\bexplique\b",
    r"\bjustifique\b",
    r"\banalise\b",
]


def _complexity_score(question: str) -> float:
    # Retorna score de complexidade entre 0.0 (simples) e 1.0 (complexa)
    # Score < ROUTING_THRESHOLD → modelo barato

    q = question.lower()
    score = 0.5  # neutro

    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, q):
            score -= 0.15

    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, q):
            score += 0.20

    word_count = len(q.split())
    if word_count < 8:
        score -= 0.10
    elif word_count > 20:
        score += 0.10

    if question.count("?") > 1 or re.search(r"\be (também|além disso|adicionalmente)\b", q):
        score += 0.15

    return max(0.0, min(1.0, score))


def route(question: str) -> str:

    # Retorna o nome do modelo a usar para esta pergunta
    # Registra estatísticas de routing

    score = _complexity_score(question)
    if score < ROUTING_THRESHOLD:
        _routing_stats["cheap"] += 1
        return LLM_MODEL_CHEAP
    _routing_stats["main"] += 1
    return LLM_MODEL_MAIN


def routing_stats() -> dict:
    # Retorna estatísticas de uso dos modelos
    total = _routing_stats["cheap"] + _routing_stats["main"]
    cheap_rate = _routing_stats["cheap"] / total if total > 0 else 0.0
    return {
        "cheap_calls": _routing_stats["cheap"],
        "main_calls": _routing_stats["main"],
        "total": total,
        "cheap_rate": round(cheap_rate, 3),
        "cheap_model": LLM_MODEL_CHEAP,
        "main_model": LLM_MODEL_MAIN,
        "routing_threshold": ROUTING_THRESHOLD,
    }