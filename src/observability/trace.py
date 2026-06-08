"""Logging estruturado para rastreamento de chamadas do pipeline."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("progit-qa")


def log_request(
    question: str,
    model: str,
    cache_hit: bool,
    usage: dict[str, int] | None = None,
    latency_ms: float | None = None,
    tool_calls: list | None = None,
) -> None:
    # Grava um log em JSON a cada chamada do pipeline, com informações sobre a pergunta, modelo, cache e latência
    payload: dict[str, Any] = {
        "question": question[:120],
        "model": model,
        "cache_hit": cache_hit,
    }
    if usage:
        payload["tokens"] = usage
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 1)
    if tool_calls:
        payload["tool_calls"] = [t["tool"] for t in tool_calls]
    logger.info(json.dumps(payload, ensure_ascii=False))


class Timer:
    # Context manager para medir latência de qualquer bloco de código
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
