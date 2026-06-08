"""
Tools (function-calling) do Pro Git Q&A.
TODO 2: tool genérica de busca no corpus.
TODO 4: tool de domínio — lookup_chapter, retorna sumário e chunks do capítulo N.
"""

from __future__ import annotations

import json
from typing import Any

from src.pipeline.rag import get_chroma_collection, get_llm_client, LLM_MODEL_MAIN, retrieve

# Definições das tools para a API 
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Busca trechos relevantes do livro Pro Git para responder à pergunta. "
                "Use sempre que precisar de informação sobre Git."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto de busca em linguagem natural.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Número de trechos a retornar (padrão: 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_chapter",
            "description": (
                "Retorna o conteúdo completo de um capítulo específico do Pro Git. "
                "Use quando o usuário perguntar sobre um capítulo pelo número ou título."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter": {
                        "type": "integer",
                        "description": "Número do capítulo (1–10).",
                    },
                },
                "required": ["chapter"],
            },
        },
    },
]

# Implementações

def search_corpus(query: str, k: int = 5) -> str:
    # TODO 2 - Busca semântica no corpus e retorna trechos formatados
    hits = retrieve(query, k=k)
    if not hits:
        return "Nenhum trecho encontrado."
    parts = []
    for h in hits:
        parts.append(
            f"[capítulo {h['chapter']}, p. {h['page']}] (dist={h['distance']:.3f})\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)


def lookup_chapter(chapter: int) -> str:
    """
    TODO 4 - Tool de domínio: retorna todos os chunks do capítulo N,
    concatenados em ordem de página. Útil para navegação dirigida.
    """
    col = get_chroma_collection()

    # Busca por metadado de capítulo exato
    result = col.get(
        where={"chapter": {"$eq": chapter}},
        include=["documents", "metadatas"],
    )

    if not result["documents"]:
        return f"Capítulo {chapter} não encontrado no corpus. Verifique se o PDF foi indexado."

    # Ordenar por página e chunk_idx
    pairs = sorted(
        zip(result["documents"], result["metadatas"]),
        key=lambda x: (x[1].get("page", 0), x[1].get("chunk_idx", 0)),
    )

    # Limitar a 10 trechos para não estourar TPM do Groq free tier
    pairs = pairs[:10]

    parts = []
    seen_pages: set[int] = set()
    for doc, meta in pairs:
        page = meta.get("page", "?")
        if page not in seen_pages:
            parts.append(f"── Página {page} ──")
            seen_pages.add(page)
        parts.append(doc)

    header = f"=== Capítulo {chapter} — {len(pairs)} trechos ===\n"
    return header + "\n\n".join(parts)


# Dispatcher 
TOOL_MAP = {
    "search_corpus": search_corpus,
    "lookup_chapter": lookup_chapter,
}


def dispatch_tool(name: str, arguments: dict) -> str:
    # Chama a tool pelo nome e retorna o resultado como string
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Tool '{name}' não existe."
    try:
        return fn(**arguments)
    except Exception as e:
        return f"Erro ao executar '{name}': {e}"


# Pipeline com tool-use 
def answer_with_tools(question: str, model: str = LLM_MODEL_MAIN) -> dict[str, Any]:
    """
    Pipeline completo com function-calling:
    1. LLM decide se usa search_corpus ou lookup_chapter
    2. Tool é executada localmente
    3. Resultado é enviado de volta ao LLM para resposta final
    """
    client = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado no livro Pro Git. "
                "Use as tools disponíveis para buscar informação antes de responder. "
                "Sempre cite o capítulo e a página na resposta final."
            ),
        },
        {"role": "user", "content": question},
    ]

    tool_calls_log = []

    # Turno 1 - LLM decide qual tool chamar
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.0,
    )
    msg = response.choices[0].message

    # Executar todas as tool calls retornadas
    while msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = dispatch_tool(tc.function.name, args)
            tool_calls_log.append({"tool": tc.function.name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Turno seguinte - LLM gera resposta com base nos resultados
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = response.choices[0].message

    return {
        "answer": msg.content,
        "tool_calls": tool_calls_log,
        "model": model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
