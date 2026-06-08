"""
Pipeline RAG para Pro Git Q&A.
TODO 1 implementado: chunking recursivo 800/100, embedding all-MiniLM-L6-v2,
retrieval via ChromaDB com metadados de capítulo e página.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Constantes 
CORPUS_DIR = Path("data/corpus")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "progit"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL_MAIN = os.getenv("LLM_MODEL_MAIN", "llama-3.3-70b-versatile")
LLM_MODEL_CHEAP = os.getenv("LLM_MODEL_CHEAP", "llama-3.1-8b-instant")

RAG_PROMPT = """\
Você é um assistente especializado no livro "Pro Git" (Scott Chacon).
Responda APENAS com base no contexto abaixo.
Se a informação for parcial, responda com o que houver e marque [INCERTO].
Se o tema não aparecer no contexto, diga exatamente: "Não encontrado no corpus."
Sempre cite a fonte no formato [capítulo N, p. X].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

# Embedding function local
_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


class LocalEmbedFn(EmbeddingFunction):
    def __init__(self):
        self._model = _get_st_model()

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), show_progress_bar=False).tolist()


# Singletons
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_chroma_collection() -> chromadb.Collection:
    # Retorna (ou cria) a collection ChromaDB com embedding local
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=LocalEmbedFn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_api_key() -> str:
    """
    Lê a GROQ_API_KEY com prioridade:
      1. st.secrets (secrets.toml ou Streamlit Cloud)
      2. variável de ambiente do sistema
    """
    # 1. Streamlit secrets
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # 2. Variável de ambiente
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key

    raise ValueError(
        "GROQ_API_KEY não encontrada.\n"
        "Opção 1 - crie .streamlit/secrets.toml com: GROQ_API_KEY = \"gsk_...\"\n"
        "Opção 2 - defina a variável de ambiente antes de iniciar o Streamlit."
    )


def get_llm_client() -> OpenAI:
    # Cliente OpenAI apontando para Groq
    return OpenAI(
        api_key=_get_api_key(),
        base_url="https://api.groq.com/openai/v1",
    )


# TODO 1: Ingestão
def ingest_corpus(force: bool = False) -> int:
    """
    Lê todos os PDFs em data/corpus/, divide em chunks 800/100,
    extrai metadados de capítulo e página, e indexa no Chroma.
    Retorna o número total de chunks indexados.
    """
    col = get_chroma_collection()
    if col.count() > 0 and not force:
        print(f"Corpus já indexado: {col.count()} chunks. Use force=True para reingerir.")
        return col.count()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for pdf_path in sorted(CORPUS_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        current_chapter = 1
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            # Detectar troca de capítulo pelo padrão "Chapter N" no topo da página
            first_line = text.strip().split("\n")[0]
            if first_line.lower().startswith("chapter"):
                try:
                    current_chapter = int(first_line.split()[1])
                except (IndexError, ValueError):
                    pass

            for i, chunk_text in enumerate(splitter.split_text(text)):
                chunks.append({
                    "id": f"{pdf_path.stem}-p{page_idx + 1}-c{i}",
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "page": page_idx + 1,
                    "chapter": current_chapter,
                    "chunk_idx": i,
                })

    if not chunks:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {CORPUS_DIR}. "
            "Baixe o Pro Git em https://git-scm.com/book/en/v2 e coloque em data/corpus/."
        )

    # Inserir em batches de 100
    BATCH = 100
    for start in range(0, len(chunks), BATCH):
        lote = chunks[start:start + BATCH]
        col.add(
            ids=[c["id"] for c in lote],
            documents=[c["text"] for c in lote],
            metadatas=[{
                "source": c["source"],
                "page": c["page"],
                "chapter": c["chapter"],
                "chunk_idx": c["chunk_idx"],
            } for c in lote],
        )

    print(f"Ingestão concluída: {col.count()} chunks de {len(list(CORPUS_DIR.glob('*.pdf')))} PDF(s).")
    return col.count()


# TODO 1: Retrieval 
def retrieve(query: str, k: int = 5) -> list[dict[str, Any]]:
    # Recupera os k chunks mais similares à query.
    # Retorna lista de dicts com text, source, page, chapter, distance.
  
    col = get_chroma_collection()
    result = col.query(query_texts=[query], n_results=k)
    hits = []
    for i in range(len(result["documents"][0])):
        hits.append({
            "text": result["documents"][0][i],
            "source": result["metadatas"][0][i]["source"],
            "page": result["metadatas"][0][i]["page"],
            "chapter": result["metadatas"][0][i].get("chapter", "?"),
            "distance": result["distances"][0][i],
        })
    return hits


# Geração 
def rag_answer(
    question: str,
    k: int = 5,
    model: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Pipeline RAG completo: retrieve -> augment -> generate.
    Se model=None, usa LLM_MODEL_MAIN.
    Se stream=True, retorna o objeto stream em vez do texto completo.
    """
    if model is None:
        model = LLM_MODEL_MAIN

    hits = retrieve(question, k=k)
    context = "\n\n---\n\n".join(
        f"[capítulo {h['chapter']}, p. {h['page']}]\n{h['text']}" for h in hits
    )
    prompt = RAG_PROMPT.format(context=context, question=question)

    client = get_llm_client()

    if stream:
        return {
            "stream": client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                stream=True,
            ),
            "sources": hits,
        }

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": hits,
        "model": model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
