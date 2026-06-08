"""
Pro Git Q&A — Interface Streamlit.
TODO 6 implementado: chat com streaming, cache semântico, model routing,
sidebar com métricas de custo e estatísticas de cache/routing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
from openai import RateLimitError

# Garante que src/ está no path quando rodado via `streamlit run src/ui/streamlit_app.py`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.observability.trace import Timer, log_request
from src.pipeline.cache import cache_get, cache_set, cache_stats
from src.pipeline.rag import ingest_corpus, rag_answer
from src.pipeline.routing import route, routing_stats
from src.pipeline.tools import answer_with_tools

# Configuração da página
st.set_page_config(
    page_title="Pro Git Q&A",
    page_icon="📖",
    layout="wide",
)

# Ingestão (roda 1x por sessão)
@st.cache_resource(show_spinner="Indexando corpus Pro Git…")
def load_corpus():
    return ingest_corpus()

try:
    chunk_count = load_corpus()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Estado da sessão 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_calls" not in st.session_state:
    st.session_state.total_calls = 0

# Sidebar 
with st.sidebar:
    st.title("⚙️ Configurações")

    mode = st.radio(
        "Modo de resposta",
        ["RAG simples", "RAG + Tool-use"],
        help="Tool-use permite ao LLM chamar lookup_chapter para navegar por capítulo.",
    )

    use_cache = st.toggle("Cache semântico", value=True)
    use_routing = st.toggle("Model routing cheap-first", value=True)
    k = st.slider("Chunks recuperados (k)", min_value=3, max_value=10, value=5)

    st.divider()
    st.subheader("📊 Métricas da sessão")

    cs = cache_stats()
    rs = routing_stats()

    col1, col2 = st.columns(2)
    col1.metric("Cache hit-rate", f"{cs['hit_rate']*100:.0f}%")
    col2.metric("Cache size", cs["cache_size"])

    col3, col4 = st.columns(2)
    col3.metric("Chamadas baratas", rs["cheap_calls"])
    col4.metric("Chamadas completas", rs["main_calls"])

    if rs["total"] > 0:
        cheap_pct = rs["cheap_rate"] * 100
        st.progress(rs["cheap_rate"], text=f"{cheap_pct:.0f}% roteado para modelo barato")

    st.divider()
    st.metric("Tokens totais (sessão)", st.session_state.total_tokens)
    st.metric("Chamadas LLM", st.session_state.total_calls)
    st.caption(f"Corpus: {chunk_count} chunks indexados")

    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

# Área principal 
st.title("📖 Pro Git Q&A")
st.caption(
    "Faça perguntas sobre o livro **Pro Git** (Scott Chacon). "
    "As respostas são geradas com RAG — apenas com base no conteúdo do livro."
)

# Exemplos de perguntas
with st.expander("💡 Exemplos de perguntas", expanded=False):
    examples = [
        "O que é o Git staging area e para que serve?",
        "Como funciona o git rebase comparado ao git merge?",
        "Quais são os principais comandos para trabalhar com branches?",
        "O que é um bare repository?",
        "Como resolver conflitos de merge no Git?",
        "Explique o que é cherry-pick e quando usar.",
        "Qual a diferença entre git fetch e git pull?",
        "Como configurar aliases no Git?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.session_state["pending_prompt"] = ex
            st.rerun()

# Histórico da conversa
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Fontes consultadas"):
                for s in msg["sources"]:
                    st.caption(
                        f"Capítulo {s.get('chapter', '?')} · p. {s.get('page', '?')} "
                        f"· dist={s.get('distance', 0):.3f}"
                    )
                    st.text(s["text"][:200] + "…")
        if msg.get("tool_calls"):
            with st.expander("🔧 Tools chamadas"):
                for tc in msg["tool_calls"]:
                    st.code(f"{tc['tool']}({tc['args']})")
        if msg.get("cache_hit"):
            st.caption("⚡ Resposta do cache semântico")
        if msg.get("model"):
            st.caption(f"🤖 Modelo: `{msg['model']}`")

# Input do usuário
if prompt := st.chat_input("Pergunte sobre o Pro Git…"):
    pass

# Recupera prompt vindo de botão de exemplo
if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Checar cache
        if use_cache:
            cached = cache_get(prompt)
            if cached:
                st.markdown(cached["answer"])
                st.caption("⚡ Resposta do cache semântico")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cached["answer"],
                    "cache_hit": True,
                    "model": "cache",
                })
                log_request(prompt, "cache", cache_hit=True)
                st.rerun()

        # 2. Routing
        model = route(prompt) if use_routing else None

        # 3. Gerar resposta
        with st.spinner("Consultando o Pro Git…"):
            with Timer() as t:
                for attempt in range(4):
                    try:
                        if mode == "RAG + Tool-use":
                            result = answer_with_tools(prompt, model=model or "llama-3.3-70b-versatile")
                        else:
                            result = rag_answer(prompt, k=k, model=model)
                        break
                    except RateLimitError:
                        wait = 2 ** attempt
                        st.toast(f"Rate limit — aguardando {wait}s…", icon="⏳")
                        time.sleep(wait)
                else:
                    st.error("Rate limit excedido após 4 tentativas. Aguarde 1 minuto e tente novamente.")
                    st.stop()
                answer = result["answer"]
                sources = result.get("sources", [])
                tool_calls = result.get("tool_calls", [])

        # 4. Exibir resposta
        st.markdown(answer)

        if sources:
            with st.expander("📚 Fontes consultadas"):
                for s in sources:
                    st.caption(
                        f"Capítulo {s.get('chapter', '?')} · p. {s.get('page', '?')} "
                        f"· dist={s.get('distance', 0):.3f}"
                    )
                    st.text(s["text"][:200] + "…")

        if tool_calls:
            with st.expander("🔧 Tools chamadas"):
                for tc in tool_calls:
                    st.code(f"{tc['tool']}({tc['args']})")

        used_model = result.get("model", model or "?")
        st.caption(f"🤖 `{used_model}` · {t.elapsed_ms:.0f}ms")

        # 5. Salvar no cache e atualizar stats
        if use_cache:
            cache_set(prompt, answer, model=used_model)

        usage = result.get("usage", {})
        st.session_state.total_tokens += usage.get("total_tokens", 0)
        st.session_state.total_calls += 1

        log_request(
            prompt,
            used_model,
            cache_hit=False,
            usage=usage,
            latency_ms=t.elapsed_ms,
            tool_calls=tool_calls,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "tool_calls": tool_calls,
            "model": used_model,
            "cache_hit": False,
        })
