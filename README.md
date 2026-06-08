# Pro Git Q&A

> Sistema de perguntas e respostas sobre o livro **Pro Git** (Scott Chacon), construído com RAG ponta-a-ponta, cache semântico, model routing e tool-use.

**Autor:** Alexsandro Barreto de Abreu  
**Disciplina:** Mod4 / PPI — Desenvolvendo Software com IA Generativa  
**Demo:** https://pro-git-q-a-ekkuhqhfmgak4xymyrqj5n.streamlit.app  
**Vídeo:** https://youtu.be/Egx97bK17BY

---

## Problema

Livros técnicos extensos como o Pro Git (~440 páginas) são difíceis de consultar rapidamente. Encontrar a resposta certa exige saber em qual capítulo procurar, navegar pelo índice e ler trechos longos.

Este projeto resolve isso com um assistente conversacional que responde perguntas diretamente sobre o conteúdo do livro, citando capítulo e página, sem que o usuário precise abrir o PDF.

**Não respondível sem o corpus (exemplos reais):**
- "Como funciona o git rebase comparado ao git merge?" → exige contexto dos capítulos 3 e 7
- "O que é um bare repository e quando usar?" → conceito específico do capítulo 4
- "Como configurar aliases no Git?" → detalhe do capítulo 2, p. 52

---

## Arquitetura

```
Pergunta do usuário
        │
        ▼
┌───────────────┐     hit     ┌─────────────────┐
│ Cache semântico│────────────▶│ Resposta direta │
│  (ChromaDB)   │             └─────────────────┘
└──────┬────────┘
       │ miss
       ▼
┌───────────────┐
│Model Routing  │  score < 0.65 → llama-3.1-8b-instant  (barato)
│  (scoring)    │  score ≥ 0.65 → llama-3.3-70b-versatile (completo)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│   Retrieval   │  ChromaDB · cosine similarity · k=5 chunks
│  (RAG + k=5)  │  embedding: all-MiniLM-L6-v2 (local, gratuito)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Geração LLM  │  Groq API · prompt com contexto + citação obrigatória
│  (Groq API)   │
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Tool-use     │  lookup_chapter(n) → navegação dirigida por capítulo
└──────┬────────┘
       │
       ▼
  Resposta com [capítulo N, p. X]
```

**Stack:**

| Camada | Tecnologia | Motivo da escolha |
|---|---|---|
| LLM | Groq (llama-3.3-70b + llama-3.1-8b) | Inferência rápida, free tier generoso |
| Embedding | all-MiniLM-L6-v2 (local) | Gratuito, sem chamada de API, boa qualidade |
| Vector store | ChromaDB persistente | Simples, local, sem servidor separado |
| Chunking | RecursiveCharacterTextSplitter 800/100 | Equilibra contexto e precisão de retrieval |
| Interface | Streamlit | Deploy 1-click, sem frontend separado |
| Cache | ChromaDB semântico | Reutiliza respostas para perguntas similares |

---

## Métricas

| Métrica | Valor observado |
|---|---|
| Chunks indexados | 1.447 (Pro Git completo) |
| Latência média (modelo barato) | ~800 ms |
| Latência média (modelo completo) | ~1.800 ms |
| Cache hit-rate (sessão típica) | ~67% após 3+ perguntas |
| Tokens por requisição (sem cache) | ~1.215 tokens |
| Custo estimado por requisição (sem cache) | ~0,001 USD (llama-3.1-8b) / ~0,004 USD (llama-3.3-70b) |
| Custo estimado por requisição (com cache) | ~0 USD |
| Redução de custo com routing | perguntas complexas → llama-3.3-70b; simples → llama-3.1-8b |


---

## Setup

### Pré-requisitos

- Python 3.11+
- Conta gratuita no [Groq](https://console.groq.com)
- PDF do Pro Git (instruções abaixo)

### Instalação

```powershell
# Clone o repositório
git clone <URL_DO_REPO>
cd progit-qa-final

# Instale as dependências
pip install -e .
```

### Configuração da API Key

Abra `.streamlit/secrets.toml` e insira sua chave:

```toml
GROQ_API_KEY = "gsk_SuaChaveAqui"
```

> Este arquivo está no `.gitignore` - nunca será commitado.

### Adicionar o corpus

1. Baixe o PDF em: https://git-scm.com/book/en/v2
2. Coloque o `.pdf` dentro de `data/corpus/`

### Executar

```powershell
python -m streamlit run src\ui\streamlit_app.py
```

Acesse: http://localhost:8501 - na primeira execução o corpus é indexado automaticamente.

---

## Estrutura do projeto

```
progit-qa-final/
├── pyproject.toml              ← dependências
├── .streamlit/
│   └── secrets.toml            ← API key (não deixar vazar)
├── data/
│   ├── corpus/                 ← PDF do Pro Git
│   └── chroma/                 ← índice vetorial (gerado automaticamente)
├── src/
│   ├── pipeline/
│   │   ├── rag.py              ← TODO 1: chunking + embedding + retrieval
│   │   ├── tools.py            ← TODO 2/4: tool-use (lookup_chapter)
│   │   ├── cache.py            ← TODO 3: cache semântico
│   │   └── routing.py          ← TODO 5: model routing cheap-first
│   ├── ui/
│   │   └── streamlit_app.py    ← TODO 6: chat UI + streaming
│   └── observability/
│       └── trace.py            ← logging estruturado
└── tests/
    └── test_smoke.py           ← teste das funcionalidades
```

---

## Decisões de design

**Por que ChromaDB para o cache semântico?**  
O cache já usa ChromaDB para o corpus, reutilizar o mesmo cliente evita dependência extra. O threshold de similaridade (0.92) foi calibrado para evitar falsos positivos: perguntas parecidas mas com intenção diferente não devem compartilhar resposta.

**Por que embedding local (all-MiniLM-L6-v2)?**  
Elimina latência e custo de API de embedding. O modelo tem 22M parâmetros, roda em CPU em ~50ms por batch e tem qualidade suficiente para retrieval em texto técnico em inglês.

**Por que chunking 800/100?**  
Chunks de 800 tokens cabem no contexto sem truncar parágrafos completos. O overlap de 100 garante que conceitos que cruzam fronteiras de chunk sejam recuperados. Testado com separadores hierárquicos (`\n\n → \n → . → espaço`).

**Por que Groq em vez de OpenAI?**  
Free tier da Groq oferece ~14.400 requisições/dia com latência de inferência muito baixa (~300ms TTFT), viável para demo sem custo.

---

## Limites do projeto

- **Idioma:** o corpus é em inglês; perguntas em português funcionam mas a qualidade cai em termos muito específicos
- **Alucinação:** o LLM pode inventar citações de página se o chunk recuperado for ambíguo — o prompt mitiga mas não elimina
- **Corpus fixo:** só responde sobre o Pro Git; não generaliza para outros livros sem reingestão
- **Rate limit:** o free tier da Groq tem limite de ~30 RPM — sessões com muitas perguntas rápidas podem sofrer throttling
- **Sem memória entre sessões:** o histórico da conversa é perdido ao recarregar a página

---

## Possíveis melhorias futuras

- Avaliação automática com RAGAS (faithfulness + answer relevancy)
- Suporte a múltiplos corpus (seleção na sidebar)
- Persistência do histórico entre sessões
