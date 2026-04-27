# Multi-Agent RAG System

A multi-agent system built with **LangChain** and **Anthropic Claude** that combines:
- 📄 **PDF RAG Agent** — retrieves answers from two local PDF directories
- 🌐 **Web Agent** — searches the web via DuckDuckGo
- 🤖 **Orchestrator Agent** — routes queries to the right tool(s)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Add PDFs
Place your PDF files in the two collection directories:
```
pdfs/
  collection1/   ← first set of PDFs
  collection2/   ← second set of PDFs
```
You can change these paths via `PDF_DIR_1` and `PDF_DIR_2` in `.env`.

### 4. Run
```bash
python main.py
```

---

## Architecture

```
User Query
    │
    ▼
Orchestrator Agent (Claude claude-sonnet-4-6)
    │
    ├─── pdf_rag_search ──► FAISS Vector Store
    │                           ├── Collection 1 PDFs
    │                           └── Collection 2 PDFs
    │
    └─── web_search ──────► DuckDuckGo
```

### How it works

1. **PDF Ingestion**: All PDFs from both directories are loaded, split into chunks, embedded with `all-MiniLM-L6-v2`, and stored in a FAISS vector store in memory.
2. **RAG Tool**: Performs similarity search on the vector store and returns the top 4 matching chunks with source metadata.
3. **Web Tool**: Wraps DuckDuckGo search for real-time web queries.
4. **Orchestrator**: A Claude-powered tool-calling agent that decides which tool(s) to use based on the query, then synthesizes a final answer.

---

## Customization

| Setting | Where | Default |
|---|---|---|
| PDF directory 1 | `.env` → `PDF_DIR_1` | `./pdfs/collection1` |
| PDF directory 2 | `.env` → `PDF_DIR_2` | `./pdfs/collection2` |
| LLM model | `main.py` → `llm` | `claude-sonnet-4-6` |
| Chunk size | `main.py` → `splitter` | `1000` |
| Top-K results | `main.py` → `rag_search` | `4` |
