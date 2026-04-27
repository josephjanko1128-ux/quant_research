# Multimodal PDF RAG Tool

Answer questions over a folder of PDFs using **text + images**, powered by:
- **Voyage multimodal embeddings** (`voyage-multimodal-3`) for retrieval
- **Claude** (`claude-opus-4-6`) for generation

---

## Setup

```bash
pip install anthropic pymupdf numpy scikit-learn tqdm
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Single question
python multimodal_rag.py --pdf_dir ./pdfs --query "Summarize the key findings"

# Interactive mode
python multimodal_rag.py --pdf_dir ./pdfs

# Reuse a saved index (skip re-embedding)
python multimodal_rag.py --pdf_dir ./pdfs --index_path ./my_index.json

python scrape_sec_pdfs.py --delay 2.0 --output ./pdfs
```

## How it works

```
PDFs in directory
      │
      ▼
┌─────────────────────────┐
│  PyMuPDF (fitz)         │  Extracts:
│  • Text → overlapping   │    • Text chunks (800 chars, 150 overlap)
│    chunks               │    • Page screenshots as PNG (catches
│  • Page screenshots     │      diagrams, tables, charts)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Voyage Multimodal      │  Embeds both text and images into a
│  Embeddings             │  shared vector space
└────────────┬────────────┘
             │  (index cached to .rag_index.json)
             ▼
┌─────────────────────────┐
│  Cosine Similarity      │  Retrieves top-K most relevant chunks
│  Retrieval              │  (default K=5)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Claude (multimodal     │  Receives text + image chunks as context,
│  messages API)          │  generates a cited answer
└─────────────────────────┘
```

## Configuration (top of script)

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per text chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `IMAGE_DPI` | 150 | DPI for page screenshots |
| `EMBED_MODEL` | `voyage-multimodal-3` | Embedding model |
| `CHAT_MODEL` | `claude-opus-4-6` | Generation model |

## Notes

- The index is cached to `.rag_index.json` in the PDF directory on first run.  
  Delete it to force re-indexing (e.g., after adding new PDFs).
- Page screenshots capture visual content (charts, diagrams, scanned text)  
  that pure text extraction would miss.
- Source file and page number are cited in every answer.
