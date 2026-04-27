"""
Multimodal RAG Tool - Reads PDFs from a directory and answers questions using
text + image content via the Anthropic API (Claude) + free local CLIP embeddings.

Dependencies:
    pip install anthropic pymupdf numpy scikit-learn tqdm pillow transformers torch

Usage:
    python multimodal_rag.py --pdf_dir ./pdfs --query "What is the main topic?"
    python multimodal_rag.py --pdf_dir ./pdfs  # launches interactive CLI

Environment variables:
    ANTHROPIC_API_KEY  — for Claude (generation only)

Embedding model (free, runs locally):
    openai/clip-vit-base-patch32 via HuggingFace Transformers — no API key needed.
    Downloaded automatically on first run (~600 MB, cached in ~/.cache/huggingface).
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────── Configuration ────────────────────────────────────

CHUNK_SIZE     = 800     # characters per text chunk
CHUNK_OVERLAP  = 150     # overlap between consecutive chunks
TOP_K          = 5       # number of chunks to retrieve per query
IMAGE_DPI      = 150     # DPI for rendering page screenshots
MIN_IMAGE_AREA = 5000    # min pixel area to keep a page screenshot
CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"  # free HuggingFace CLIP model
CHAT_MODEL     = "claude-opus-4-6"               # Claude model for generation

# ─────────────────────────── PDF Processing ───────────────────────────────────

def extract_chunks_and_images(pdf_path: str) -> list[dict]:
    """
    Extract overlapping text chunks and page screenshots from a single PDF.

    Each chunk dict has:
        type     : "text" | "image"
        content  : str (text) or base64-encoded PNG string (image)
        source   : filename
        page     : 1-based page number
        chunk_id : int
    """
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).name
    chunks = []
    chunk_id = 0

    for page_num, page in enumerate(doc, start=1):
        # ── Text chunks ──────────────────────────────────────────────────────
        text = page.get_text("text").strip()
        if text:
            for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                snippet = text[start : start + CHUNK_SIZE].strip()
                if snippet:
                    chunks.append({
                        "type": "text",
                        "content": snippet,
                        "source": filename,
                        "page": page_num,
                        "chunk_id": chunk_id,
                    })
                    chunk_id += 1

        # ── Page screenshot (captures diagrams, charts, tables, scanned text) ─
        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        if pix.width * pix.height >= MIN_IMAGE_AREA:
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            chunks.append({
                "type": "image",
                "content": b64,
                "source": filename,
                "page": page_num,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    doc.close()
    return chunks


def load_pdf_directory(pdf_dir: str) -> list[dict]:
    """Load every PDF in a directory and return a flat list of chunks."""
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {pdf_dir}")
        return []

    all_chunks = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunks = extract_chunks_and_images(str(pdf_path))
            all_chunks.extend(chunks)
            print(f"  + {pdf_path.name} -> {len(chunks)} chunks")
        except Exception as exc:
            print(f"  ! {pdf_path.name}: {exc}")

    return all_chunks

# ─────────────────────────── CLIP Embeddings (free, local) ────────────────────

def load_clip_model():
    """
    Load CLIP from HuggingFace (downloaded once, then cached).
    Returns (model, processor, device).
    """
    print(f"Loading CLIP model '{CLIP_MODEL_ID}' (first run downloads ~600 MB) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print(f"  + CLIP ready on {device}")
    return model, processor, device


def _l2(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalise along the last dimension."""
    return tensor / tensor.norm(dim=-1, keepdim=True)


def embed_chunks(chunks: list[dict], model, processor, device: str) -> np.ndarray:
    """
    Embed every chunk with CLIP.

    CLIP projects both text and images into the same 512-dim space, so cosine
    similarity works across modalities:
      - text query  <->  text chunk
      - text query  <->  image chunk  (the key multimodal benefit)

    Text is truncated to CLIP's 77-token limit; images are resized by the processor.
    """
    BATCH_SIZE = 32
    embedding_map: dict[int, np.ndarray] = {}

    text_chunks  = [(i, c) for i, c in enumerate(chunks) if c["type"] == "text"]
    image_chunks = [(i, c) for i, c in enumerate(chunks) if c["type"] == "image"]

    # ── Text ──────────────────────────────────────────────────────────────────
    for b in tqdm(range(0, len(text_chunks), BATCH_SIZE), desc="Embedding text"):
        batch = text_chunks[b : b + BATCH_SIZE]
        texts = [c["content"] for _, c in batch]
        inputs = processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        ).to(device)
        with torch.no_grad():
            feats = _l2(model.get_text_features(**inputs))
        for (idx, _), vec in zip(batch, feats.cpu().numpy()):
            embedding_map[idx] = vec

    # ── Images ────────────────────────────────────────────────────────────────
    for b in tqdm(range(0, len(image_chunks), BATCH_SIZE), desc="Embedding images"):
        batch = image_chunks[b : b + BATCH_SIZE]
        pil_imgs = []
        for _, c in batch:
            img_bytes = base64.b64decode(c["content"])
            pil_imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _l2(model.get_image_features(**inputs))
        for (idx, _), vec in zip(batch, feats.cpu().numpy()):
            embedding_map[idx] = vec

    # Return in original chunk order
    return np.array([embedding_map[i] for i in range(len(chunks))], dtype="float32")


def embed_query(query: str, model, processor, device: str) -> np.ndarray:
    """Embed a single text query with CLIP."""
    inputs = processor(
        text=[query], return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        feats = _l2(model.get_text_features(**inputs))
    return feats.cpu().numpy()

# ─────────────────────────── Index persistence ────────────────────────────────

def build_index(chunks: list[dict], model, processor, device: str) -> dict:
    print("\n[1/2] Embedding all chunks with CLIP ...")
    embeddings = embed_chunks(chunks, model, processor, device)
    print(f"[2/2] Index ready -- {len(chunks)} chunks, dim={embeddings.shape[1]}")
    return {"chunks": chunks, "embeddings": embeddings}


def save_index(index: dict, path: str):
    """Save the index to a JSON file so re-embedding is skipped on future runs."""
    data = {
        "chunks": index["chunks"],
        "embeddings": index["embeddings"].tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Index saved -> {path}")


def load_index(path: str) -> dict:
    """Load a previously saved index."""
    with open(path) as f:
        data = json.load(f)
    return {
        "chunks": data["chunks"],
        "embeddings": np.array(data["embeddings"], dtype="float32"),
    }

# ─────────────────────────── Retrieval ────────────────────────────────────────

def retrieve(query: str, index: dict, model, processor, device: str,
             top_k: int = TOP_K) -> list[dict]:
    """Return the top-k most relevant chunks for a text query."""
    q_vec = embed_query(query, model, processor, device)
    sims  = cosine_similarity(q_vec, index["embeddings"])[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [index["chunks"][i] for i in top_indices]

# ─────────────────────────── Generation ──────────────────────────────────────

def build_context_message(retrieved: list[dict]) -> list[dict]:
    """Assemble a multimodal content list from retrieved chunks."""
    content: list[dict] = [{"type": "text", "text": "Relevant context from the documents:\n"}]

    for chunk in retrieved:
        tag = f"[{chunk['source']} -- page {chunk['page']}]"
        if chunk["type"] == "text":
            content.append({"type": "text", "text": f"\n{tag}\n{chunk['content']}\n"})
        else:
            content.append({"type": "text", "text": f"\n{tag} (page screenshot):\n"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": chunk["content"],
                },
            })

    return content


def answer(query: str, index: dict, claude_client: anthropic.Anthropic,
           model, processor, device: str) -> str:
    """Retrieve relevant chunks and generate a cited answer with Claude."""
    retrieved = retrieve(query, index, model, processor, device)

    context = build_context_message(retrieved)
    context.append({
        "type": "text",
        "text": f"\nUsing only the context above, answer this question:\n\n{query}",
    })

    response = claude_client.messages.create(
        model=CHAT_MODEL,
        max_tokens=2048,
        system=(
            "You are a helpful assistant that answers questions based solely on the "
            "provided document context (text and images). Always cite the source file "
            "and page number when referencing specific information. If the answer is "
            "not in the context, say so clearly."
        ),
        messages=[{"role": "user", "content": context}],
    )
    return response.content[0].text

# ─────────────────────────── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multimodal PDF RAG Tool")
    parser.add_argument("--pdf_dir",    required=True,            help="Directory containing PDF files")
    parser.add_argument("--query",      default=None,             help="Single query (omit for interactive mode)")
    parser.add_argument("--index_path", default=None,             help="Path to save/load index JSON")
    parser.add_argument("--top_k",      type=int, default=TOP_K,  help="Chunks to retrieve per query")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set the ANTHROPIC_API_KEY environment variable.")

    claude_client = anthropic.Anthropic(api_key=api_key)

    # Load CLIP once; all subsequent calls reuse the same in-memory model
    clip_model, clip_processor, device = load_clip_model()

    # ── Build or reuse index ─────────────────────────────────────────────────
    index_file = args.index_path or str(Path(args.pdf_dir) / ".rag_index.json")

    if Path(index_file).exists():
        print(f"Loading existing index from {index_file} ...")
        index = load_index(index_file)
    else:
        chunks = load_pdf_directory(args.pdf_dir)
        if not chunks:
            sys.exit("No chunks extracted. Exiting.")
        index = build_index(chunks, clip_model, clip_processor, device)
        save_index(index, index_file)

    # ── Query ────────────────────────────────────────────────────────────────
    if args.query:
        print(f"\nQuery: {args.query}\n")
        print(answer(args.query, index, claude_client, clip_model, clip_processor, device))
    else:
        print("\n--- Multimodal RAG -- Interactive Mode ---")
        print("Type your question and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                break
            print("\nAssistant:", answer(query, index, claude_client, clip_model, clip_processor, device), "\n")


if __name__ == "__main__":
    main()