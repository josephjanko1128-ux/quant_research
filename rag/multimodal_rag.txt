"""
Multi-Agent RAG System
- PDF RAG Agent: Reads from two local PDF directories
- Web Agent: Searches and reads from the web
- Orchestrator Agent: Routes queries to the right agent
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

# ── Configuration ─────────────────────────────────────────────────────────────
PDF_DIR_1 = Path(os.getenv("PDF_DIR_1", "./pdfs/collection1"))
PDF_DIR_2 = Path(os.getenv("PDF_DIR_2", "./pdfs/collection2"))

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0,
)

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── PDF Loading ───────────────────────────────────────────────────────────────
def load_pdfs_from_directory(directory: Path, label: str) -> list[Document]:
    """Load all PDFs from a directory."""
    docs = []
    if not directory.exists():
        print(f"[WARNING] Directory '{directory}' does not exist. Creating it...")
        directory.mkdir(parents=True, exist_ok=True)
        return docs

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        print(f"[INFO] No PDFs found in '{directory}'")
        return docs

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_collection"] = label
            docs.extend(loaded)
            print(f"  ✓ Loaded: {pdf_path.name} ({len(loaded)} pages)")
        except Exception as e:
            print(f"  ✗ Failed to load {pdf_path.name}: {e}")

    return docs


def build_vector_store() -> FAISS | None:
    """Build a FAISS vector store from both PDF directories."""
    print("\n📚 Loading PDFs...")
    print(f"  Collection 1: {PDF_DIR_1}")
    docs1 = load_pdfs_from_directory(PDF_DIR_1, "collection1")

    print(f"  Collection 2: {PDF_DIR_2}")
    docs2 = load_pdfs_from_directory(PDF_DIR_2, "collection2")

    all_docs = docs1 + docs2

    if not all_docs:
        print("[WARNING] No documents loaded. RAG tool will return empty results.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"\n✓ Created {len(chunks)} chunks from {len(all_docs)} pages")

    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✓ Vector store built")
    return vector_store


# ── RAG Tool ──────────────────────────────────────────────────────────────────
def make_rag_tool(vector_store: FAISS | None) -> Tool:
    def rag_search(query: str) -> str:
        if vector_store is None:
            return (
                "No PDF documents are currently loaded. "
                "Please add PDFs to the configured directories:\n"
                f"  - {PDF_DIR_1}\n"
                f"  - {PDF_DIR_2}"
            )
        results = vector_store.similarity_search(query, k=4)
        if not results:
            return "No relevant content found in the PDF knowledge base."

        formatted = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            collection = doc.metadata.get("source_collection", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"[Result {i}] Source: {Path(source).name} (Collection: {collection}, Page: {page})\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(formatted)

    return Tool(
        name="pdf_rag_search",
        func=rag_search,
        description=(
            "Search the local PDF knowledge base for information. "
            "Use this for questions about documents, reports, or files stored locally. "
            "Input should be a search query string."
        ),
    )


# ── Web Tool (direct LLM call) ────────────────────────────────────────────────
def make_web_tool() -> Tool:
    def web_search(query: str) -> str:
        try:
            messages = [
                SystemMessage(content=(
                    "You are a knowledgeable research assistant. "
                    "Answer the user's question as thoroughly and accurately as possible "
                    "using your training knowledge. Be factual and cite any relevant context."
                )),
                HumanMessage(content=query),
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"LLM query failed: {e}"

    return Tool(
        name="web_search",
        func=web_search,
        description=(
            "Query the LLM directly for general knowledge, current events up to its training cutoff, "
            "or topics not covered in the local PDF knowledge base. "
            "Input should be a question or search query string."
        ),
    )


# ── Agent Builder ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent research assistant with access to two types of knowledge sources:

1. **pdf_rag_search** – A local knowledge base built from PDF documents in two collections. 
   Use this when the user asks about specific documents, internal reports, or content likely found in local files.

2. **web_search** – Real-time web search. Use this for current events, general knowledge, 
   or topics not likely covered in the local PDFs.

**Strategy:**
- For specific document queries → use pdf_rag_search first
- For current events or broad topics → use web_search
- When uncertain → try pdf_rag_search first, then supplement with web_search if needed
- Synthesize results from multiple sources when helpful

Always cite your sources and be transparent about where information came from."""


def build_orchestrator_agent(tools: list[Tool]):
    """Build a LangGraph ReAct agent with in-memory conversation history."""
    memory = MemorySaver()
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT, checkpointer=memory)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Multi-Agent RAG System")
    print("  Powered by LangChain + Anthropic Claude")
    print("=" * 60)

    # Build components
    vector_store = build_vector_store()
    rag_tool = make_rag_tool(vector_store)
    web_tool = make_web_tool()

    # Build orchestrator
    agent_executor = build_orchestrator_agent([rag_tool, web_tool])

    print("\n✅ System ready! Type 'quit' to exit.\n")
    print("-" * 60)

    # Each session gets a unique thread_id so memory is scoped per run
    thread_id = "session-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        try:
            result = agent_executor.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config,
            )
            # LangGraph returns a list of messages; grab the last AI message
            final = result["messages"][-1].content
            print("\n" + "=" * 60)
            print("📋 ANSWER:")
            print("=" * 60)
            print(final)
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()