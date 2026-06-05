"""
rag_service.py — FAISS vectorstore loading and semantic retrieval.
Extracted from app.py monolith for isolated testability.
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_vectorstore(vectorstore_path: Path, embedding_model_name: str):
    """Load and return a FAISS vectorstore. Returns None on failure."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        if not vectorstore_path.exists():
            logger.error("❌ Vectorstore path not found: %s", vectorstore_path)
            return None

        vs = FAISS.load_local(
            folder_path=str(vectorstore_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("✓ Vectorstore loaded from %s", vectorstore_path)
        return vs
    except Exception as exc:
        logger.error("❌ Error loading vectorstore: %s", exc)
        return None


def get_relevant_context(vectorstore, query: str, k: int = 3) -> str:
    """Retrieve top-k semantically relevant chunks from the vectorstore."""
    if vectorstore is None:
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as exc:
        logger.error("Error retrieving context: %s", exc)
        return ""
