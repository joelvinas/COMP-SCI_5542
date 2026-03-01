"""
rag_pipeline.py — Document RAG pipeline using sentence-transformers
====================================================================
Loads PDF documents, chunks text, embeds with all-MiniLM-L6-v2,
and retrieves relevant chunks via cosine similarity.

Usage:
    from rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    results = rag.search("What is Snowflake?", top_k=3)
"""

import os
import json
import numpy as np
from typing import Optional

# ── Config ──
DOC_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
EMBED_MODEL = "all-MiniLM-L6-v2"  # ~80MB, fast, good quality
CHUNK_SIZE = 300  # words per chunk
CHUNK_OVERLAP = 50  # word overlap between chunks


class RAGPipeline:
    """Text-based RAG pipeline for PDF document retrieval."""

    def __init__(self):
        """Initialize the RAG pipeline: load PDFs, chunk, and embed."""
        self.chunks = []       # list of {"text": ..., "source": ..., "chunk_id": ...}
        self.embeddings = None  # numpy array of shape (n_chunks, embed_dim)
        self.model = None

        self._load_and_chunk()
        self._embed_chunks()

    def _load_and_chunk(self):
        """Load all PDFs from DOC_DIR and split into chunks."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            print("PyPDF2 not installed. Run: pip install PyPDF2")
            return

        if not os.path.exists(DOC_DIR):
            print(f"Document directory not found: {DOC_DIR}")
            return

        for filename in sorted(os.listdir(DOC_DIR)):
            if not filename.lower().endswith(".pdf"):
                continue
            filepath = os.path.join(DOC_DIR, filename)
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + " "

                # Chunk the text
                words = full_text.split()
                for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk_words = words[i:i + CHUNK_SIZE]
                    if len(chunk_words) < 20:  # skip tiny tail chunks
                        continue
                    chunk_text = " ".join(chunk_words)
                    self.chunks.append({
                        "text": chunk_text,
                        "source": filename,
                        "chunk_id": len(self.chunks),
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        print(f"RAG Pipeline: loaded {len(self.chunks)} chunks from {DOC_DIR}")

    def _embed_chunks(self):
        """Embed all chunks using sentence-transformers."""
        if not self.chunks:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            return

        print(f"Loading embedding model: {EMBED_MODEL}...")
        self.model = SentenceTransformer(EMBED_MODEL)
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        print(f"Embedded {len(texts)} chunks (dim={self.embeddings.shape[1]})")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for relevant document chunks.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with "text", "source", "score", "chunk_id".
        """
        if self.model is None or self.embeddings is None or len(self.chunks) == 0:
            return [{"error": "RAG pipeline not initialized. No documents found."}]

        # Embed the query
        query_emb = self.model.encode([query], convert_to_numpy=True)

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        similarities = similarities / norms

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": round(float(similarities[idx]), 4),
                "chunk_id": self.chunks[idx]["chunk_id"],
            })
        return results


# Singleton instance (lazy loaded)
_rag_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the singleton RAG pipeline instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance
