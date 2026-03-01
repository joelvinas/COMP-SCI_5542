"""
multimodal_rag.py — Image RAG pipeline using sentence-transformers
===================================================================
Loads images and their captions from data/images/, embeds captions
using all-MiniLM-L6-v2, and retrieves relevant images via cosine
similarity between query and caption embeddings.

Usage:
    from multimodal_rag import ImageRAG
    rag = ImageRAG()
    results = rag.search("data pipeline diagram", top_k=2)
"""

import os
import json
import numpy as np
from typing import Optional

# ── Config ──
IMG_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
CAPTIONS_FILE = os.path.join(IMG_DIR, "captions.json")
EMBED_MODEL = "all-MiniLM-L6-v2"  # same lightweight model as text RAG


class ImageRAG:
    """Multimodal RAG pipeline for image retrieval via caption embedding."""

    def __init__(self):
        """Initialize: load images + captions, embed captions."""
        self.images = []       # list of {"filename": ..., "caption": ..., "path": ...}
        self.embeddings = None  # numpy array of shape (n_images, embed_dim)
        self.model = None

        self._load_images()
        self._embed_captions()

    def _load_images(self):
        """Load image metadata and captions from captions.json."""
        if not os.path.exists(CAPTIONS_FILE):
            print(f"Captions file not found: {CAPTIONS_FILE}")
            return

        with open(CAPTIONS_FILE, "r") as f:
            captions = json.load(f)

        for filename, caption in captions.items():
            img_path = os.path.join(IMG_DIR, filename)
            if os.path.exists(img_path):
                self.images.append({
                    "filename": filename,
                    "caption": caption,
                    "path": img_path,
                })

        print(f"Image RAG: loaded {len(self.images)} images from {IMG_DIR}")

    def _embed_captions(self):
        """Embed all captions using sentence-transformers."""
        if not self.images:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            return

        print(f"Loading embedding model: {EMBED_MODEL}...")
        self.model = SentenceTransformer(EMBED_MODEL)
        captions = [img["caption"] for img in self.images]
        self.embeddings = self.model.encode(captions, show_progress_bar=False, convert_to_numpy=True)
        print(f"Embedded {len(captions)} image captions (dim={self.embeddings.shape[1]})")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for relevant images by matching query against captions.

        Args:
            query: Natural language description of desired image.
            top_k: Number of results to return.

        Returns:
            List of dicts with "filename", "caption", "path", "score".
        """
        if self.model is None or self.embeddings is None or len(self.images) == 0:
            return [{"error": "Image RAG not initialized. No images found."}]

        # Embed the query
        query_emb = self.model.encode([query], convert_to_numpy=True)

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1, norms)
        similarities = similarities / norms

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "filename": self.images[idx]["filename"],
                "caption": self.images[idx]["caption"],
                "path": self.images[idx]["path"],
                "score": round(float(similarities[idx]), 4),
            })
        return results


# Singleton instance (lazy loaded)
_image_rag_instance: Optional[ImageRAG] = None


def get_image_rag() -> ImageRAG:
    """Get or create the singleton Image RAG instance."""
    global _image_rag_instance
    if _image_rag_instance is None:
        _image_rag_instance = ImageRAG()
    return _image_rag_instance
