# ============================================================
# src/embeddings.py
# Phase 3 — Embedding Generation
#
# Objective:
#   Convert text chunks into dense vector representations
#   (embeddings) using a local Sentence Transformer model.
#
# Architecture:
#   EmbeddingGenerator class
#     ├── embed_chunks(chunks)       -> np.ndarray  (N, D)
#     ├── embed_query(query)         -> np.ndarray  (D,)
#     ├── save_embeddings(arr, path)
#     └── load_embeddings(path)      -> np.ndarray
#
# Model: all-MiniLM-L6-v2
#   - 384-dimensional embeddings
#   - Fast on CPU, excellent quality for semantic search
#   - Downloads automatically from HuggingFace on first use
#
# Dependencies: sentence-transformers, numpy, tqdm, logging
# ============================================================

import os
import logging

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, EMBEDDINGS_DIR, LOG_LEVEL

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates sentence embeddings for text chunks and queries.

    Usage:
        gen = EmbeddingGenerator()
        embeddings = gen.embed_chunks(chunks)   # shape: (N, 384)
        query_vec  = gen.embed_query("What is attention?")
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        logger.info("  (First run downloads ~90 MB from HuggingFace — once only)")
        self.model_name = model_name
        self._model     = None
        self.dim        = None

    def _load_model(self):
        """Lazy-initialise the embedding model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self.dim    = self._model.get_sentence_embedding_dimension()
            logger.info(f"  Model ready | embedding dim = {self.dim}")

    @property
    def model(self):
        """Property that auto-loads the model on first access."""
        self._load_model()
        return self._model

    # ── Embed chunks ─────────────────────────────────────────

    def embed_chunks(self, chunks: list, batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of chunk dicts.

        Args:
            chunks    : List of chunk dicts with key 'text'.
            batch_size: How many texts to encode at once.

        Returns:
            numpy array of shape (N, embedding_dim).

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("chunks list is empty — nothing to embed.")

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size}) …")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise for cosine similarity
        )

        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    # ── Embed single query ───────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The user question or search string.

        Returns:
            1-D numpy array of shape (embedding_dim,).
        """
        if not query or not query.strip():
            raise ValueError("Query string is empty.")

        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        logger.debug(f"Query embedded | shape: {vec.shape}")
        return vec

    # ── Persistence ──────────────────────────────────────────

    def save_embeddings(self, embeddings: np.ndarray, name: str = "embeddings") -> str:
        """
        Save embeddings to disk as a .npy file.

        Args:
            embeddings: numpy array to save.
            name      : Base filename (without extension).

        Returns:
            Path to the saved file.
        """
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
        np.save(path, embeddings)
        logger.info(f"Embeddings saved → {path}  ({embeddings.shape})")
        return path

    def load_embeddings(self, name: str = "embeddings") -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            name: Base filename (without .npy extension).

        Returns:
            numpy array of embeddings.

        Raises:
            FileNotFoundError: If the .npy file does not exist.
        """
        path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        embeddings = np.load(path)
        logger.info(f"Embeddings loaded ← {path}  ({embeddings.shape})")
        return embeddings


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": 0, "source": "test.pdf", "text": "Attention is all you need."},
        {"chunk_id": 1, "source": "test.pdf", "text": "Transformers changed NLP forever."},
        {"chunk_id": 2, "source": "test.pdf", "text": "BERT uses bidirectional attention."},
    ]

    gen = EmbeddingGenerator()
    emb = gen.embed_chunks(sample_chunks)
    print(f"\n✅ Chunk embeddings shape: {emb.shape}")

    qvec = gen.embed_query("What is the transformer model?")
    print(f"✅ Query embedding shape : {qvec.shape}")

    # Test cosine similarity (arrays are already normalised)
    scores = emb @ qvec
    for i, s in enumerate(scores):
        print(f"  Chunk {i} similarity: {s:.4f}")
