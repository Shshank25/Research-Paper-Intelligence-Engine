# ============================================================
# src/vector_db.py
# Phase 4 — FAISS Vector Store
#
# Objective:
#   Build a FAISS index from embeddings, persist it to disk,
#   reload it, and perform fast nearest-neighbour search.
#
# Architecture:
#   VectorDB class
#     ├── build_index(embeddings)
#     ├── save(index_path, meta_path)
#     ├── load(index_path, meta_path)
#     ├── search(query_vec, top_k) -> list[dict]
#     └── add_embeddings(new_embeddings, new_chunks)
#
# Why FAISS?
#   FAISS (Facebook AI Similarity Search) provides highly
#   optimised C++ similarity search accessible from Python.
#   IndexFlatIP uses inner-product (= cosine when vectors
#   are L2-normalised) — perfect for our use case.
#
# Dependencies: faiss-cpu, numpy, json, logging
# ============================================================

import os
import json
import logging

import numpy as np
import faiss

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FAISS_INDEX_FILE, CHUNKS_META_FILE, TOP_K_RESULTS, LOG_LEVEL

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class VectorDB:
    """
    Wraps a FAISS index alongside chunk metadata for
    building, saving, loading, and querying a vector store.

    Usage:
        db = VectorDB()
        db.build_index(embeddings, chunks)
        db.save()
        # --- later ---
        db = VectorDB()
        db.load()
        results = db.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_FILE,
        meta_path : str = CHUNKS_META_FILE,
    ):
        self.index_path = index_path
        self.meta_path  = meta_path
        self.index      = None   # faiss.Index instance
        self.chunks     = []     # list of chunk dicts (metadata)
        logger.info("VectorDB initialised.")

    # ── Build ────────────────────────────────────────────────

    def build_index(self, embeddings: np.ndarray, chunks: list) -> None:
        """
        Create a new FAISS index from embeddings.

        Args:
            embeddings: numpy array of shape (N, D), L2-normalised.
            chunks    : List of chunk dicts matching the embeddings.

        Raises:
            ValueError: If embeddings is empty or shapes mismatch.
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Cannot build index from empty embeddings.")
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks."
            )

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP | dim={dim} | vectors={len(embeddings)}")

        # IndexFlatIP = exact inner-product search.
        # With L2-normalised vectors, IP == cosine similarity.
        self.index  = faiss.IndexFlatIP(dim)
        self.chunks = chunks

        # FAISS requires float32
        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)

        logger.info(f"Index built. Total vectors stored: {self.index.ntotal}")

    # ── Save ─────────────────────────────────────────────────

    def save(
        self,
        index_path: str | None = None,
        meta_path : str | None = None,
    ) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Args:
            index_path: Override default index file path.
            meta_path : Override default metadata file path.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        idx_path  = index_path or self.index_path
        meta_path = meta_path  or self.meta_path

        os.makedirs(os.path.dirname(idx_path),  exist_ok=True)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)

        faiss.write_index(self.index, idx_path)
        logger.info(f"FAISS index saved → {idx_path}")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Chunk metadata saved → {meta_path}")

    # ── Load ─────────────────────────────────────────────────

    def load(
        self,
        index_path: str | None = None,
        meta_path : str | None = None,
    ) -> None:
        """
        Load FAISS index and chunk metadata from disk.

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        idx_path  = index_path or self.index_path
        meta_path = meta_path  or self.meta_path

        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Chunks metadata not found: {meta_path}")

        self.index = faiss.read_index(idx_path)
        logger.info(f"FAISS index loaded ← {idx_path}  ({self.index.ntotal} vectors)")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Chunk metadata loaded ← {meta_path}  ({len(self.chunks)} chunks)")

    # ── Search ───────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int = TOP_K_RESULTS) -> list:
        """
        Find the top-k most similar chunks to a query vector.

        Args:
            query_vec: 1-D numpy array (embedding_dim,), L2-normalised.
            top_k    : Number of results to return.

        Returns:
            List of result dicts:
            [
              {
                "rank"   : 1,
                "score"  : 0.92,        # cosine similarity
                "chunk_id": 42,
                "source" : "paper.pdf",
                "text"   : "...",
              },
              ...
            ]

        Raises:
            RuntimeError: If index is not loaded.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call build_index() or load() first.")

        query_f32 = query_vec.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_f32, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:               # FAISS returns -1 for missing results
                continue
            chunk = self.chunks[idx].copy()
            chunk["rank"]  = rank
            chunk["score"] = float(score)
            results.append(chunk)

        logger.debug(f"Search returned {len(results)} results.")
        return results

    # ── Incremental add ──────────────────────────────────────

    def add_embeddings(self, new_embeddings: np.ndarray, new_chunks: list) -> None:
        """
        Add more embeddings to an existing index (incremental update).

        Args:
            new_embeddings: numpy array (M, D).
            new_chunks    : Corresponding chunk dicts (length M).
        """
        if self.index is None:
            raise RuntimeError("Index not initialised. Call build_index() first.")

        self.index.add(new_embeddings.astype(np.float32))
        self.chunks.extend(new_chunks)
        logger.info(
            f"Added {len(new_chunks)} chunks. Total: {self.index.ntotal} vectors."
        )

    # ── Utility ──────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the index is built/loaded and non-empty."""
        return self.index is not None and self.index.ntotal > 0

    def get_stats(self) -> dict:
        """Return basic statistics about the current index."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_chunks" : len(self.chunks),
            "sources"      : list({c.get("source", "?") for c in self.chunks}),
        }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.embeddings import EmbeddingGenerator

    chunks = [
        {"chunk_id": 0, "source": "test.pdf", "text": "Attention is all you need."},
        {"chunk_id": 1, "source": "test.pdf", "text": "Transformers use multi-head attention."},
        {"chunk_id": 2, "source": "test.pdf", "text": "BERT fine-tunes on downstream tasks."},
    ]

    gen = EmbeddingGenerator()
    emb = gen.embed_chunks(chunks)

    db = VectorDB()
    db.build_index(emb, chunks)
    db.save()

    print("\n✅ Index built and saved.")
    print(f"   Stats: {db.get_stats()}")

    # Reload and search
    db2 = VectorDB()
    db2.load()
    qvec = gen.embed_query("What model uses self-attention?")
    results = db2.search(qvec, top_k=2)

    print("\n🔍 Search results:")
    for r in results:
        print(f"  Rank {r['rank']} | Score {r['score']:.4f} | {r['source']}")
        print(f"  Text: {r['text']}\n")
