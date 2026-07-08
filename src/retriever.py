# ============================================================
# src/retriever.py
# Phase 5 — Semantic Search / Retriever
#
# Objective:
#   Provide a clean, high-level API that takes a natural-
#   language question and returns the most relevant text
#   chunks from the FAISS index.
#
# Architecture:
#   Retriever class
#     └── retrieve(query, top_k) -> list[dict]
#     └── retrieve_with_context(query, top_k) -> str (joined)
#
# This module is the "R" in RAG — it bridges the query
# and the vector store, formatting results for the LLM.
#
# Dependencies: embeddings.py, vector_db.py
# ============================================================

import os
import logging

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_RESULTS, LOG_LEVEL
from src.embeddings import EmbeddingGenerator
from src.vector_db  import VectorDB

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic search over the FAISS index.

    Usage:
        retriever = Retriever()
        retriever.load_index()
        results = retriever.retrieve("What are the key findings?")
        context = retriever.retrieve_with_context("What is BERT?")
    """

    def __init__(
        self,
        embedding_gen: EmbeddingGenerator | None = None,
        vector_db    : VectorDB | None = None,
    ):
        """
        Args:
            embedding_gen: Pre-initialised EmbeddingGenerator.
                           If None, a new one is created automatically.
            vector_db    : Pre-initialised VectorDB.
                           If None, a new one is created; call load_index().
        """
        self.embedding_gen = embedding_gen or EmbeddingGenerator()
        self.vector_db     = vector_db     or VectorDB()
        logger.info("Retriever initialised.")

    # ── Index management ─────────────────────────────────────

    def load_index(self) -> None:
        """Load the FAISS index and chunk metadata from disk."""
        self.vector_db.load()
        logger.info("Retriever: index loaded and ready.")

    def is_ready(self) -> bool:
        """Return True if the index is loaded and non-empty."""
        return self.vector_db.is_ready

    # ── Core retrieval ───────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list:
        """
        Embed a query and return the top-k matching chunks.

        Args:
            query  : Natural language question or search string.
            top_k  : Number of results to return.

        Returns:
            List of chunk dicts, each with a 'score' field.

        Raises:
            RuntimeError: If the index is not ready.
            ValueError  : If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        if not self.is_ready():
            raise RuntimeError(
                "Vector index is not ready. Call load_index() or build the index first."
            )

        logger.info(f"Retrieving top-{top_k} chunks for query: '{query[:60]}...' ")
        query_vec = self.embedding_gen.embed_query(query)
        results   = self.vector_db.search(query_vec, top_k=top_k)

        logger.info(f"  Retrieved {len(results)} chunks.")
        for r in results:
            logger.debug(
                f"  Rank {r['rank']} | score={r['score']:.4f} | source={r['source']}"
            )

        return results

    # ── Context string for LLM prompt ────────────────────────

    def retrieve_with_context(
        self,
        query : str,
        top_k : int = TOP_K_RESULTS,
        sep   : str = "\n\n---\n\n",
    ) -> str:
        """
        Retrieve top-k chunks and join them into a single
        context string for use in an LLM prompt.

        Args:
            query: The user's question.
            top_k: Number of chunks to include.
            sep  : Separator between chunks.

        Returns:
            A multi-paragraph string with source annotations.
        """
        results  = self.retrieve(query, top_k=top_k)
        parts    = []

        for r in results:
            header = f"[Source: {r['source']} | Rank: {r['rank']} | Score: {r['score']:.3f}]"
            parts.append(f"{header}\n{r['text']}")

        context = sep.join(parts)
        logger.debug(f"Context length: {len(context):,} characters")
        return context

    # ── Source-filtered retrieval ─────────────────────────────

    def retrieve_from_source(
        self,
        query : str,
        source: str,
        top_k : int = TOP_K_RESULTS,
    ) -> list:
        """
        Retrieve chunks restricted to a specific source document.

        Args:
            query : The search query.
            source: Filename of the paper to restrict to.
            top_k : Number of results to return.

        Returns:
            Filtered list of chunk dicts.
        """
        all_results = self.retrieve(query, top_k=top_k * 3)  # over-fetch then filter
        filtered = [r for r in all_results if r.get("source") == source]
        return filtered[:top_k]


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    # This test assumes vector_db.py was run first to build the index
    retriever = Retriever()

    if not retriever.is_ready():
        print("⚠️  Index not found — run vector_db.py first to build the index.")
    else:
        results = retriever.retrieve("What is the attention mechanism?", top_k=3)
        print(f"\n✅ Retrieved {len(results)} results.\n")
        for r in results:
            print(f"  Rank {r['rank']} | Score {r['score']:.4f} | {r['source']}")
            print(f"  {r['text'][:120]}…\n")
