# ============================================================
# src/ranker.py
# Agentic Workflow: Semantic Paper Ranking Node
# Phase 3 — Cosine Similarity Paper Ranker
# ============================================================

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT_TOP_K, LOG_LEVEL
from src.embeddings import EmbeddingGenerator

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class SemanticRanker:
    """
    Ranks discovered paper metadata against the research topic
    using SentenceTransformers ('all-MiniLM-L6-v2') cosine similarity.
    """
    def __init__(self, top_k: int = AGENT_TOP_K, embed_gen: Optional[EmbeddingGenerator] = None):
        self.top_k = top_k
        self.embed_gen = embed_gen or EmbeddingGenerator()

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes normalized cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        score = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(np.clip(score, 0.0, 1.0))

    def rank_papers(
        self,
        topic: str,
        papers: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks papers by abstract similarity against the research topic.
        
        Args:
            topic: Research topic query string.
            papers: List of paper metadata dictionaries from arXiv discovery.
            top_k: Optional override for the number of top papers to select.

        Returns:
            Sorted list of papers containing assigned rank, similarity score,
            title, year, and URL.
        """
        if not papers:
            logger.warning("No papers provided for semantic ranking.")
            return []

        k = top_k if top_k is not None else self.top_k
        logger.info(f"Ranking {len(papers)} candidate papers against topic: '{topic}' (Top-{k})")

        try:
            # 1. Encode Research Topic
            topic_vec = self.embed_gen.model.encode(topic, convert_to_numpy=True)

            # 2. Encode Abstracts
            abstracts = [p.get("abstract", "") or p.get("title", "") for p in papers]
            abstract_vecs = self.embed_gen.model.encode(abstracts, convert_to_numpy=True)

            # 3. Calculate Cosine Similarity per paper
            for i, paper in enumerate(papers):
                score = self.calculate_cosine_similarity(topic_vec, abstract_vecs[i])
                paper["relevance_score"] = round(score, 4)

            # 4. Sort in descending order of relevance
            sorted_papers = sorted(papers, key=lambda x: x.get("relevance_score", 0.0), reverse=True)

            # 5. Assign 1-indexed rank and truncate to Top-K
            top_papers = []
            for rank_idx, paper in enumerate(sorted_papers[:k], start=1):
                paper_copy = dict(paper)
                paper_copy["rank"] = rank_idx
                top_papers.append(paper_copy)

            logger.info(f"Successfully ranked and selected Top-{len(top_papers)} papers.")
            return top_papers

        except Exception as exc:
            logger.error(f"Error during semantic paper ranking: {exc}")
            # Fallback: assign rank 1..N with 0 score
            fallback = []
            for rank_idx, p in enumerate(papers[:k], start=1):
                p_copy = dict(p)
                p_copy["rank"] = rank_idx
                p_copy["relevance_score"] = 0.0
                fallback.append(p_copy)
            return fallback
