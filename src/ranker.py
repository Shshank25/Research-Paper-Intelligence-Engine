# ============================================================
# src/ranker.py
# Agentic Workflow: Semantic Paper Ranking Node
# ============================================================

import os
import numpy as np
import logging

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
    def __init__(self, top_k: int = AGENT_TOP_K):
        self.top_k = top_k
        self.embed_gen = EmbeddingGenerator()

    def rank_papers(self, topic: str, papers: list[dict]) -> list[dict]:
        """
        Ranks papers by computing cosine similarity between the topic
        and the paper abstracts.
        """
        if not papers:
            return []
            
        logger.info(f"Ranking {len(papers)} papers against topic: '{topic}'")
        
        # Embed the topic
        topic_embedding = self.embed_gen.model.encode(topic)
        
        # Embed abstracts
        abstracts = [p["abstract"] for p in papers]
        abstract_embeddings = self.embed_gen.model.encode(abstracts)
        
        # Calculate cosine similarities
        scores = []
        for emb in abstract_embeddings:
            # Cosine similarity
            score = np.dot(topic_embedding, emb) / (np.linalg.norm(topic_embedding) * np.linalg.norm(emb))
            scores.append(float(score))
            
        # Add scores to papers
        for i, paper in enumerate(papers):
            paper["relevance_score"] = scores[i]
            
        # Sort and take top_k
        ranked_papers = sorted(papers, key=lambda x: x["relevance_score"], reverse=True)
        top_papers = ranked_papers[:self.top_k]
        
        logger.info(f"Selected top {len(top_papers)} papers based on relevance.")
        return top_papers
