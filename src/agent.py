# ============================================================
# src/agent.py
# Agentic Workflow: LangGraph Orchestrator
# ============================================================

import os
import logging
from typing import TypedDict, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_LEVEL
from langgraph.graph import StateGraph, END

from src.arxiv_search import ArxivSearcher
from src.ranker import SemanticRanker
from src.rag_pipeline import RAGPipeline
from src.summarizer import Summarizer
from src.analyzer import AgentAnalyzer

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── State Definition ─────────────────────────────────────────

class ResearchState(TypedDict):
    topic: str
    raw_papers: List[dict]
    ranked_papers: List[dict]
    selected_papers: List[dict]
    analyses: List[dict]
    comparison: str
    literature_review: str
    status: str

# ── Nodes ────────────────────────────────────────────────────

def search_papers_node(state: ResearchState):
    logger.info("Node: Search Papers")
    state["status"] = "Searching arXiv for relevant papers..."
    searcher = ArxivSearcher()
    papers = searcher.search_and_download(state["topic"])
    state["raw_papers"] = papers
    return state

def rank_papers_node(state: ResearchState):
    logger.info("Node: Rank Papers")
    state["status"] = "Semantically ranking discovered papers..."
    ranker = SemanticRanker()
    ranked = ranker.rank_papers(state["topic"], state["raw_papers"])
    state["ranked_papers"] = ranked
    # For now, just auto-select top papers
    state["selected_papers"] = ranked
    return state

def analyze_papers_node(state: ResearchState):
    logger.info("Node: Analyze Papers")
    state["status"] = "Analyzing selected papers using RAG Engine..."
    
    selected = state.get("selected_papers", [])
    if not selected:
        state["analyses"] = []
        return state

    # Initialize RAG and Summarizer
    rag = RAGPipeline()
    summarizer = Summarizer()
    analyzer = AgentAnalyzer(rag, summarizer)
    
    # Collect all local paths
    pdf_paths = [p["local_path"] for p in selected if p.get("local_path") and os.path.exists(p["local_path"])]
    
    if pdf_paths:
        logger.info(f"Ingesting {len(pdf_paths)} PDFs into RAG...")
        # Rebuild index for these specific papers
        rag.ingest_and_build_index(pdf_paths, force_rebuild=True)
    
    analyses = []
    # Retrieve chunks from state/retriever
    # RAG pipeline retriever has the chunks now
    all_chunks = rag.retriever.vector_db.chunks if hasattr(rag.retriever.vector_db, 'chunks') else []
    
    for paper in selected:
        title = paper.get("title", "Unknown Title")
        source_name = f"{paper.get('id', 'unknown')}.pdf"
        
        # Filter chunks for this paper
        paper_chunks = [c for c in all_chunks if c.get("source") == source_name]
        
        try:
            analysis = analyzer.analyze_paper(title, paper_chunks, source_name)
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"Error analyzing {title}: {e}")
            
    state["analyses"] = analyses
    return state

def compare_papers_node(state: ResearchState):
    logger.info("Node: Compare Papers")
    state["status"] = "Generating multi-paper comparison..."
    analyzer = AgentAnalyzer(None, None) # Don't need RAG/Summ for comparison logic
    comp = analyzer.compare_papers(state["analyses"])
    state["comparison"] = comp
    return state

def review_papers_node(state: ResearchState):
    logger.info("Node: Review Papers")
    state["status"] = "Synthesizing literature review..."
    analyzer = AgentAnalyzer(None, None)
    review = analyzer.generate_literature_review(state["topic"], state["analyses"])
    state["literature_review"] = review
    state["status"] = "Research Analysis Complete."
    return state

# ── Graph Construction ───────────────────────────────────────

def build_research_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("search", search_papers_node)
    workflow.add_node("rank", rank_papers_node)
    workflow.add_node("analyze", analyze_papers_node)
    workflow.add_node("compare", compare_papers_node)
    workflow.add_node("review", review_papers_node)
    
    # Define edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "rank")
    workflow.add_edge("rank", "analyze")
    workflow.add_edge("analyze", "compare")
    workflow.add_edge("compare", "review")
    workflow.add_edge("review", END)
    
    return workflow.compile()

class ResearchAgent:
    def __init__(self):
        self.graph = build_research_graph()
        
    def run(self, topic: str):
        initial_state = ResearchState(
            topic=topic,
            raw_papers=[],
            ranked_papers=[],
            selected_papers=[],
            analyses=[],
            comparison="",
            literature_review="",
            status="Starting..."
        )
        logger.info(f"Starting research workflow for topic: {topic}")
        # Execute workflow
        final_state = self.graph.invoke(initial_state)
        return final_state
