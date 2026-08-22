# ============================================================
# src/agent.py
# Agentic AI Research Assistant — LangGraph Orchestrator
# Phase 1: Research Planner & State Graph Definition
# ============================================================

import os
import logging
from typing import TypedDict, List, Dict, Any

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
    research_topic: str
    topic: str                      # Alias for backwards compatibility
    research_questions: List[str]   # Decomposed questions guiding the analysis
    discovered_papers: List[dict]   # Raw papers returned by arXiv search
    raw_papers: List[dict]          # Alias for backwards compatibility
    ranked_papers: List[dict]       # Papers sorted by semantic relevance
    selected_papers: List[dict]     # Top-K papers selected for RAG ingestion
    paper_analysis: List[dict]      # Structured insight dictionaries per paper
    analyses: List[dict]            # Alias for backwards compatibility
    comparison: str                 # Markdown / HTML comparison matrix
    literature_review: str          # Multi-paper structured synthesis
    status: str                     # Current execution status for UI
    errors: List[str]               # Accumulated non-fatal execution errors


# ── Nodes ────────────────────────────────────────────────────

def plan_research_node(state: ResearchState) -> ResearchState:
    """
    Stage 1: Research Planner Node
    Decomposes the research topic into structured guiding research questions.
    """
    topic = state.get("research_topic") or state.get("topic", "")
    logger.info(f"Node [1/6]: Research Planner for topic: '{topic}'")
    state["status"] = "Planning research scope and formulating key questions..."
    
    # Deterministic decomposition into core analytical dimensions
    questions = [
        f"What are the foundational methodologies and models proposed for {topic}?",
        f"What empirical datasets and evaluation metrics are used to validate {topic}?",
        f"What are the major performance outcomes, key findings, and limitations in current {topic} research?"
    ]
    
    state["research_questions"] = questions
    state["topic"] = topic
    state["research_topic"] = topic
    logger.info(f"Formulated {len(questions)} research questions.")
    return state


def search_papers_node(state: ResearchState) -> ResearchState:
    """
    Stage 2: Paper Search Node
    Queries arXiv API for papers matching the research topic.
    """
    topic = state.get("research_topic") or state.get("topic", "")
    logger.info(f"Node [2/6]: Paper Search for: '{topic}'")
    state["status"] = "Searching arXiv for relevant scholarly papers..."
    
    try:
        searcher = ArxivSearcher()
        papers = searcher.search_and_download(topic)
        state["discovered_papers"] = papers
        state["raw_papers"] = papers
        logger.info(f"Discovered {len(papers)} papers from arXiv.")
    except Exception as exc:
        err_msg = f"Paper Search Error: {str(exc)}"
        logger.error(err_msg)
        state["errors"].append(err_msg)
        state["discovered_papers"] = []
        state["raw_papers"] = []
        
    return state


def rank_papers_node(state: ResearchState) -> ResearchState:
    """
    Stage 3: Paper Ranking Node
    Ranks discovered papers using Sentence Transformers cosine similarity.
    """
    topic = state.get("research_topic") or state.get("topic", "")
    papers = state.get("discovered_papers") or state.get("raw_papers", [])
    logger.info(f"Node [3/6]: Paper Ranking for {len(papers)} papers against '{topic}'")
    state["status"] = "Semantically ranking papers against research topic..."
    
    if not papers:
        logger.warning("No papers to rank.")
        state["ranked_papers"] = []
        state["selected_papers"] = []
        return state

    try:
        ranker = SemanticRanker()
        ranked = ranker.rank_papers(topic, papers)
        state["ranked_papers"] = ranked
        state["selected_papers"] = ranked
        logger.info(f"Successfully ranked and selected top {len(ranked)} papers.")
    except Exception as exc:
        err_msg = f"Paper Ranking Error: {str(exc)}"
        logger.error(err_msg)
        state["errors"].append(err_msg)
        state["ranked_papers"] = papers
        state["selected_papers"] = papers
        
    return state


def analyze_papers_node(state: ResearchState) -> ResearchState:
    """
    Stage 4: Paper Analysis Node
    Ingests selected papers into RAG and extracts structured analytical fields.
    """
    logger.info("Node [4/6]: Paper Analysis")
    state["status"] = "Ingesting papers into RAG engine & extracting insights..."
    
    selected = state.get("selected_papers", [])
    if not selected:
        state["paper_analysis"] = []
        state["analyses"] = []
        return state

    try:
        rag = RAGPipeline()
        summarizer = Summarizer()
        analyzer = AgentAnalyzer(rag, summarizer)
        
        pdf_paths = [p["local_path"] for p in selected if p.get("local_path") and os.path.exists(p["local_path"])]
        
        if pdf_paths:
            logger.info(f"Ingesting {len(pdf_paths)} paper PDF(s) into FAISS Vector DB...")
            rag.ingest_and_build_index(pdf_paths, force_rebuild=True)
        
        analyses = []
        all_chunks = rag.retriever.vector_db.chunks if hasattr(rag.retriever.vector_db, 'chunks') else []
        
        for paper in selected:
            title = paper.get("title", "Unknown Title")
            source_name = f"{paper.get('id', 'unknown')}.pdf"
            paper_chunks = [c for c in all_chunks if c.get("source") == source_name]
            
            try:
                analysis = analyzer.analyze_paper(title, paper_chunks, source_name)
                analyses.append(analysis)
            except Exception as paper_exc:
                logger.error(f"Error analyzing paper '{title}': {paper_exc}")
                state["errors"].append(f"Analysis error for '{title}': {str(paper_exc)}")
                
        state["paper_analysis"] = analyses
        state["analyses"] = analyses
    except Exception as exc:
        err_msg = f"Global Paper Analysis Error: {str(exc)}"
        logger.error(err_msg)
        state["errors"].append(err_msg)
        state["paper_analysis"] = []
        state["analyses"] = []
        
    return state


def compare_papers_node(state: ResearchState) -> ResearchState:
    """
    Stage 5: Multi-Paper Comparison Node
    Generates a structured comparison matrix across all analyzed papers.
    """
    logger.info("Node [5/6]: Multi-Paper Comparison")
    state["status"] = "Generating multi-paper comparative matrix..."
    
    analyses = state.get("paper_analysis") or state.get("analyses", [])
    try:
        analyzer = AgentAnalyzer(None, None)
        comp = analyzer.compare_papers(analyses)
        state["comparison"] = comp
    except Exception as exc:
        err_msg = f"Comparison Generation Error: {str(exc)}"
        logger.error(err_msg)
        state["errors"].append(err_msg)
        state["comparison"] = "Unable to generate comparison matrix."
        
    return state


def review_papers_node(state: ResearchState) -> ResearchState:
    """
    Stage 6: Literature Review Node
    Synthesizes a grounded literature review document from paper analyses.
    """
    logger.info("Node [6/6]: Literature Review Synthesis")
    state["status"] = "Synthesizing multi-paper literature review..."
    
    topic = state.get("research_topic") or state.get("topic", "")
    analyses = state.get("paper_analysis") or state.get("analyses", [])
    
    try:
        analyzer = AgentAnalyzer(None, None)
        review = analyzer.generate_literature_review(topic, analyses)
        state["literature_review"] = review
        state["status"] = "Agentic Research Analysis Complete."
    except Exception as exc:
        err_msg = f"Literature Review Error: {str(exc)}"
        logger.error(err_msg)
        state["errors"].append(err_msg)
        state["literature_review"] = "Unable to generate literature review."
        state["status"] = "Workflow Completed with Errors."
        
    return state


# ── Graph Construction ───────────────────────────────────────

def build_research_graph():
    """
    Compiles the 6-stage LangGraph workflow.
    """
    workflow = StateGraph(ResearchState)
    
    # Register logical nodes
    workflow.add_node("plan", plan_research_node)
    workflow.add_node("search", search_papers_node)
    workflow.add_node("rank", rank_papers_node)
    workflow.add_node("analyze", analyze_papers_node)
    workflow.add_node("compare", compare_papers_node)
    workflow.add_node("review", review_papers_node)
    
    # Define linear execution flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "rank")
    workflow.add_edge("rank", "analyze")
    workflow.add_edge("analyze", "compare")
    workflow.add_edge("compare", "review")
    workflow.add_edge("review", END)
    
    return workflow.compile()


class ResearchAgent:
    """
    High-level runner interface for the Agentic AI Research Assistant.
    """
    def __init__(self):
        self.graph = build_research_graph()
        
    def run(self, topic: str) -> dict:
        initial_state = ResearchState(
            research_topic=topic,
            topic=topic,
            research_questions=[],
            discovered_papers=[],
            raw_papers=[],
            ranked_papers=[],
            selected_papers=[],
            paper_analysis=[],
            analyses=[],
            comparison="",
            literature_review="",
            status="Initiating Research Workflow...",
            errors=[]
        )
        logger.info(f"Starting 6-stage Research Workflow for topic: '{topic}'")
        final_state = self.graph.invoke(initial_state)
        return final_state
