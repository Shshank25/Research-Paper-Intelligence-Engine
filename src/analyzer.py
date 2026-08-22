# ============================================================
# src/analyzer.py
# Agentic Workflow: Structured Analysis, Comparison & Review
# Phase 5, 6, 7 — Pydantic Insight Extractor & Synthesis Engine
# ============================================================

import os
import json
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_LEVEL
from src.rag_pipeline import RAGPipeline
from src.summarizer import Summarizer

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Structured Data Schema ───────────────────────────────────

class PaperAnalysis(BaseModel):
    title: str = Field(description="Title of the research paper")
    research_problem: str = Field(default="Not explicitly mentioned in the paper.")
    objective: str = Field(default="Not explicitly mentioned in the paper.")
    proposed_method: str = Field(default="Not explicitly mentioned in the paper.")
    dataset: str = Field(default="Not explicitly mentioned in the paper.")
    evaluation_metrics: str = Field(default="Not explicitly mentioned in the paper.")
    experimental_results: str = Field(default="Not explicitly mentioned in the paper.")
    key_findings: Union[str, List[str]] = Field(default="Not explicitly mentioned in the paper.")
    limitations: Union[str, List[str]] = Field(default="Not explicitly mentioned in the paper.")
    future_work: Union[str, List[str]] = Field(default="Not explicitly mentioned in the paper.")


class AgentAnalyzer:
    """
    Analyzes individual research papers using the existing VT RAG Pipeline
    and generates multi-paper comparative matrices and literature reviews.
    """
    def __init__(self, rag_pipeline: Optional[RAGPipeline], summarizer: Optional[Summarizer]):
        self.rag = rag_pipeline
        self.summarizer = summarizer

    def _sanitize_field(self, val: Any) -> str:
        """Helper to ensure clean, non-empty text string."""
        if not val or val is None:
            return "Not explicitly mentioned in the paper."
        text = str(val).strip()
        if not text or "cannot answer" in text.lower() or "not found" in text.lower() or "error" in text.lower():
            return "Not explicitly mentioned in the paper."
        return text

    def analyze_paper(self, paper_title: str, chunks: list, source_name: str) -> dict:
        """
        Phase 5: Extracts structured fields from paper text using the existing RAG pipeline.
        """
        logger.info(f"Extracting structured analysis for: '{paper_title}'")
        
        def safe_ask(question: str) -> str:
            q = f"In the paper '{paper_title}', {question}"
            try:
                if not self.rag:
                    return "Not explicitly mentioned in the paper."
                res = self.rag.answer(q, top_k=3)
                ans = res.get("answer", "")
                return self._sanitize_field(ans)
            except Exception as e:
                logger.error(f"QA extraction error for query '{question}': {e}")
                return "Not explicitly mentioned in the paper."

        problem = safe_ask("what is the main research problem being addressed?")
        objective = safe_ask("what is the primary objective or goal?")
        method = safe_ask("what is the proposed method, model, algorithm, or architecture?")
        dataset = safe_ask("what datasets were used for experiments or evaluation?")
        metrics = safe_ask("what evaluation metrics were used?")
        results = safe_ask("what were the main experimental results, findings, or performance numbers?")

        # Use Summarizer for qualitative insights
        insights = {}
        if self.summarizer and chunks:
            try:
                insights = self.summarizer.full_analysis(chunks, source=source_name)
            except Exception as sum_err:
                logger.error(f"Summarizer error for '{source_name}': {sum_err}")

        key_findings = insights.get("key_findings") or "Not explicitly mentioned in the paper."
        limitations = insights.get("limitations") or "Not explicitly mentioned in the paper."
        future_work = insights.get("future_work") or "Not explicitly mentioned in the paper."

        analysis = PaperAnalysis(
            title=paper_title,
            research_problem=problem,
            objective=objective,
            proposed_method=method,
            dataset=dataset,
            evaluation_metrics=metrics,
            experimental_results=results,
            key_findings=key_findings,
            limitations=limitations,
            future_work=future_work
        )
        
        return analysis.model_dump()

    def compare_papers(self, analyses: List[dict]) -> str:
        """
        Phase 6: Generates a multi-paper comparison matrix in Markdown table format.
        """
        logger.info(f"Generating multi-paper comparison matrix for {len(analyses)} papers...")
        if not analyses:
            return "No analyzed papers available for comparison."

        md = "### ⚖️ Multi-Paper Comparative Matrix\n\n"
        md += "| Paper Title | Research Problem | Methodology | Dataset | Metrics & Results | Key Findings | Limitations |\n"
        md += "|---|---|---|---|---|---|---|\n"
        
        for p in analyses:
            title = p.get('title', 'Unknown').replace('|', '-')
            prob = self._sanitize_field(p.get('research_problem')).replace('|', '-').replace('\n', ' ')
            method = self._sanitize_field(p.get('proposed_method')).replace('|', '-').replace('\n', ' ')
            dataset = self._sanitize_field(p.get('dataset')).replace('|', '-').replace('\n', ' ')
            results = self._sanitize_field(p.get('experimental_results')).replace('|', '-').replace('\n', ' ')
            
            findings = p.get('key_findings', '')
            if isinstance(findings, list):
                findings = findings[0] if findings else "Not explicitly mentioned in the paper."
            findings = self._sanitize_field(findings).replace('|', '-').replace('\n', ' ')

            limits = p.get('limitations', '')
            if isinstance(limits, list):
                limits = limits[0] if limits else "Not explicitly mentioned in the paper."
            limits = self._sanitize_field(limits).replace('|', '-').replace('\n', ' ')

            md += f"| **{title}** | {prob[:120]} | {method[:120]} | {dataset[:90]} | {results[:120]} | {findings[:120]} | {limits[:90]} |\n"

        return md

    def generate_literature_review(self, topic: str, analyses: List[dict]) -> str:
        """
        Phase 7: Synthesizes a structured 6-section literature review grounded strictly in paper analyses.
        """
        logger.info(f"Synthesizing literature review for topic: '{topic}'...")
        if not analyses:
            return "No papers available to generate literature review."

        review = f"# 📝 Literature Review: {topic}\n\n"
        
        # 1. Introduction
        review += "## 1. Introduction\n"
        review += f"This literature review provides a structured synthesis of **{len(analyses)}** peer-reviewed scholarly papers addressing the topic of **{topic}**. "
        review += "The reviewed studies focus on advancing domain-specific solutions, evaluating model performance, and highlighting current research challenges.\n\n"

        # 2. Existing Approaches
        review += "## 2. Existing Approaches\n"
        for p in analyses:
            title = p.get("title", "Unknown")
            prob = self._sanitize_field(p.get("research_problem"))
            obj = self._sanitize_field(p.get("objective"))
            review += f"- **{title}**: Investigates the problem of *{prob}* with the primary objective to *{obj}*.\n"
        review += "\n"

        # 3. Methodology Comparison
        review += "## 3. Methodology Comparison\n"
        for p in analyses:
            title = p.get("title", "Unknown")
            method = self._sanitize_field(p.get("proposed_method"))
            review += f"- **{title}**: Employs **{method}** as its core framework.\n"
        review += "\n"

        # 4. Major Findings & Performance
        review += "## 4. Major Findings & Performance\n"
        for p in analyses:
            title = p.get("title", "Unknown")
            dataset = self._sanitize_field(p.get("dataset"))
            metrics = self._sanitize_field(p.get("evaluation_metrics"))
            results = self._sanitize_field(p.get("experimental_results"))
            review += f"- **{title}**: Evaluated on **{dataset}** using **{metrics}**. Experimental outcomes demonstrated: *{results}*.\n"
        review += "\n"

        # 5. Common Limitations
        review += "## 5. Common Limitations\n"
        for p in analyses:
            title = p.get("title", "Unknown")
            limits = p.get("limitations", "Not explicitly mentioned in the paper.")
            if isinstance(limits, list):
                limits = "; ".join(str(x) for x in limits if x)
            review += f"- **{title}**: {self._sanitize_field(limits)}\n"
        review += "\n"

        # 6. Future Directions
        review += "## 6. Future Directions\n"
        review += "Across the analyzed literature, key avenues for future investigation include:\n"
        for p in analyses:
            title = p.get("title", "Unknown")
            fw = p.get("future_work", "Not explicitly mentioned in the paper.")
            if isinstance(fw, list):
                fw = "; ".join(str(x) for x in fw if x)
            review += f"- **{title}**: {self._sanitize_field(fw)}\n"
        review += "\n---\n*Synthesis generated autonomously by the Agentic AI Research Assistant.*"

        return review
