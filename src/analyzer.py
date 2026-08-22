# ============================================================
# src/analyzer.py
# Agentic Workflow: Structured Analysis & Comparison
# ============================================================

import os
import json
import logging
from pydantic import BaseModel
from typing import List

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

class PaperAnalysis(BaseModel):
    title: str
    research_problem: str
    objective: str
    proposed_method: str
    dataset: str
    evaluation_metrics: str
    experimental_results: str
    key_findings: str | list
    limitations: str | list
    future_work: str | list

class AgentAnalyzer:
    def __init__(self, rag_pipeline: RAGPipeline, summarizer: Summarizer):
        self.rag = rag_pipeline
        self.summarizer = summarizer

    def analyze_paper(self, paper_title: str, chunks: list, source_name: str) -> dict:
        """
        Uses RAG and Summarizer to extract structured fields.
        Assumes the RAGPipeline is already populated with the paper's chunks.
        """
        logger.info(f"Analyzing paper: {paper_title}")
        
        # We need to answer specific questions using RAG.
        # We'll filter the retriever to only look at this specific source if possible,
        # but since our RAGPipeline doesn't currently support source filtering in `answer`,
        # we assume for the agent workflow that we ingest one paper at a time into a temp index,
        # OR we just rely on top_k retrieving relevant info (which works if the index is small).
        # For safety, let's just ask the questions. The agent index will contain only the top-K papers.
        
        def safe_ask(question: str) -> str:
            # We append the paper title to contextulize the query
            q = f"In the paper '{paper_title}', {question}"
            try:
                res = self.rag.answer(q, top_k=3)
                ans = res.get("answer", "Not found.")
                # basic cleanup
                if "I cannot answer this" in ans or "Could not generate" in ans:
                    return "Not explicitly stated."
                return ans
            except Exception as e:
                logger.error(f"Failed to ask '{question}': {e}")
                return "Error during extraction."

        problem = safe_ask("what is the main research problem being addressed?")
        objective = safe_ask("what is the primary objective or goal?")
        method = safe_ask("what is the proposed method, model, or architecture?")
        dataset = safe_ask("what datasets were used for experiments or evaluation?")
        metrics = safe_ask("what evaluation metrics were used?")
        results = safe_ask("what were the main experimental results or performance numbers?")

        # Use summarizer for the rest (Insights)
        logger.info("Extracting insights via Summarizer...")
        insights = self.summarizer.full_analysis(chunks, source=source_name)
        
        analysis = PaperAnalysis(
            title=paper_title,
            research_problem=problem,
            objective=objective,
            proposed_method=method,
            dataset=dataset,
            evaluation_metrics=metrics,
            experimental_results=results,
            key_findings=insights.get("key_findings", "Not found"),
            limitations=insights.get("limitations", "Not found"),
            future_work=insights.get("future_work", "Not found")
        )
        
        return analysis.model_dump()

    def compare_papers(self, analyses: List[dict]) -> str:
        """
        Generates a markdown comparison table from the analyses.
        """
        logger.info(f"Comparing {len(analyses)} papers...")
        if not analyses:
            return "No papers to compare."

        md = "### Multi-Paper Comparison\n\n"
        md += "| Paper | Proposed Method | Dataset | Key Findings | Limitations |\n"
        md += "|---|---|---|---|---|\n"
        
        for p in analyses:
            title = p['title'].replace('|', '-')
            method = str(p['proposed_method']).replace('|', '-').replace('\n', ' ')
            dataset = str(p['dataset']).replace('|', '-').replace('\n', ' ')
            
            # Format findings
            findings = p['key_findings']
            if isinstance(findings, list):
                findings = findings[0] if findings else "None"
            findings = str(findings).replace('|', '-').replace('\n', ' ')
            
            # Format limitations
            limitations = p['limitations']
            if isinstance(limitations, list):
                limitations = limitations[0] if limitations else "None"
            limitations = str(limitations).replace('|', '-').replace('\n', ' ')

            md += f"| **{title}** | {method[:150]}... | {dataset[:100]} | {findings[:150]}... | {limitations[:100]} |\n"

        return md

    def generate_literature_review(self, topic: str, analyses: List[dict]) -> str:
        """
        Synthesizes a structured literature review based on the paper analyses.
        Uses BART summarizer or rule-based templates since we don't have a large generative LLM.
        """
        logger.info("Generating literature review...")
        if not analyses:
            return "No papers available for review."

        review = f"## Literature Review: {topic}\n\n"
        
        review += "### 1. Introduction\n"
        review += f"This review synthesizes findings from {len(analyses)} recently retrieved papers concerning the topic of **{topic}**. "
        review += "The analyzed papers address various research problems and propose novel methodologies to advance the state-of-the-art.\n\n"

        review += "### 2. Existing Approaches & Methodologies\n"
        for p in analyses:
            review += f"- **{p['title']}**: Proposed {p['proposed_method']} to address the problem of {p['research_problem']}.\n"
        review += "\n"

        review += "### 3. Major Findings & Performance\n"
        for p in analyses:
            review += f"- **{p['title']}**: Evaluated on {p['dataset']} using {p['evaluation_metrics']}. Results showed: {p['experimental_results']}.\n"
        review += "\n"
        
        review += "### 4. Common Limitations\n"
        for p in analyses:
            lim = p['limitations']
            if isinstance(lim, list):
                lim = ", ".join(lim)
            review += f"- **{p['title']}**: {lim}\n"
        review += "\n"

        review += "### 5. Future Directions\n"
        review += "Based on the reviewed literature, several promising directions for future research emerge:\n"
        for p in analyses:
            fw = p['future_work']
            if isinstance(fw, list):
                fw = ", ".join(fw)
            review += f"- {fw}\n"
        review += "\n"
        
        review += "---\n*Note: This review was automatically synthesized using the Agentic AI Research Assistant.*"
        
        return review
