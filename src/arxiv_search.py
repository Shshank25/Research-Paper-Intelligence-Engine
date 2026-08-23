# ============================================================
# src/arxiv_search.py
# Agentic Workflow: Scholarly Paper Discovery (arXiv API)
# Phase 2 — Robust arXiv Search & Metadata Extractor
# ============================================================

import os
import re
import logging
import urllib.request
import urllib.error
import arxiv
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, ARXIV_SEARCH_LIMIT, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class ArxivSearcher:
    """
    Independent paper discovery module using the official arXiv API.
    Retrieves scholarly paper metadata and manages PDF downloads.
    """
    def __init__(self, limit: int = ARXIV_SEARCH_LIMIT):
        self.limit = limit
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=3
        )

    def clean_text(self, text: str) -> str:
        """Removes extra line breaks and collapses whitespace."""
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text.replace("\n", " "))
        return cleaned.strip()

    def search_papers(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Searches arXiv for papers matching the research query.
        Returns structured metadata list without downloading PDFs.
        
        Output format per paper:
        - title (str)
        - authors (List[str])
        - abstract (str)
        - publication_date (str YYYY-MM-DD)
        - year (int)
        - arxiv_id / id (str)
        - url (str PDF link)
        """
        max_results = limit if limit is not None else self.limit
        cleaned_query = self.clean_text(query.strip('\'"'))
        
        if not cleaned_query:
            logger.warning("Empty search query provided to ArxivSearcher.")
            return []

        logger.info(f"Querying arXiv API for: '{cleaned_query}' (limit: {max_results})")
        
        try:
            search = arxiv.Search(
                query=cleaned_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in self.client.results(search):
                arxiv_id = result.entry_id.split("/")[-1]
                pub_date = result.published.strftime("%Y-%m-%d") if result.published else "N/A"
                pub_year = result.published.year if result.published else 0
                
                pdf_url = result.pdf_url
                if pdf_url and not pdf_url.endswith(".pdf"):
                    pdf_url += ".pdf"

                paper_info = {
                    "title": self.clean_text(result.title),
                    "authors": [a.name for a in result.authors],
                    "abstract": self.clean_text(result.summary),
                    "publication_date": pub_date,
                    "year": pub_year,
                    "arxiv_id": arxiv_id,
                    "id": arxiv_id,
                    "url": pdf_url,
                    "local_path": os.path.join(DATA_DIR, f"{arxiv_id}.pdf")
                }
                results.append(paper_info)

            logger.info(f"Successfully discovered {len(results)} papers from arXiv.")
            return results

        except (urllib.error.URLError, TimeoutError) as net_err:
            logger.error(f"Network error during arXiv search for '{cleaned_query}': {net_err}")
            return []
        except Exception as exc:
            logger.error(f"Unexpected error querying arXiv API for '{cleaned_query}': {exc}")
            return []

    def download_paper_pdf(self, paper_info: Dict[str, Any]) -> Optional[str]:
        """
        Downloads PDF for a specific paper into data/ directory if not present.
        Returns the local file path on success, or None on failure.
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        arxiv_id = paper_info.get("arxiv_id") or paper_info.get("id") or "paper"
        pdf_path = paper_info.get("local_path") or os.path.join(DATA_DIR, f"{arxiv_id}.pdf")
        
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            logger.info(f"Paper PDF already cached locally: {os.path.basename(pdf_path)}")
            return pdf_path

        download_url = paper_info.get("url")
        if not download_url:
            logger.warning(f"No PDF URL found for paper: {paper_info.get('title')}")
            return None

        try:
            logger.info(f"Downloading PDF for '{paper_info.get('title', '')[:40]}...' from {download_url}")
            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(pdf_path, "wb") as out_file:
                out_file.write(response.read())
            logger.info(f"Downloaded PDF → {pdf_path}")
            return pdf_path
        except Exception as exc:
            logger.error(f"Failed to download PDF for '{paper_info.get('title')}': {exc}")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return None

    def search_and_download(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Backwards-compatible method: Searches arXiv and downloads PDFs for discovered papers.
        """
        papers = self.search_papers(query, limit=limit)
        valid_papers = []
        for paper in papers:
            pdf_path = self.download_paper_pdf(paper)
            if pdf_path:
                paper["local_path"] = pdf_path
                valid_papers.append(paper)
        return valid_papers
