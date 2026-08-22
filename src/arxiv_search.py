# ============================================================
# src/arxiv_search.py
# Agentic Workflow: Paper Discovery Node
# ============================================================

import os
import arxiv
import logging

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, ARXIV_SEARCH_LIMIT, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

class ArxivSearcher:
    def __init__(self, limit: int = ARXIV_SEARCH_LIMIT):
        self.limit = limit
        self.client = arxiv.Client()

    def search_and_download(self, query: str) -> list[dict]:
        """
        Searches arXiv for the query, extracts metadata, and downloads PDFs.
        """
        query_cleaned = query.strip('\'"')
        logger.info(f"Searching arXiv for: '{query_cleaned}' (limit: {self.limit})")
        search = arxiv.Search(
            query = query_cleaned,
            max_results = self.limit,
            sort_by = arxiv.SortCriterion.Relevance
        )

        results = []
        os.makedirs(DATA_DIR, exist_ok=True)

        for result in self.client.results(search):
            paper_info = {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.replace("\n", " "),
                "year": result.published.year,
                "url": result.pdf_url,
                "id": result.entry_id.split("/")[-1]
            }
            
            # Download PDF
            pdf_path = os.path.join(DATA_DIR, f"{paper_info['id']}.pdf")
            paper_info["local_path"] = pdf_path
            
            if not os.path.exists(pdf_path):
                try:
                    logger.info(f"Downloading {paper_info['title'][:50]}...")
                    import urllib.request
                    
                    # Ensure url resolves to a PDF (arXiv links usually do, but adding .pdf can help)
                    download_url = paper_info["url"]
                    if not download_url.endswith(".pdf"):
                        download_url += ".pdf"
                        
                    urllib.request.urlretrieve(download_url, pdf_path)
                except Exception as e:
                    logger.error(f"Failed to download {paper_info['id']}: {e}")
                    continue # Skip if download fails
                    
            results.append(paper_info)

        logger.info(f"Successfully fetched {len(results)} papers.")
        return results
