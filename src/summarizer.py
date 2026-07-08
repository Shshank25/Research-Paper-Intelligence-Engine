# ============================================================
# src/summarizer.py
# Phase 7 & 8 — Summarization + Research Insights Extraction
#
# Objective:
#   1. Generate a concise abstract-style summary of a paper.
#   2. Extract structured research insights:
#      - Key Findings
#      - Limitations
#      - Future Work
#
# Architecture:
#   Summarizer class
#     ├── summarize(text)               -> str
#     ├── extract_insights(text)        -> dict
#     └── full_analysis(chunks, source) -> dict
#
# Strategy:
#   - Use BART (facebook/bart-large-cnn) for summarization.
#   - Use keyword-pattern extraction as a reliable, fast way
#     to pull structured sections without needing a GPU.
#   - Fallback: retrieval-based extraction when sections
#     are not explicitly labelled.
#
# Dependencies: transformers, retriever.py (optional)
# ============================================================

import re
import os
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUMMARIZATION_MODEL, LOG_LEVEL

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Section header patterns (case-insensitive) ────────────────
_FINDINGS_PATTERNS  = [
    r"(?i)(key\s+(?:findings?|contributions?|results?|insights?|discoveries))",
    r"(?i)(main\s+(?:findings?|contributions?|results?))",
    r"(?i)(conclusion[s]?|summary\s+of\s+(?:findings?|results?))",
    r"(?i)(our\s+(?:results?|findings?|contributions?))",
]

_LIMITATIONS_PATTERNS = [
    r"(?i)(limitation[s]?)",
    r"(?i)(drawback[s]?|weakness(?:es)?)",
    r"(?i)(constraint[s]?)",
    r"(?i)(we\s+do\s+not\s+address|beyond\s+the\s+scope)",
]

_FUTURE_WORK_PATTERNS = [
    r"(?i)(future\s+work|future\s+direction[s]?|future\s+research)",
    r"(?i)(open\s+problem[s]?|open\s+question[s]?)",
    r"(?i)(further\s+research|further\s+investigation)",
    r"(?i)(promising\s+(?:avenue[s]?|direction[s]?))",
]


class Summarizer:
    """
    Summarizes research papers and extracts structured insights.

    Usage:
        summ = Summarizer()
        summary   = summ.summarize(full_text)
        insights  = summ.extract_insights(full_text)
        analysis  = summ.full_analysis(chunks, source="paper.pdf")
    """

    def __init__(self, model_name: str = SUMMARIZATION_MODEL):
        logger.info(f"Loading summarization model: {model_name}")
        logger.info("  (First run downloads ~1.6 GB from HuggingFace — once only)")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Lazy-initialise the summarization model."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info("Summarization model loaded.")

    # ── Summarization ────────────────────────────────────────

    def summarize(
        self,
        text        : str,
        max_length  : int = 300,
        min_length  : int = 80,
        chunk_limit : int = 3000,   # chars to feed to BART (token limit ~1024)
    ) -> str:
        """
        Generate a concise summary of the provided text.

        BART has a token limit, so we truncate to chunk_limit
        characters before passing to the model.

        Args:
            text       : Full paper or section text.
            max_length : Max summary tokens.
            min_length : Min summary tokens.
            chunk_limit: Max input characters (BART token budget).

        Returns:
            A concise summary string.
        """
        if not text or not text.strip():
            return "No text provided for summarization."

        # Truncate to stay within model token limit
        input_text = text[:chunk_limit]

        logger.info(f"Summarizing {len(input_text):,} characters …")
        try:
            self._load_model()
            inputs = self.tokenizer([input_text], return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.info(f"Summary generated ({len(summary)} chars).")
            return summary
        except Exception as exc:
            logger.error(f"Summarization failed: {exc}")
            return f"Summarization error: {exc}"

    # ── Section extraction ───────────────────────────────────

    def _extract_section(self, text: str, patterns: list) -> list:
        """
        Extract sentences that follow a matched section header.

        Strategy:
          1. Find the first heading that matches any pattern.
          2. Extract up to 1000 characters.
          3. Split into sentences and discard any incomplete trailing sentence.

        Args:
            text    : Full paper text.
            patterns: List of regex patterns for section headers.

        Returns:
            List of sentences.
        """
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                start = match.end() # Start AFTER the heading
                snippet = text[start : start + 1000]
                # Clean up newlines
                snippet = snippet.replace('\n', ' ')
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', snippet)
                
                # Discard the last sentence if it's incomplete
                if sentences and not re.search(r'[.!?]$', sentences[-1].strip()):
                    sentences = sentences[:-1]
                
                # Filter out very short fragments
                cleaned = [s.strip() for s in sentences if len(s.strip()) > 20]
                if cleaned:
                    return cleaned

        return []

    def _extract_bullet_sentences(self, text: str, keywords: list) -> list:
        """
        Find sentences containing specific keywords and return as list.

        Fallback extraction when sections aren't explicitly labelled.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        hits = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw.lower() in sent_lower for kw in keywords):
                cleaned = sent.strip()
                if len(cleaned) > 20:  # skip very short fragments
                    hits.append(cleaned)
        return hits[:5]  # return top 5 matches

    # ── Insights extraction ──────────────────────────────────

    def extract_insights(self, text: str) -> dict:
        """
        Extract structured research insights from paper text.

        Returns:
            {
                "key_findings": str or list,
                "limitations" : str or list,
                "future_work" : str or list,
            }

        Strategy:
          - First, try to find explicit section headings.
          - If not found, fall back to sentence-level keyword search.
        """
        logger.info("Extracting research insights …")

        # ── Key Findings ──
        findings_text = self._extract_section(text, _FINDINGS_PATTERNS)
        if findings_text:
            key_findings = findings_text
        else:
            sentences = self._extract_bullet_sentences(
                text,
                ["we show", "we propose", "we demonstrate", "our method",
                 "achieves", "outperforms", "significantly", "novel", "key finding"]
            )
            key_findings = sentences if sentences else ["Not explicitly stated in the paper."]

        # ── Limitations ──
        limit_text = self._extract_section(text, _LIMITATIONS_PATTERNS)
        if limit_text:
            limitations = limit_text
        else:
            sentences = self._extract_bullet_sentences(
                text,
                ["limitation", "drawback", "weakness", "constraint",
                 "does not", "cannot", "unable to", "restricted to"]
            )
            limitations = sentences if sentences else ["Not explicitly stated in the paper."]

        # ── Future Work ──
        future_text = self._extract_section(text, _FUTURE_WORK_PATTERNS)
        if future_text:
            future_work = future_text
        else:
            sentences = self._extract_bullet_sentences(
                text,
                ["future work", "future research", "further investigation",
                 "open problem", "promising direction", "plan to", "will explore"]
            )
            future_work = sentences if sentences else ["Not explicitly stated in the paper."]

        logger.info("Insights extraction complete.")
        return {
            "key_findings": key_findings,
            "limitations" : limitations,
            "future_work" : future_work,
        }

    # ── Full analysis (convenience) ──────────────────────────

    def full_analysis(self, chunks: list, source: str = "paper") -> dict:
        """
        Run summary + insights on a set of chunks from one paper.

        Args:
            chunks: List of chunk dicts with 'text' and 'source'.
            source: Name of the paper for logging.

        Returns:
            {
                "source"      : str,
                "summary"     : str,
                "key_findings": str or list,
                "limitations" : str or list,
                "future_work" : str or list,
            }
        """
        logger.info(f"Full analysis for: {source}")

        # Reconstruct full text from chunks
        full_text = "\n\n".join(
            c["text"] for c in chunks
            if c.get("source") == source or source == "all"
        )

        if not full_text.strip():
            full_text = "\n\n".join(c["text"] for c in chunks)

        summary  = self.summarize(full_text)
        insights = self.extract_insights(full_text)

        return {
            "source"      : source,
            "summary"     : summary,
            **insights,
        }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    Abstract
    We propose the Transformer, a new model architecture based entirely on attention
    mechanisms, dispensing with recurrence and convolutions. The model achieves state-
    of-the-art results on machine translation tasks.

    Key Findings
    Our model achieves 28.4 BLEU on WMT 2014 English-to-German translation, improving
    over existing best results by over 2 BLEU. We demonstrate that Transformers
    generalize well to other tasks by applying them to English constituency parsing.

    Limitations
    The model has a limitation in handling very long sequences due to the quadratic
    memory complexity of self-attention. We do not address streaming or online inference.

    Future Work
    Future work will explore sparse attention mechanisms to reduce computational cost.
    We plan to investigate applying Transformers to video and audio tasks.
    """

    summ = Summarizer()
    print("\n📝 Summary:")
    print(summ.summarize(sample_text))

    print("\n🔍 Insights:")
    insights = summ.extract_insights(sample_text)
    for key, val in insights.items():
        print(f"\n  {key.upper()}:")
        if isinstance(val, list):
            for item in val:
                print(f"    • {item}")
        else:
            print(f"    {val}")
