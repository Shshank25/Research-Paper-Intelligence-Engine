# ============================================================
# src/pdf_processor.py
# Phase 1 — PDF Processing
#
# Objective:
#   Extract raw text from one or many PDF files, clean it,
#   and save plain-text versions to the documents/ folder.
#
# Architecture:
#   PDFProcessor class
#     ├── extract_text(pdf_path) -> str
#     ├── clean_text(raw_text)   -> str
#     └── process_all(pdf_dir)  -> dict[filename, clean_text]
#
# Dependencies: PyMuPDF (fitz), os, logging, re
# ============================================================

import os
import re
import logging
import unicodedata
import fitz  # PyMuPDF

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DOCUMENTS_DIR, LOG_LEVEL

# ── Logger setup ─────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles extraction and cleaning of text from PDF research papers.

    Usage:
        processor = PDFProcessor()
        texts = processor.process_all()   # reads from data/
        # or process a single file:
        text = processor.extract_text("path/to/paper.pdf")
    """

    def __init__(self, data_dir: str = DATA_DIR, docs_dir: str = DOCUMENTS_DIR):
        self.data_dir = data_dir
        self.docs_dir = docs_dir
        os.makedirs(self.docs_dir, exist_ok=True)
        logger.info("PDFProcessor initialised.")
        logger.info(f"  Input  dir : {self.data_dir}")
        logger.info(f"  Output dir : {self.docs_dir}")

    # ── Core extraction ──────────────────────────────────────

    def extract_text(self, pdf_path: str = None, stream: bytes = None) -> str:
        """
        Extract raw text from a single PDF file page by page.

        Args:
            pdf_path: Absolute or relative path to the PDF (optional).
            stream: Raw bytes of the PDF file (optional).

        Returns:
            A single string containing all pages concatenated.
        """
        if stream is not None:
            logger.info("Extracting text from memory stream")
            try:
                doc = fitz.open(stream=stream, filetype="pdf")
            except Exception as exc:
                raise RuntimeError(f"Failed to open PDF from stream: {exc}") from exc
        elif pdf_path is not None:
            if not os.path.isfile(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
            try:
                doc = fitz.open(pdf_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc
        else:
            raise ValueError("Must provide either pdf_path or stream")

        pages_text = []
        try:
            for page in doc:
                pages_text.append(page.get_text("text"))
            doc.close()
        except Exception as exc:
            raise RuntimeError(f"Error during extraction: {exc}") from exc

        raw_text = "\n".join(pages_text)
        logger.info(f"  Extracted {len(raw_text):,} chars.")
        return raw_text

    # ── Text cleaning ────────────────────────────────────────

    def clean_text(self, raw_text: str) -> str:
        """
        Remove artefacts common in PDF-extracted text:
          - Excessive whitespace / blank lines
          - Hyphenated line breaks (re-join words)
          - Non-ASCII junk characters
          - Page headers/footers patterns

        Args:
            raw_text: The raw string from extract_text().

        Returns:
            A cleaner, more readable string.
        """
        text = raw_text

        # Normalize Unicode ligatures (e.g. \ufb01 -> fi, \ufb02 -> fl)
        text = unicodedata.normalize("NFKD", text)

        # Re-join hyphenated words split across lines (common in PDFs)
        text = re.sub(r"-\n", "", text)

        # Remove excessive newlines (3+ → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace non-breaking spaces and tabs with regular space
        text = text.replace("\xa0", " ").replace("\t", " ")

        # Remove lines that are just page numbers (e.g. "— 3 —" or just "3")
        text = re.sub(r"^\s*[\-–—]?\s*\d+\s*[\-–—]?\s*$", "", text, flags=re.MULTILINE)

        # Collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Final strip
        text = text.strip()

        logger.debug(f"  Cleaned text length: {len(text):,} characters.")
        return text

    # ── Save to disk ─────────────────────────────────────────

    def save_text(self, filename: str, text: str) -> str:
        """
        Save cleaned text to documents/ folder as a .txt file.

        Args:
            filename: Base name (e.g. 'paper1.pdf' → 'paper1.txt').
            text: The cleaned text string.

        Returns:
            Path to the saved .txt file.
        """
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(self.docs_dir, f"{base}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"  Saved cleaned text → {out_path}")
        return out_path

    # ── Batch processing ─────────────────────────────────────

    def process_all(self) -> dict:
        """
        Process every PDF in data_dir.

        Returns:
            {
                "paper1.pdf": {
                    "raw_text": "...",
                    "clean_text": "...",
                    "txt_path": "documents/paper1.txt"
                },
                ...
            }
        """
        pdf_files = [
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            return {}

        logger.info(f"Found {len(pdf_files)} PDF(s) to process.")
        results = {}

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_dir, pdf_file)
            try:
                raw   = self.extract_text(pdf_path)
                clean = self.clean_text(raw)
                path  = self.save_text(pdf_file, clean)
                results[pdf_file] = {
                    "raw_text"  : raw,
                    "clean_text": clean,
                    "txt_path"  : path,
                }
            except (FileNotFoundError, RuntimeError) as exc:
                logger.error(f"Skipping '{pdf_file}': {exc}")

        logger.info(f"Processing complete. {len(results)}/{len(pdf_files)} PDFs succeeded.")
        return results

    # ── Single-file convenience method ───────────────────────

    def process_single(self, pdf_path: str = None, stream: bytes = None, filename: str = None) -> dict:
        """
        Extract, clean, save and return result for one PDF.

        Args:
            pdf_path: Path to the PDF file (optional).
            stream: Raw bytes of the PDF file (optional).
            filename: Override the filename (optional).

        Returns:
            Dict with keys: raw_text, clean_text, txt_path.
        """
        if filename is None:
            filename = os.path.basename(pdf_path) if pdf_path else "document.pdf"
            
        raw_text  = self.extract_text(pdf_path=pdf_path, stream=stream)
        clean     = self.clean_text(raw_text)
        txt_path  = self.save_text(filename, clean)
        return {
            "raw_text"  : raw_text,
            "clean_text": clean,
            "txt_path"  : txt_path,
        }


# ── Quick test (run this file directly) ─────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        processor = PDFProcessor()
        result = processor.process_single(pdf)
        print(f"\n✅ Extracted {len(result['clean_text']):,} characters.")
        print("First 500 chars:\n", result["clean_text"][:500])
    else:
        processor = PDFProcessor()
        results = processor.process_all()
        print(f"\n✅ Processed {len(results)} PDF(s).")
