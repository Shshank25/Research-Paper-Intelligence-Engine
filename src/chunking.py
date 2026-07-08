# ============================================================
# src/chunking.py
# Phase 2 — Text Chunking
#
# Objective:
#   Split long cleaned text into smaller, overlapping chunks
#   suitable for embedding and retrieval.
#
# Architecture:
#   TextChunker class
#     └── chunk_text(text, metadata) -> list[dict]
#     └── chunk_documents(docs_dict) -> list[dict]
#
# Why chunking matters:
#   Embedding models have a token limit (~512 tokens for
#   MiniLM). Splitting with overlap ensures context is not
#   lost at chunk boundaries.
#
# Dependencies: langchain, logging
# ============================================================

import logging
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP, LOG_LEVEL

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits research paper text into overlapping chunks.

    Each chunk is a dict:
      {
          "chunk_id"   : int,           # global unique index
          "source"     : str,           # source PDF filename
          "text"       : str,           # the chunk content
          "char_start" : int,           # approx start position
      }

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk_text(clean_text, source="paper.pdf")
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries to split on
        # paragraph → sentence → word boundaries in order,
        # falling back to characters only as a last resort.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"TextChunker ready | chunk_size={chunk_size} | overlap={chunk_overlap}"
        )

    # ── Single document ──────────────────────────────────────

    def chunk_text(self, text: str, source: str = "unknown") -> list:
        """
        Chunk a single document's text.

        Args:
            text   : Cleaned text string.
            source : Filename/label for provenance tracking.

        Returns:
            List of chunk dicts (see class docstring).
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for source: {source}")
            return []

        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "chunk_id"  : idx,
                "source"    : source,
                "text"      : chunk_text.strip(),
                "char_start": text.find(chunk_text[:50]) if len(chunk_text) >= 50 else 0,
            })

        logger.info(
            f"'{source}' → {len(chunks)} chunks "
            f"(avg {sum(len(c['text']) for c in chunks) // max(len(chunks),1)} chars each)"
        )
        return chunks

    # ── Multiple documents ───────────────────────────────────

    def chunk_documents(self, docs_dict: dict) -> list:
        """
        Chunk multiple documents and return a flat list with
        globally unique chunk IDs.

        Args:
            docs_dict: Output of PDFProcessor.process_all()
                       {filename: {"clean_text": str, ...}}

        Returns:
            Flat list of all chunks across all documents,
            with globally sequential chunk_ids.
        """
        all_chunks = []
        global_id  = 0

        for filename, doc_data in docs_dict.items():
            clean_text = doc_data.get("clean_text", "")
            chunks     = self.chunk_text(clean_text, source=filename)

            # Re-number chunk_ids globally
            for chunk in chunks:
                chunk["chunk_id"] = global_id
                global_id += 1

            all_chunks.extend(chunks)

        logger.info(f"Total chunks across all documents: {len(all_chunks)}")
        return all_chunks

    # ── Utility ──────────────────────────────────────────────

    def chunk_from_file(self, txt_path: str) -> list:
        """
        Load a saved .txt file and chunk it.

        Args:
            txt_path: Path to a plain-text file.

        Returns:
            List of chunk dicts.
        """
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Text file not found: {txt_path}")

        source = os.path.basename(txt_path)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.chunk_text(text, source=source)


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Attention Is All You Need

    Abstract
    The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks that include an encoder and a decoder.
    The best performing models also connect the encoder and decoder through an
    attention mechanism. We propose a new simple network architecture, the
    Transformer, based solely on attention mechanisms, dispensing with recurrence
    and convolutions entirely.

    1. Introduction
    Recurrent neural networks, long short-term memory and gated recurrent neural
    networks in particular, have been firmly established as state of the art
    approaches in sequence modelling and transduction problems such as language
    modelling and machine translation.
    """ * 10  # repeat to create enough text for chunking

    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    chunks  = chunker.chunk_text(sample, source="test_paper.pdf")

    print(f"\n✅ Created {len(chunks)} chunks.\n")
    for c in chunks[:3]:
        print(f"  Chunk {c['chunk_id']} | {c['source']} | {len(c['text'])} chars")
        print(f"  Preview: {c['text'][:80]}...\n")
