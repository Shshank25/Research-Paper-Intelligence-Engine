# ============================================================
# src/rag_pipeline.py
# Phase 6 — Retrieval-Augmented Generation (RAG) Pipeline
#
# Objective:
#   Answer user questions by:
#     1. Retrieving relevant context chunks (Retriever)
#     2. Building a prompt with context + question
#     3. Running a local HuggingFace QA model for the answer
#
# Architecture:
#   RAGPipeline class
#     ├── answer(question, top_k) -> dict
#     │     returns: answer, context, sources, confidence
#     └── _build_prompt(question, context) -> str
#
# Model: deepset/roberta-base-squad2
#   - Extractive QA: finds the answer span within the context
#   - Works fully offline after first download
#   - Fast on CPU
#
# For generative answers (replacing extractive QA with a
# seq2seq model), swap to google/flan-t5-base and use the
# text2text-generation pipeline.
#
# Dependencies: transformers, retriever.py
# ============================================================

import os
import logging

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QA_MODEL, TOP_K_RESULTS, LOG_LEVEL
from src.retriever import Retriever

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
from threading import Thread

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Combine semantic retrieval with extractive QA to answer
    natural language questions from your research papers.

    Usage:
        rag = RAGPipeline()
        result = rag.answer("What dataset was used for evaluation?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        retriever : Retriever | None = None,
        qa_model  : str = QA_MODEL,
    ):
        """
        Args:
            retriever: Pre-initialised Retriever.
                       If None, a new one is created; index auto-loaded.
            qa_model : HuggingFace model ID for QA.
        """
        # ── Retriever setup ──
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = Retriever()
            # Try to load the index; warn if not available yet
            try:
                self.retriever.load_index()
            except FileNotFoundError:
                logger.warning(
                    "No vector index found. "
                    "Call build_index_from_pdfs() or ingest_pdfs() first."
                )

        # ── QA model setup ──
        logger.info(f"Loading QA model: {qa_model}")
        logger.info("  (First run downloads the model from HuggingFace — once only)")
        self.qa_model_name = qa_model
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(qa_model)
        logger.info("RAGPipeline ready.")

    # ── Core QA ──────────────────────────────────────────────

    def answer(self, question: str, top_k: int = TOP_K_RESULTS, stream: bool = False) -> dict:
        """
        Answer a question using retrieved context.

        Args:
            question: The user's natural language question.
            top_k   : Number of context chunks to retrieve.
            stream  : If True, returns a 'streamer' and 'thread' instead of 'answer'.

        Returns:
            {
                "question"  : str,
                "answer"    : str,   # if stream=False
                "streamer"  : TextIteratorStreamer, # if stream=True
                "thread"    : Thread, # if stream=True
                "confidence": float,
                "context"   : str,
                "sources"   : list,
            }

        Raises:
            RuntimeError: If the retriever index is not ready.
            ValueError  : If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        if not self.retriever.is_ready():
            raise RuntimeError(
                "No vector index found. Please upload and index papers first."
            )

        # Step 1: Retrieve relevant chunks
        logger.info(f"RAG → Question: '{question[:80]}'")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        # Step 2: Build context from retrieved chunks
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}]: {c['text']}")
        context = "\n\n".join(context_parts)
        sources = [{"source": c["source"], "score": c["score"]} for c in chunks]

        # Step 3: Run Generative QA model
        logger.info(f"Running QA model on context ({len(context)} chars) …")
        prompt = f"Answer the following question based ONLY on the provided context. If the context does not contain the answer, explicitly state 'I cannot answer this based on the provided text.' Do not use outside knowledge.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.3,
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            return {
                "question"  : question,
                "streamer"  : streamer,
                "thread"    : thread,
                "confidence": 1.0,
                "context"   : context,
                "sources"   : sources,
            }
        else:
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=250,
                        num_beams=4,
                        early_stopping=True
                    )
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                confidence = 1.0
            except Exception as exc:
                logger.error(f"QA model failed: {exc}")
                answer     = "Could not generate an answer. Please rephrase your question."
                confidence = 0.0

            logger.info(f"Answer: '{answer[:80]}'")

            return {
                "question"  : question,
                "answer"    : answer,
                "confidence": confidence,
                "context"   : context,
                "sources"   : sources,
            }

    # ── Full ingest pipeline (convenience) ───────────────────

    def ingest_and_build_index(
        self,
        pdf_paths: list,
        force_rebuild: bool = False,
    ) -> dict:
        """
        Full pipeline: extract PDFs → chunk → embed → build FAISS index.
        Call this once when new PDFs are uploaded.

        Args:
            pdf_paths    : List of paths to PDF files.
            force_rebuild: If True, rebuild even if index exists.

        Returns:
            {
                "num_pdfs"  : int,
                "num_chunks": int,
                "index_ready": bool,
            }
        """
        import shutil
        from src.pdf_processor import PDFProcessor
        from src.chunking      import TextChunker
        from src.embeddings    import EmbeddingGenerator
        from src.vector_db     import VectorDB
        from config            import DATA_DIR

        logger.info(f"Ingesting {len(pdf_paths)} PDF(s) …")

        # Copy PDFs to data dir
        os.makedirs(DATA_DIR, exist_ok=True)
        for p in pdf_paths:
            dest = os.path.join(DATA_DIR, os.path.basename(p))
            if not os.path.isfile(dest) or force_rebuild:
                shutil.copy2(p, dest)

        # Phase 1 — Extract
        processor = PDFProcessor()
        docs      = {}
        for p in pdf_paths:
            filename = os.path.basename(p)
            result   = processor.process_single(p)
            docs[filename] = result

        # Phase 2 — Chunk
        chunker    = TextChunker()
        all_chunks = chunker.chunk_documents(docs)

        # Phase 3 — Embed
        gen        = EmbeddingGenerator()
        embeddings = gen.embed_chunks(all_chunks)
        gen.save_embeddings(embeddings)

        # Phase 4 — Index
        db = VectorDB()
        db.build_index(embeddings, all_chunks)
        db.save()

        # Update retriever's reference
        self.retriever.vector_db = db
        logger.info("Ingest complete. Index ready.")

        return {
            "num_pdfs"   : len(pdf_paths),
            "num_chunks" : len(all_chunks),
            "index_ready": True,
        }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGPipeline()
    if rag.retriever.is_ready():
        result = rag.answer("What is the main contribution of this paper?")
        print(f"\n✅ Answer     : {result['answer']}")
        print(f"   Confidence : {result['confidence']:.3f}")
        print(f"   Sources    : {[s['source'] for s in result['sources']]}")
    else:
        print("⚠️  No index found. Please ingest PDFs first.")
