# 🧠 Research Paper Intelligence Engine - ChatGPT Codebase Guide

Welcome! This ZIP archive contains the complete source code, notebooks, and architectural overview for the **Research Paper Intelligence Engine** — an Agentic AI Research Assistant & local RAG pipeline.

---

## 📁 Repository Map & File Descriptions

### 🟢 Core Application & Entry Points
- `app.py`: Streamlit multi-tab web application (Agentic Workflow, Manual RAG Chat, arXiv Search, Citation Graph, Literature Review Synthesis, Export).
- `run.py`: Entry script to launch the Streamlit dashboard.
- `Research_Paper_Intelligence_Engine.ipynb`: Primary interactive Jupyter Notebook.
- `Research_Engine_Standalone.ipynb`: Standalone single-file notebook implementation.
- `colab_runner.ipynb`: Runner notebook optimized for Google Colab environment.
- `config.py`: Global configuration parameters (model names, path settings, score thresholds).
- `requirements.txt`: Dependencies specification.
- `setup_local.ps1` & `start_app.ps1`: PowerShell convenience setup and run scripts.

### ⚙️ Engine Modules (`src/`)
- [agent.py](file:///e:/VT/src/agent.py): LangGraph & arXiv autonomous agent orchestrating research steps.
- [rag_pipeline.py](file:///e:/VT/src/rag_pipeline.py): Core RAG pipeline combining vector retrieval with QA model synthesis.
- [pdf_processor.py](file:///e:/VT/src/pdf_processor.py): Text and metadata extraction from PDF files (PyMuPDF / pdfplumber).
- [chunking.py](file:///e:/VT/src/chunking.py): Text chunking strategies (Recursive Character Splitting, Semantic Chunking).
- [embeddings.py](file:///e:/VT/src/embeddings.py): Vector embedding generator (`sentence-transformers/all-MiniLM-L6-v2`).
- [vector_db.py](file:///e:/VT/src/vector_db.py): FAISS vector store indexing and query manager.
- [retriever.py](file:///e:/VT/src/retriever.py): Dense semantic retrieval & hybrid search logic.
- [ranker.py](file:///e:/VT/src/ranker.py): Cross-encoder re-ranking for query context relevance.
- [summarizer.py](file:///e:/VT/src/summarizer.py): HuggingFace summarization (`facebook/bart-large-cnn`).
- [analyzer.py](file:///e:/VT/src/analyzer.py): Paper insight extraction (Key Findings, Methodologies, Limitations).
- [citation_graph.py](file:///e:/VT/src/citation_graph.py): NetworkX citation graph construction & visualization.
- [arxiv_search.py](file:///e:/VT/src/arxiv_search.py): ArXiv API search and automated paper downloader.
- [export.py](file:///e:/VT/src/export.py): Export engine (PDF, HTML, Markdown, BibTeX).

---

## 🏗️ System Architecture

```
[Research Query / PDF Input]
            │
            ▼
   ┌─────────────────┐
   │ PDF Extraction  │ ──► PyMuPDF / pdfplumber
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Text Chunking   │ ──► Recursive / Semantic Splitting
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Vector Encoding │ ──► SentenceTransformers (all-MiniLM-L6-v2)
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ FAISS Vector DB │ ──► Cosine Similarity Index
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Hybrid Retrieval│ ──► Top-K Candidates + Cross-Encoder Re-ranking
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ QA & Synthesis  │ ──► HuggingFace RoBERTa / BART
   └─────────────────┘
```

---

## 💬 ChatGPT Prompting Recommendation
When uploading this zip to ChatGPT, ask questions such as:
1. *"Explain the architecture and data flow of this RAG pipeline in detail."*
2. *"How does the Agentic workflow in `src/agent.py` integrate with the RAG pipeline in `src/rag_pipeline.py`?"*
3. *"Review the codebase for potential optimizations, edge-case bug fixes, or performance improvements."*
