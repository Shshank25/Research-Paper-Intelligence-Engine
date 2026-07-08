# 🧠 Research Paper Intelligence Engine — RAG System

> **Phase 1 of the Agentic AI Research Scientist System**
> A complete, local, beginner-friendly RAG pipeline for research papers.

---

## 📌 What This Does

Upload one or more research paper PDFs and:
- **Ask questions** in plain English → get AI-generated answers
- **Generate summaries** of any paper automatically
- **Extract insights**: Key Findings, Limitations, Future Work
- **Explore retrieved chunks** for full transparency

Everything runs **locally** — no OpenAI API key needed.

---

## 🏗️ System Architecture

```
PDF Upload → Text Extraction (PyMuPDF)
         → Text Cleaning
         → Chunking (LangChain)
         → Embeddings (Sentence Transformers)
         → FAISS Index (Vector DB)
         → Semantic Retrieval
         → RAG Pipeline (HuggingFace QA)
         → Streamlit UI
```

---

## 📁 Folder Structure

```
VT/
├── data/                      # Uploaded PDFs (auto-created)
├── documents/                 # Extracted .txt files (auto-created)
├── embeddings/                # Saved .npy embedding arrays
├── vector_store/              # FAISS index + chunk metadata
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py       # Phase 1 — PDF extraction
│   ├── chunking.py            # Phase 2 — Text chunking
│   ├── embeddings.py          # Phase 3 — Embeddings
│   ├── vector_db.py           # Phase 4 — FAISS store
│   ├── retriever.py           # Phase 5 — Semantic search
│   ├── rag_pipeline.py        # Phase 6 — RAG Q&A
│   └── summarizer.py          # Phase 7&8 — Summarization + Insights
├── app.py                     # Phase 9 — Streamlit UI
├── config.py                  # Central configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the app
```bash
streamlit run app.py
```

### Step 3: Use the app
1. Drag & drop PDF(s) in the sidebar
2. Click **Process & Index PDFs**
3. Use the tabs to ask questions, summarize, or extract insights

---

## 🧪 Testing Each Module

You can test each module independently from the `E:\VT` directory:

### Phase 1 — PDF Processor
```bash
python src/pdf_processor.py path/to/paper.pdf
```

### Phase 2 — Chunking
```bash
python src/chunking.py
```

### Phase 3 — Embeddings
```bash
python src/embeddings.py
```

### Phase 4 — FAISS Vector DB
```bash
python src/vector_db.py
```

### Phase 5 — Retriever
```bash
python src/retriever.py
```

### Phase 6 — RAG Pipeline
```bash
python src/rag_pipeline.py
```

### Phase 7 & 8 — Summarizer
```bash
python src/summarizer.py
```

---

## ⚙️ Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `TOP_K_RESULTS` | 5 | Retrieved chunks per query |
| `QA_MODEL` | `deepset/roberta-base-squad2` | Extractive QA model |
| `SUMMARIZATION_MODEL` | `facebook/bart-large-cnn` | Summarization model |

---

## 🤖 Technology Stack

| Component | Technology |
|---|---|
| PDF Extraction | PyMuPDF (`fitz`) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | FAISS (CPU) |
| Question Answering | HuggingFace `deepset/roberta-base-squad2` |
| Summarization | HuggingFace `facebook/bart-large-cnn` |
| UI | Streamlit |

---

## ❗ Common Errors and Fixes

| Error | Fix |
|---|---|
| `No module named 'fitz'` | `pip install PyMuPDF` |
| `No module named 'faiss'` | `pip install faiss-cpu` |
| `No FAISS index found` | Upload and process PDFs first via the UI |
| `CUDA out of memory` | Models run on CPU by default — no GPU needed |
| HuggingFace model download hangs | Check internet connection; models are cached after first download |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |

---

## 🔮 Future Phases (Agentic AI Research Scientist)

- **Phase 2**: Multi-agent research workflow (LangGraph / AutoGen)
- **Phase 3**: Autonomous paper discovery (ArXiv API)
- **Phase 4**: Cross-paper synthesis and citation graph
- **Phase 5**: Research report auto-generation

---

## 👨‍💻 Author

Built as a Vocational Training (VT) Project — Phase 1 of the Agentic AI Research Scientist system.

*June 2026*
