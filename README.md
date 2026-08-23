<div align="center">
  
# 🧠 Research Paper Intelligence Engine

**A complete, local, beginner-friendly RAG pipeline for research papers.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Overview](#-overview) • [Motivation](#-motivation) • [Features](#-features) • [System Architecture](#-system-architecture) • [Technology Stack](#-technology-stack) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots & Demo](#-screenshots--demo) • [Project Structure](#-project-structure) • [Future Work](#-future-work)

</div>

---

## 📌 Overview

**7th Semester Update:** This project has been upgraded to an **Agentic AI Research Assistant**. It now features an end-to-end automated workflow that takes a research topic, discovers relevant scholarly papers (via arXiv), semantically ranks them, extracts structured insights, generates a multi-paper comparison, and synthesizes a literature review.

You can still upload your own PDFs to the manual RAG pipeline, but the new Agentic Workflow tab handles the entire research lifecycle autonomously!

---

## 💡 Motivation

Reading and synthesizing academic research papers is highly time-consuming. Researchers and students often spend hours parsing through dense PDFs just to extract key methodologies or findings. This project was built to automate the extraction of knowledge, serving as a personal, privacy-first AI Research Assistant that drastically cuts down literature review time.

---

## ✨ Features

- 💬 **Ask questions** in plain English and get AI-generated answers grounded in the text.
- 📝 **Generate summaries** of any paper automatically.
- 🔍 **Extract insights**: Identify Key Findings, Limitations, and Future Work instantly.
- 📖 **Explore retrieved chunks** for full transparency and citation checking.

---

## 🏗️ System Architecture

This project employs a **LangGraph-driven Agentic Workflow** leveraging the existing RAG pipeline as a tool:

```mermaid
graph TD
    A[Research Topic Input] --> B(Paper Discovery\narXiv API)
    B --> C(Semantic Ranking\nSentence Transformers)
    C --> D(Top-K Papers)
    
    D --> E[Structured Analysis via RAG]
    E -.->|Ingest| F[(FAISS Vector DB)]
    F -.->|Retrieve| G(QA Pipeline\nHuggingFace)
    G -.->|Extract| E
    
    E --> H(Multi-Paper Comparison)
    H --> I(Literature Review Synthesis)
    I --> J[Streamlit Dashboard]
```

---

## 🤖 Technology Stack & Versions

We use robust open-source libraries for this local pipeline:

| Component | Technology | Version |
|---|---|---|
| **Language** | Python | `>=3.9` |
| **PDF Extraction** | PyMuPDF (`fitz`) | `1.24.5` |
| **Text Splitting** | LangChain | `0.2.x` |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) | `3.0.1` |
| **Vector Store** | FAISS (CPU) | `1.8.0` |
| **Agent Orchestration** | LangGraph | `>=0.0.20` |
| **Paper Discovery** | arXiv API | `>=2.1.0` |
| **QA Model** | HuggingFace `deepset/roberta-base-squad2` | `transformers` |
| **Summarization** | HuggingFace `facebook/bart-large-cnn` | `transformers` |
| **Frontend** | Streamlit | `>=1.30.0` |

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shshank25/Research-Paper-Intelligence-Engine.git
   cd Research-Paper-Intelligence-Engine
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Usage

1. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```
2. **Interact!**
   - Open your browser to `http://localhost:8501`.
   - **(NEW) Agentic Workflow Tab:** Enter a research topic and watch the AI discover, rank, analyze, compare, and review papers autonomously.
   - **(Legacy) Manual RAG:** Drag & drop your PDF(s) into the sidebar, process them, and Ask Questions, Summarize, or Extract Insights manually.

---

## 📸 Screenshots & Demo

### Live Demo
![Demo](assets/demo.webp)

### UI Screenshots
*(Add screenshots of your application here)*
- **Home Interface:** `![Home Interface](assets/home.png)`
- **PDF Upload:** `![Upload](assets/upload.png)`
- **Question Answering:** `![QA](assets/qa.png)`
- **Paper Summary:** `![Summary](assets/summary.png)`

---

## 📁 Project Structure

```text
Research-Paper-Intelligence-Engine/
├── src/
│   ├── agent.py               # NEW: LangGraph orchestrator
│   ├── analyzer.py            # NEW: Structured extraction & comparison
│   ├── arxiv_search.py        # NEW: Paper discovery module
│   ├── ranker.py              # NEW: Semantic ranking module
│   ├── pdf_processor.py       # Extract text from PDFs
│   ├── chunking.py            # Break text into semantic chunks
│   ├── embeddings.py          # Generate vector embeddings
│   ├── vector_db.py           # Manage FAISS index
│   ├── retriever.py           # Perform semantic search
│   ├── rag_pipeline.py        # Execute RAG Q&A
│   └── summarizer.py          # Generate summaries & insights
├── data/                      # Uploaded PDFs (auto-created)
├── documents/                 # Extracted .txt files (auto-created)
├── embeddings/                # Saved .npy embedding arrays
├── vector_store/              # FAISS index + chunk metadata
├── assets/                    # Demo videos and screenshots
├── app.py                     # Main Streamlit UI
├── config.py                  # Global configurations
└── run.py                     # App Launcher script
```

---

## 🔮 Future Work (Phase 3)

This 7th-semester project successfully transitioned from a passive RAG system to an active **Agentic AI Research Assistant**. Future enhancements may include:

- **Knowledge Graph Integration:** Mapping relationships between citations and concepts across multiple papers.
- **Research Gap Detection:** Using LLMs to automatically identify unaddressed areas in current literature.
- **Hypothesis & Experiment Generation:** Generating proposals based on detected research gaps.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <i>Built with ❤️ as a Vocational Training (VT) Project.</i><br>
  <strong>Phase 1 of the Agentic AI Research Scientist system.</strong>
</div>
