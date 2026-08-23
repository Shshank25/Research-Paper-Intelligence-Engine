import os
import sys
from datetime import datetime
from fpdf import FPDF

class TechStackPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, "Research Paper Intelligence Engine -- Technology Stack Specification", new_x="LMARGIN", new_y="NEXT", align="R")
            self.set_draw_color(220, 220, 220)
            self.line(10, 15, 200, 15)
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def safe(text: str) -> str:
    """Sanatize text for standard FPDF Helvetica font (Latin-1)."""
    return text.replace("—", "--").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("•", "*")

def build_tech_stack_pdf(output_filename="Tech_Stack_Specification.pdf"):
    pdf = TechStackPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---------------------------------------------------------
    # COVER PAGE
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.ln(25)
    
    # Title Box
    pdf.set_fill_color(15, 23, 42) # Dark Slate Blue
    pdf.rect(10, 35, 190, 45, style="F")
    
    pdf.set_xy(15, 43)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(180, 12, safe("RESEARCH PAPER INTELLIGENCE ENGINE"), new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(56, 189, 248) # Sky Blue accent
    pdf.cell(180, 8, safe("Comprehensive Technology Stack & System Architecture"), new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_xy(10, 95)
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe("Document Metadata"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(56, 189, 248)
    pdf.set_line_width(0.8)
    pdf.line(10, 104, 200, 104)
    pdf.ln(4)

    metadata = [
        ("Project Name:", "Research Paper Intelligence Engine (VT)"),
        ("Architecture Pattern:", "Agentic RAG (Retrieval-Augmented Generation) & Multi-Stage Workflow"),
        ("Primary Languages:", "Python 3.10+, PowerShell (Automation Scripting)"),
        ("Orchestration Framework:", "LangGraph (StateGraph DAG Engine)"),
        ("Vector DB & Search:", "FAISS (Facebook AI Similarity Search) + Sentence-Transformers"),
        ("UI Framework:", "Streamlit Web Engine (Custom Animated Glassmorphism CSS)"),
        ("Report Generation Date:", datetime.now().strftime("%B %d, %Y - %H:%M:%S")),
    ]

    pdf.set_font("Helvetica", "", 10)
    for label, val in metadata:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(52, 7, safe(label), border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(0, 7, safe(val), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, 175, 190, 65, style="F")
    pdf.set_xy(15, 180)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(180, 7, safe("Executive Architecture Summary"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(51, 65, 85)
    summary_text = (
        "The Research Paper Intelligence Engine is an autonomous, production-grade AI system designed "
        "to discover, process, rank, analyze, and summarize complex academic research papers. "
        "Built on a modern modular Python stack, it combines state-of-the-art Dense Vector Retrieval (FAISS), "
        "Sentence Embeddings (Sentence-Transformers), Local Transformer Models (HuggingFace/PyTorch), "
        "and a 6-node LangGraph Agentic Orchestrator to execute complex multi-paper literature reviews without manual oversight."
    )
    pdf.set_xy(15, 190)
    pdf.multi_cell(180, 5.5, safe(summary_text))

    # ---------------------------------------------------------
    # MAIN TECH STACK SECTIONS
    # ---------------------------------------------------------
    
    sections = [
        {
            "num": "1",
            "title": "User Interface & Presentation Layer",
            "files": ["app.py"],
            "tech": [
                ("Framework", "Streamlit >= 1.30.0", "Python web framework for data & AI apps"),
                ("Styling & Theme", "Vanilla CSS / HTML5", "Glassmorphism UI, custom dark-mode gradients, smooth animated tab transitions"),
                ("Interactivity", "Streamlit Session State", "Stateful management for uploaded files, active tabs, RAG chat history, and workflow logs"),
                ("Media Handling", "PyMuPDF / Streamlit PDF Viewer", "Embedded PDF document previews and interactive chat UI")
            ],
            "desc": "The presentation layer provides a web dashboard featuring multi-tab navigation, custom progress trackers, interactive QA interfaces, and dynamic visualization widgets."
        },
        {
            "num": "2",
            "title": "Agentic AI Workflow & Orchestration Engine",
            "files": ["src/agent.py", "src/analyzer.py", "run_agent_test.py"],
            "tech": [
                ("Orchestrator", "LangGraph >= 0.0.20", "StateGraph directed acyclic graph (DAG) engine for autonomous 6-stage workflows"),
                ("State Management", "TypedDict (ResearchState)", "Strongly typed state container tracking topic, questions, candidates, rankings, and reviews"),
                ("Schema Validation", "Pydantic >= 2.0.0", "BaseModel schema enforcement for structured extraction of problem statements, methods, datasets, and limitations"),
                ("Paper Discovery", "arxiv >= 2.1.0 API Client", "Automated fetching and local PDF downloading from arXiv open-access database")
            ],
            "desc": "The Agentic layer automates research planning, arXiv paper retrieval, semantic relevance ranking, FAISS ingestion, structured analytical extraction, comparative matrix creation, and literature review synthesis."
        },
        {
            "num": "3",
            "title": "Document Ingestion & Text Processing",
            "files": ["src/pdf_processor.py", "src/chunking.py"],
            "tech": [
                ("PDF Parser", "PyMuPDF (fitz) >= 1.23.0", "High-speed text, layout, and metadata extraction from raw PDF files"),
                ("Text Splitter", "LangChain RecursiveCharacterTextSplitter", "Context-aware chunking with customizable chunk sizes (1000 chars) and overlap (200 chars)"),
                ("Sanitization", "Python re (Regex Engine)", "Unicode sanitization, noise removal, reference stripping, and header extraction")
            ],
            "desc": "Converts binary research PDF documents into structured text blocks, preserving page metadata, titles, abstracts, and key section headings for downstream indexing."
        },
        {
            "num": "4",
            "title": "Embedding Generation & Tensor Computing",
            "files": ["src/embeddings.py"],
            "tech": [
                ("Embedding Library", "sentence-transformers >= 2.2.2", "Dense vector embedding generation using Sentence-BERT architectures"),
                ("Default Models", "all-MiniLM-L6-v2 / BAAI/bge-small-en-v1.5", "384-dimensional dense vectors fine-tuned for semantic text similarity"),
                ("Tensor Backend", "PyTorch >= 2.0.0 / NumPy >= 1.24.0", "CPU tensor computations with automated HuggingFace model cache directory management")
            ],
            "desc": "Transforms text chunks and natural language queries into normalized dense vector embeddings for semantic vector indexing."
        },
        {
            "num": "5",
            "title": "Vector Storage & Retrieval Engine",
            "files": ["src/vector_db.py", "src/retriever.py", "src/ranker.py"],
            "tech": [
                ("Vector Store", "FAISS CPU (faiss-cpu >= 1.7.4)", "Facebook AI Similarity Search engine using IndexFlatL2 dense vector index"),
                ("Ranking Engine", "SemanticRanker (Cosine Similarity)", "Reranking candidate papers against research topics using sentence embeddings"),
                ("Persistence", "FAISS binary index + Pickle", "On-disk index persistence and fast warm-start vector database loading")
            ],
            "desc": "Enables millisecond-level similarity search over indexed research paper chunks to retrieve relevant contextual snippets for RAG QA."
        },
        {
            "num": "6",
            "title": "RAG Pipeline & QA Generation",
            "files": ["src/rag_pipeline.py", "src/summarizer.py"],
            "tech": [
                ("Pipeline Framework", "LangChain >= 0.1.0", "Retrieval-Augmented Generation chain combining FAISS retrieval with LLM inference"),
                ("Transformers", "HuggingFace Transformers >= 4.36.0", "Pretrained transformer pipelines for question answering and summarization"),
                ("Default Models", "google/flan-t5-base / facebook/bart-large-cnn", "Sequence-to-sequence text generation and abstractive summarization models"),
                ("Fallback Engine", "Heuristic Context Analyzer", "Deterministic fallback engine ensuring answer availability even in offline / low-resource environments")
            ],
            "desc": "Synthesizes precise, grounded answers to user questions using retrieved paper chunks, preventing hallucinations through strict context constraint."
        },
        {
            "num": "7",
            "title": "Citation Graph Analysis & Export Engine",
            "files": ["src/citation_graph.py", "src/export.py"],
            "tech": [
                ("Citation Engine", "Custom Regex & PyMuPDF", "Parses paper bibliography sections, extracts inline citation keys, and builds paper reference graphs"),
                ("PDF Generator", "fpdf2 >= 2.7.0", "Programmatic PDF document generation for research summary reports, metrics export, and tech specifications")
            ],
            "desc": "Builds citation connectivity graphs and exports comprehensive formatted research reports in PDF format."
        },
        {
            "num": "8",
            "title": "DevOps, Automation & Scripts",
            "files": ["config.py", "setup_local.ps1", "start_app.ps1", "build_colab_notebook.py"],
            "tech": [
                ("Configuration", "Python Config (config.py)", "Centralized environment settings, log levels, model parameters, and path specifications"),
                ("Windows Automation", "PowerShell (ps1)", "Automated virtual environment creation, pip dependency installation, and server startup"),
                ("Notebook Integration", "Jupyter / Google Colab", "Standalone notebook runners (Research_Engine_Standalone.ipynb) for cloud GPU deployment")
            ],
            "desc": "Provides zero-configuration local setup scripts for Windows environments and notebook builders for Google Colab."
        }
    ]

    for sec in sections:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, safe(f"Section {sec['num']}: {sec['title']}"), new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_draw_color(56, 189, 248)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, safe(f"Primary Files: {', '.join(sec['files'])}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 5, safe(sec['desc']))
        pdf.ln(5)

        # Tech Table
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(40, 7, safe(" Component"), border=1, fill=True)
        pdf.cell(50, 7, safe(" Technology / Library"), border=1, fill=True)
        pdf.cell(100, 7, safe(" Description & Role"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 8.5)
        for i, (comp, tech, role) in enumerate(sec['tech']):
            if i % 2 == 0:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            
            y_before = pdf.get_y()
            pdf.set_text_color(15, 23, 42)
            
            # Print row columns
            pdf.cell(40, 6, safe(f" {comp}"), border=1, fill=True)
            pdf.cell(50, 6, safe(f" {tech}"), border=1, fill=True)
            pdf.cell(100, 6, safe(f" {role}"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(8)

    # ---------------------------------------------------------
    # MASTER DEPENDENCY MATRIX
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, safe("Master Dependency & Package Matrix"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(56, 189, 248)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    deps = [
        ("PyMuPDF (fitz)", ">= 1.23.0", "High-performance PDF document parsing and text extraction"),
        ("langchain", ">= 0.1.0", "Core RAG framework, text splitters, and LLM chain abstractions"),
        ("langchain-community", ">= 0.0.20", "Community integrations and extended vector store splitters"),
        ("langgraph", ">= 0.0.20", "StateGraph engine for multi-agent stateful workflow orchestration"),
        ("sentence-transformers", ">= 2.2.2", "Pretrained transformer embeddings for dense semantic retrieval"),
        ("faiss-cpu", ">= 1.7.4", "Facebook AI Similarity Search engine for efficient vector search"),
        ("transformers", ">= 4.36.0", "HuggingFace deep learning models for QA and summarization"),
        ("torch", ">= 2.0.0", "PyTorch deep learning framework for model execution & tensor math"),
        ("streamlit", ">= 1.30.0", "Interactive web dashboard framework"),
        ("pydantic", ">= 2.0.0", "Data validation, state management, and schema enforcement"),
        ("arxiv", ">= 2.1.0", "Python wrapper for arXiv API research paper retrieval"),
        ("fpdf2", ">= 2.7.0", "PDF generation library for reporting and specifications"),
        ("numpy", ">= 1.24.0", "Array and matrix numerical calculations"),
        ("tqdm", ">= 4.66.0", "Console and UI progress bars for ingestion pipelines")
    ]

    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(45, 7, safe(" Package Name"), border=1, fill=True)
    pdf.cell(30, 7, safe(" Version Req."), border=1, fill=True)
    pdf.cell(115, 7, safe(" System Functional Purpose"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 8.5)
    for i, (pkg, ver, purpose) in enumerate(deps):
        if i % 2 == 0:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(45, 6, safe(f" {pkg}"), border=1, fill=True)
        pdf.cell(30, 6, safe(f" {ver}"), border=1, fill=True)
        pdf.cell(115, 6, safe(f" {purpose}"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_filename)
    print(f"Successfully generated PDF: {output_filename}")

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tech_Stack_Specification.pdf")
    build_tech_stack_pdf(out_path)
