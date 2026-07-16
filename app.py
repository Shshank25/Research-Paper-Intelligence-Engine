# ============================================================
# app.py
# Phase 9 — Streamlit UI
#
# Research Paper Intelligence Engine
# A complete web interface for uploading PDFs, asking
# questions via RAG, generating summaries, and extracting
# structured research insights.
# ============================================================

import os
import sys
import tempfile
import logging

# Move HuggingFace cache to E: drive immediately because C: drive is full
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")

import streamlit as st

# ── Path setup (so src/ is importable) ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import TOP_K_RESULTS
from src.pdf_processor import PDFProcessor
from src.chunking      import TextChunker
from src.embeddings    import EmbeddingGenerator
from src.vector_db     import VectorDB
from src.retriever     import Retriever
from src.rag_pipeline  import RAGPipeline
from src.summarizer    import Summarizer

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Streamlit Page Configuration
# ============================================================
st.set_page_config(
    page_title  = "Research Paper Intelligence Engine",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ============================================================
# Custom CSS — Premium Dark Theme
# ============================================================
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary   : #0d1117;
    --bg-secondary : #161b22;
    --bg-card      : #1c2128;
    --accent-blue  : #58a6ff;
    --accent-purple: #bc8cff;
    --accent-green : #3fb950;
    --accent-amber : #d29922;
    --text-primary : #e6edf3;
    --text-muted   : #8b949e;
    --border       : #30363d;
    --radius       : 12px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

/* ── Headers ── */
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 600; }
h1 { font-size: 1.8rem; background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.3rem; color: var(--accent-blue); border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
h3 { font-size: 1.1rem; color: var(--accent-purple); }

/* ── Cards ── */
.rag-card {
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-radius : var(--radius);
    padding       : 1.2rem 1.5rem;
    margin-bottom : 1rem;
    transition    : border-color 0.2s ease;
}
.rag-card:hover { border-color: var(--accent-blue); }

/* ── Answer box ── */
.answer-box {
    background    : linear-gradient(135deg, #1a2744, #1c2128);
    border        : 1px solid var(--accent-blue);
    border-radius : var(--radius);
    padding       : 1.2rem 1.5rem;
    font-size     : 1.05rem;
    line-height   : 1.7;
    margin-bottom : 1rem;
}

/* ── Insight sections ── */
.insight-findings  { border-left: 4px solid var(--accent-green);  background: #0d2010; }
.insight-limits    { border-left: 4px solid var(--accent-amber);  background: #1f1700; }
.insight-future    { border-left: 4px solid var(--accent-purple); background: #1a0d2e; }
.insight-box {
    border-radius : var(--radius);
    padding       : 1rem 1.2rem;
    margin-bottom : 0.8rem;
    font-size     : 0.95rem;
    line-height   : 1.6;
}

/* ── Confidence badge ── */
.badge {
    display       : inline-block;
    padding       : 0.2rem 0.7rem;
    border-radius : 20px;
    font-size     : 0.78rem;
    font-weight   : 600;
    margin-left   : 0.5rem;
}
.badge-high   { background: #1a3a1a; color: var(--accent-green); border: 1px solid var(--accent-green); }
.badge-medium { background: #2a2000; color: var(--accent-amber); border: 1px solid var(--accent-amber); }
.badge-low    { background: #2a0000; color: #ff7b72;             border: 1px solid #ff7b72; }

/* ── Source chips ── */
.source-chip {
    display       : inline-block;
    background    : #21262d;
    border        : 1px solid var(--border);
    border-radius : 6px;
    padding       : 0.1rem 0.6rem;
    font-size     : 0.75rem;
    color         : var(--text-muted);
    margin        : 0.2rem 0.2rem 0.2rem 0;
    font-family   : 'JetBrains Mono', monospace;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background    : linear-gradient(135deg, #1f6feb, #388bfd);
    color         : white;
    border        : none;
    border-radius : 8px;
    font-weight   : 500;
    padding       : 0.5rem 1.2rem;
    transition    : opacity 0.2s ease, transform 0.1s ease;
}
[data-testid="stButton"] > button:hover {
    opacity   : 0.88;
    transform : translateY(-1px);
}
[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border        : 1.5px dashed var(--border);
    border-radius : var(--radius);
    background    : var(--bg-card);
    transition    : border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-blue); }

/* ── Text input / textarea ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-radius : 8px;
    color         : var(--text-primary);
    font-family   : 'Inter', sans-serif;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: var(--accent-blue); }

/* ── Expander ── */
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: var(--radius); }

/* ── Divider ── */
hr { border-color: var(--border); }

/* ── Metric ── */
[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialisation
# ============================================================
def init_session():
    defaults = {
        "index_built"   : False,
        "chunks"        : [],
        "sources"       : [],
        "rag_pipeline"  : None,
        "summarizer"    : None,
        "retriever"     : None,
        "embedding_gen" : None,
        "vector_db"     : None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ============================================================
# Cached Resource Loaders (load once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_gen():
    return EmbeddingGenerator()

@st.cache_resource(show_spinner="Loading QA model…")
def load_rag_pipeline():
    return RAGPipeline(retriever=None)

@st.cache_resource(show_spinner="Loading summarizer…")
def load_summarizer():
    return Summarizer()


# ============================================================
# Helper: Confidence Badge HTML
# ============================================================
def confidence_badge(score: float) -> str:
    return '<span class="badge badge-high" style="background: #1a1a3a; color: var(--accent-blue); border: 1px solid var(--accent-blue);">🤖 Synthesized</span>'


# ============================================================
# Helper: Format Insight Block
# ============================================================
def render_insight(title: str, content, css_class: str, icon: str):
    if isinstance(content, list):
        items_html = "".join(f"<li>{item}</li>" for item in content if item)
        body = f"<ul>{items_html}</ul>"
    else:
        body = f"<p>{content}</p>"

    st.markdown(f"""
    <div class="insight-box {css_class}">
        <strong>{icon} {title}</strong><br/>
        {body}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 Research Engine")
    st.markdown("*Powered by RAG + HuggingFace*")
    st.divider()

    # ── Upload Section ──
    st.markdown("### 📂 Upload Papers")
    uploaded_files = st.file_uploader(
        label       = "Drop PDF files here",
        type        = ["pdf"],
        accept_multiple_files = True,
        key         = "pdf_uploader",
        help        = "Upload one or more research paper PDFs.",
    )

    # ── Chunking settings ──
    st.divider()
    st.markdown("### ⚙️ Settings")
    chunk_size    = st.slider("Chunk Size (chars)",    200, 1000, 500, 50,
                              help="Size of each text chunk passed to the embedder.")
    chunk_overlap = st.slider("Chunk Overlap (chars)",  0,  200, 100, 10,
                              help="Overlap between consecutive chunks.")
    top_k         = st.slider("Top-K Retrieval",        1,   10,   5,  1,
                              help="Number of context chunks retrieved per query.")

    # ── Process button ──
    st.divider()
    process_btn = st.button("🚀 Process & Index PDFs", use_container_width=True)

    # ── Index stats ──
    if st.session_state.index_built:
        st.divider()
        st.markdown("### 📊 Index Stats")
        vdb   = st.session_state.vector_db
        stats = vdb.get_stats() if vdb else {}
        st.metric("Total Chunks",   stats.get("total_chunks",  0))
        st.metric("Total Vectors",  stats.get("total_vectors", 0))
        sources = stats.get("sources", [])
        st.markdown(f"**Papers indexed:** {len(sources)}")
        for s in sources:
            st.markdown(f'<span class="source-chip">📄 {s}</span>', unsafe_allow_html=True)

    # ── Footer ──
    st.divider()
    st.caption("Phase 1 of Agentic AI Research Scientist System")
    st.caption("Built with ❤️ using Streamlit + LangChain + FAISS")


# ============================================================
# MAIN — Header
# ============================================================
st.markdown("# 🧠 Research Paper Intelligence Engine")
st.markdown(
    "Upload research papers → ask questions → get AI-powered answers, "
    "summaries, and structured insights."
)
st.divider()


# ============================================================
# PDF Processing Logic (triggered by button)
# ============================================================
if process_btn:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF before processing.")
    else:
        with st.status("📥 Processing PDFs…", expanded=True) as status:
            try:
                processor   = PDFProcessor()
                chunker     = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                embed_gen   = load_embedding_gen()
                vdb         = VectorDB()

                all_chunks  = []
                docs        = {}

                # Step 1: Extract text from each PDF
                for i, uf in enumerate(uploaded_files):
                    st.write(f"📄 Extracting: **{uf.name}**")

                    # Step 1: Extract text directly from memory buffer
                    result = processor.process_single(stream=uf.getvalue(), filename=uf.name)
                    docs[uf.name] = result

                # Step 2: Chunk all documents
                st.write("✂️ Chunking text…")
                all_chunks = chunker.chunk_documents(docs)

                # Fix source names (temp file names → original PDF names)
                # Already handled because we pass docs with original keys

                # Step 3: Generate embeddings
                st.write(f"🔢 Generating embeddings for {len(all_chunks)} chunks…")
                embeddings = embed_gen.embed_chunks(all_chunks)

                # Step 4: Build FAISS index
                st.write("🗂️ Building FAISS index…")
                vdb.build_index(embeddings, all_chunks)
                vdb.save()

                # Save to session state
                retriever = Retriever(embedding_gen=embed_gen, vector_db=vdb)
                st.session_state.index_built   = True
                st.session_state.chunks        = all_chunks
                st.session_state.sources       = list(docs.keys())
                st.session_state.vector_db     = vdb
                st.session_state.embedding_gen = embed_gen
                st.session_state.retriever     = retriever

                status.update(
                    label=f"✅ Indexed {len(all_chunks)} chunks from {len(docs)} PDF(s)!",
                    state="complete",
                )
            except Exception as exc:
                status.update(label=f"❌ Error: {exc}", state="error")
                logger.exception(exc)


# ============================================================
# TABS: Q&A | Summarize | Insights | Retrieved Context
# ============================================================
tab_qa, tab_summary, tab_insights, tab_context = st.tabs([
    "💬 Ask a Question",
    "📝 Summarize Paper",
    "🔍 Research Insights",
    "📚 Retrieved Context",
])


# ──────────────────────────────────────────────────────────
# TAB 1: Q&A via RAG
# ──────────────────────────────────────────────────────────
with tab_qa:
    st.markdown("## 💬 Ask Questions About Your Papers")
    st.markdown("*Ask anything — the AI retrieves relevant context and answers from your papers.*")

    if not st.session_state.index_built:
        st.info("👈 Upload and process PDFs first using the sidebar.")
    else:
        question = st.text_input(
            label       = "Your question",
            placeholder = "What is the main contribution of this paper?",
            key         = "qa_question",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            ask_btn = st.button("🔎 Ask", key="ask_btn", use_container_width=True)

        if ask_btn and question:
            try:
                rag = load_rag_pipeline()
                rag.retriever = st.session_state.retriever

                st.markdown(f"### Answer {confidence_badge(1.0)}", unsafe_allow_html=True)
                answer_placeholder = st.empty()
                
                with st.spinner("🤔 Searching papers & thinking…"):
                    result = rag.answer(question, top_k=top_k, stream=True)

                streamer = result["streamer"]
                thread = result["thread"]

                full_answer = ""
                for new_text in streamer:
                    full_answer += new_text
                    # Display with a blinking cursor
                    answer_placeholder.markdown(
                        f'<div class="answer-box">{full_answer} ▌</div>',
                        unsafe_allow_html=True,
                    )
                
                # Remove cursor when done
                answer_placeholder.markdown(
                    f'<div class="answer-box">{full_answer}</div>',
                    unsafe_allow_html=True,
                )
                thread.join()

                # ── Sources ──
                st.markdown("**Sources used:**")
                unique_sources = {s["source"] for s in result["sources"]}
                chips = " ".join(
                    f'<span class="source-chip">📄 {s}</span>'
                    for s in unique_sources
                )
                st.markdown(chips, unsafe_allow_html=True)

                # ── Expandable context ──
                with st.expander("🔎 View retrieved context"):
                    st.code(result["context"][:2000], language=None)

            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                logger.exception(exc)

        elif ask_btn and not question:
            st.warning("Please type a question first.")

        # ── Example questions ──
        st.divider()
        st.markdown("**💡 Example questions:**")
        examples = [
            "What is the main contribution of this paper?",
            "What dataset was used for evaluation?",
            "What are the experimental results?",
            "What deep learning architecture is proposed?",
            "What problem does this paper solve?",
        ]
        
        def set_q(q):
            st.session_state.qa_question = q

        for ex in examples:
            st.button(f"▷ {ex}", key=f"ex_{ex[:20]}", on_click=set_q, args=(ex,))


# ──────────────────────────────────────────────────────────
# TAB 2: Summarization
# ──────────────────────────────────────────────────────────
with tab_summary:
    st.markdown("## 📝 Paper Summarization")
    st.markdown("*Generate a concise abstract-style summary of any indexed paper.*")

    if not st.session_state.index_built:
        st.info("👈 Upload and process PDFs first.")
    else:
        sources = st.session_state.sources
        selected = st.selectbox(
            "Select a paper to summarize",
            options=["All Papers"] + sources,
            key="summary_source",
        )

        sum_btn = st.button("📝 Generate Summary", key="sum_btn")

        if sum_btn:
            with st.spinner("🤔 Generating summary with BART…"):
                try:
                    summ = load_summarizer()
                    chunks = st.session_state.chunks

                    if selected == "All Papers":
                        full_text = "\n\n".join(c["text"] for c in chunks)
                        src_label = "All Papers"
                    else:
                        full_text = "\n\n".join(
                            c["text"] for c in chunks if c.get("source") == selected
                        )
                        src_label = selected

                    summary = summ.summarize(full_text)

                    st.markdown(f"### Summary: *{src_label}*")
                    st.markdown(
                        f'<div class="rag-card">{summary}</div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "⬇️ Download Summary",
                        data=summary,
                        file_name=f"summary_{src_label.replace(' ','_')}.txt",
                        mime="text/plain",
                        key="download_summary",
                    )
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")
                    logger.exception(exc)


# ──────────────────────────────────────────────────────────
# TAB 3: Research Insights
# ──────────────────────────────────────────────────────────
with tab_insights:
    st.markdown("## 🔍 Research Insights Extraction")
    st.markdown(
        "*Automatically extract **Key Findings**, **Limitations**, and **Future Work** "
        "from your research paper.*"
    )

    if not st.session_state.index_built:
        st.info("👈 Upload and process PDFs first.")
    else:
        sources  = st.session_state.sources
        selected = st.selectbox(
            "Select a paper",
            options=sources,
            key="insights_source",
        )

        ins_btn = st.button("🔍 Extract Insights", key="ins_btn")

        if ins_btn:
            with st.spinner("🕵️ Extracting structured insights…"):
                try:
                    summ   = load_summarizer()
                    chunks = st.session_state.chunks
                    result = summ.full_analysis(chunks, source=selected)

                    # ── Summary card ──
                    st.markdown(f"### 📄 {selected}")
                    st.markdown(
                        f'<div class="rag-card"><strong>Summary:</strong> {result["summary"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Insight cards ──
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        render_insight(
                            "Key Findings",
                            result["key_findings"],
                            "insight-box insight-findings",
                            "🟢",
                        )
                    with col2:
                        render_insight(
                            "Limitations",
                            result["limitations"],
                            "insight-box insight-limits",
                            "🟡",
                        )
                    with col3:
                        render_insight(
                            "Future Work",
                            result["future_work"],
                            "insight-box insight-future",
                            "🔵",
                        )

                    # ── Download ──
                    import json
                    report = json.dumps(result, indent=2, ensure_ascii=False)
                    st.download_button(
                        "⬇️ Download Insights (JSON)",
                        data=report,
                        file_name=f"insights_{selected.replace(' ','_')}.json",
                        mime="application/json",
                        key="download_insights",
                    )

                except Exception as exc:
                    st.error(f"❌ Error: {exc}")
                    logger.exception(exc)


# ──────────────────────────────────────────────────────────
# TAB 4: Retrieved Context Explorer
# ──────────────────────────────────────────────────────────
with tab_context:
    st.markdown("## 📚 Semantic Search Explorer")
    st.markdown(
        "*Search the indexed chunks directly to see exactly what chunks are retrieved "
        "for any query — useful for debugging and understanding the pipeline.*"
    )

    if not st.session_state.index_built:
        st.info("👈 Upload and process PDFs first.")
    else:
        search_query = st.text_input(
            "Search query",
            placeholder="attention mechanism transformer",
            key="ctx_query",
        )
        search_btn = st.button("🔎 Search Chunks", key="ctx_btn")

        if search_btn and search_query:
            with st.spinner("Searching…"):
                try:
                    retriever = st.session_state.retriever
                    results   = retriever.retrieve(search_query, top_k=top_k)

                    st.markdown(f"**Found {len(results)} chunks:**")
                    for r in results:
                        with st.expander(
                            f"Rank {r['rank']} | Score: {r['score']:.4f} | 📄 {r['source']}"
                        ):
                            st.text(r["text"])

                except Exception as exc:
                    st.error(f"❌ Error: {exc}")
