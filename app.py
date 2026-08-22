# ============================================================
# app.py
# Phase 9 — Streamlit UI (Premium Animated Edition)
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
import time

# Move HuggingFace cache to E: drive immediately because C: drive is full
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")

# Suppress harmless torchvision/transformers warnings from Streamlit file watcher
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
from src.agent         import ResearchAgent

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
# Custom CSS — Premium Animated Dark Theme
# ============================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary    : #0a0a1a;
    --bg-secondary  : #0f0f23;
    --bg-card       : rgba(20, 20, 50, 0.6);
    --bg-glass      : rgba(255, 255, 255, 0.03);
    --accent-cyan   : #00d4ff;
    --accent-purple : #a855f7;
    --accent-green  : #22c55e;
    --accent-amber  : #f59e0b;
    --accent-rose   : #f43f5e;
    --accent-blue   : #3b82f6;
    --text-primary  : #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted    : #64748b;
    --border-glass  : rgba(255, 255, 255, 0.08);
    --border-glow   : rgba(0, 212, 255, 0.3);
    --radius        : 16px;
    --radius-sm     : 10px;
    --shadow-glow   : 0 0 30px rgba(0, 212, 255, 0.1);
    --transition    : all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Keyframe Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 10px rgba(0, 212, 255, 0.2); }
    50%      { box-shadow: 0 0 25px rgba(0, 212, 255, 0.5); }
}

@keyframes borderGlow {
    0%, 100% { border-color: rgba(0, 212, 255, 0.2); }
    50%      { border-color: rgba(0, 212, 255, 0.6); }
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-8px); }
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0; }
}

@keyframes spinSlow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

@keyframes progressStripe {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 0; }
}

@keyframes nodeActivePulse {
    0%, 100% { transform: scale(1); box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
    50%      { transform: scale(1.08); box-shadow: 0 0 25px rgba(0, 212, 255, 0.7); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}

@keyframes typewriter {
    from { width: 0; }
    to   { width: 100%; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-purple); }

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"] > section > div {
    background: var(--bg-primary) !important;
}

[data-testid="stHeader"] {
    background: rgba(10, 10, 26, 0.8) !important;
    backdrop-filter: blur(12px);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d24 0%, #0a0a1a 100%) !important;
    border-right: 1px solid var(--border-glass);
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-green));
    background-size: 200% 100%;
    animation: gradientShift 4s ease infinite;
    z-index: 999;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.35rem !important; color: var(--accent-cyan) !important; }
h3 { font-size: 1.15rem !important; color: var(--accent-purple) !important; }

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    animation: fadeInUp 0.5s ease-out both;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.glass-card:hover::before {
    opacity: 1;
}

/* Staggered animation delays */
.glass-card:nth-child(1) { animation-delay: 0.05s; }
.glass-card:nth-child(2) { animation-delay: 0.1s; }
.glass-card:nth-child(3) { animation-delay: 0.15s; }
.glass-card:nth-child(4) { animation-delay: 0.2s; }
.glass-card:nth-child(5) { animation-delay: 0.25s; }

/* ── Animated Header ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    animation: fadeInDown 0.6s ease-out;
}
.app-header h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-green), var(--accent-cyan));
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 5s ease infinite;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.6;
}
.header-decoration {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 1rem;
}
.header-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    animation: float 3s ease-in-out infinite;
}
.header-dot:nth-child(1) { background: var(--accent-cyan); animation-delay: 0s; }
.header-dot:nth-child(2) { background: var(--accent-purple); animation-delay: 0.3s; }
.header-dot:nth-child(3) { background: var(--accent-green); animation-delay: 0.6s; }
.header-dot:nth-child(4) { background: var(--accent-amber); animation-delay: 0.9s; }
.header-dot:nth-child(5) { background: var(--accent-rose); animation-delay: 1.2s; }

/* ── Pipeline Progress ── */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 2rem 1rem;
    margin: 1.5rem 0;
    animation: fadeInUp 0.5s ease-out;
}
.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    position: relative;
    z-index: 2;
}
.pipeline-icon {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    border: 2px solid var(--border-glass);
    background: var(--bg-secondary);
    transition: var(--transition);
    position: relative;
}
.pipeline-icon.pending {
    opacity: 0.4;
    border-color: var(--text-muted);
}
.pipeline-icon.active {
    border-color: var(--accent-cyan);
    background: rgba(0, 212, 255, 0.1);
    animation: nodeActivePulse 2s ease-in-out infinite;
}
.pipeline-icon.done {
    border-color: var(--accent-green);
    background: rgba(34, 197, 94, 0.15);
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.3);
}
.pipeline-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    transition: var(--transition);
    white-space: nowrap;
}
.pipeline-label.active { color: var(--accent-cyan); }
.pipeline-label.done   { color: var(--accent-green); }
.pipeline-connector {
    width: 60px;
    height: 3px;
    background: var(--border-glass);
    position: relative;
    margin: 0 -4px;
    margin-bottom: 22px;
    z-index: 1;
    border-radius: 2px;
    overflow: hidden;
}
.pipeline-connector.done {
    background: var(--accent-green);
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
}
.pipeline-connector.active {
    background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan));
}
.pipeline-connector.active::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.8), transparent);
    animation: shimmer 1.5s ease-in-out infinite;
}

/* ── Result Cards ── */
.result-paper {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-sm);
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    animation: fadeInUp 0.4s ease-out both;
    transition: var(--transition);
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.result-paper:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
    transform: translateY(-1px);
}
.paper-rank {
    min-width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #d97706); color: #1a1a2e; }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: #1a1a2e; }
.rank-3 { background: linear-gradient(135deg, #a16207, #92400e); color: #fef3c7; }
.rank-default { background: rgba(100, 116, 139, 0.2); color: var(--text-secondary); border: 1px solid var(--border-glass); }
.paper-info { flex: 1; }
.paper-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.paper-meta {
    font-size: 0.8rem;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
}
.paper-score-bar {
    width: 80px;
    height: 5px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 3px;
    overflow: hidden;
}
.paper-score-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
    transition: width 0.8s ease-out;
}

/* ── Answer Box ── */
.answer-box {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    line-height: 1.75;
    margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease-out;
    color: var(--text-primary);
}
.answer-box.streaming {
    animation: borderGlow 2s ease-in-out infinite;
}
.cursor-blink {
    animation: blink 1s step-end infinite;
    color: var(--accent-cyan);
    font-weight: 700;
}

/* ── Insight Cards ── */
.insight-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-sm);
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    animation: fadeInUp 0.4s ease-out both;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.insight-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}
.insight-card.findings {
    border-left: 4px solid var(--accent-green);
}
.insight-card.findings:hover { box-shadow: 0 0 25px rgba(34, 197, 94, 0.15); }
.insight-card.limits {
    border-left: 4px solid var(--accent-amber);
}
.insight-card.limits:hover { box-shadow: 0 0 25px rgba(245, 158, 11, 0.15); }
.insight-card.future {
    border-left: 4px solid var(--accent-purple);
}
.insight-card.future:hover { box-shadow: 0 0 25px rgba(168, 85, 247, 0.15); }
.insight-card h4 {
    font-size: 0.95rem !important;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.insight-card ul {
    margin: 0;
    padding-left: 1.2rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.7;
}
.insight-card li { margin-bottom: 0.3rem; }

/* ── Confidence Badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-synth {
    background: rgba(0, 212, 255, 0.1);
    color: var(--accent-cyan);
    border: 1px solid rgba(0, 212, 255, 0.3);
}

/* ── Source Chips ── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(168, 85, 247, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    color: var(--accent-purple);
    margin: 0.2rem 0.2rem 0.2rem 0;
    font-family: 'JetBrains Mono', monospace;
    transition: var(--transition);
}
.source-chip:hover {
    background: rgba(168, 85, 247, 0.2);
    border-color: rgba(168, 85, 247, 0.5);
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.2);
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(168, 85, 247, 0.15)) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: var(--transition) !important;
    letter-spacing: 0.01em;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.25), rgba(168, 85, 247, 0.25)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-1px);
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px);
}

/* ── Primary Action Button ── */
.stButton > button[kind="primary"],
.action-btn > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    border: none !important;
    color: #0a0a1a !important;
    font-weight: 700 !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0, 212, 255, 0.2) !important;
    border-radius: var(--radius-sm) !important;
    background: rgba(0, 212, 255, 0.03) !important;
    transition: var(--transition);
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 212, 255, 0.4) !important;
    background: rgba(0, 212, 255, 0.06) !important;
}

/* ── Text Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(15, 15, 35, 0.8) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: var(--transition);
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.15) !important;
    outline: none !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important;
    color: var(--text-muted) !important;
    transition: var(--transition);
    font-size: 0.88rem !important;
}
[data-testid="stTabs"] button:hover {
    color: var(--text-primary) !important;
    background: rgba(0, 212, 255, 0.05) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: rgba(0, 212, 255, 0.08) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px);
    transition: var(--transition);
}
[data-testid="stExpander"]:hover {
    border-color: rgba(0, 212, 255, 0.2) !important;
}

/* ── Divider ── */
hr { border-color: var(--border-glass) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
    transition: var(--transition);
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.1) !important;
}

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(15, 15, 35, 0.8) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    color: var(--accent-cyan) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
    color: var(--accent-green) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(34, 197, 94, 0.2) !important;
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.2) !important;
}

/* ── Literature Review Glass Container ── */
.lit-review {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 2rem;
    animation: fadeInUp 0.5s ease-out;
    line-height: 1.8;
    font-size: 0.95rem;
}
.lit-review h2 { color: var(--accent-cyan) !important; margin-bottom: 1rem; }
.lit-review h3 { color: var(--accent-purple) !important; margin-top: 1.5rem; }
.lit-review li { margin-bottom: 0.4rem; color: var(--text-secondary); }
.lit-review strong { color: var(--text-primary); }
.lit-review hr { border-color: var(--border-glass); margin: 1.5rem 0; }

/* ── Comparison Table ── */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border-glass);
    font-size: 0.88rem;
}
.comparison-table th {
    background: rgba(0, 212, 255, 0.08);
    color: var(--accent-cyan);
    font-weight: 600;
    padding: 0.8rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-glass);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.comparison-table td {
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--border-glass);
    color: var(--text-secondary);
    vertical-align: top;
}
.comparison-table tr:hover td {
    background: rgba(0, 212, 255, 0.03);
}
.comparison-table tr:last-child td {
    border-bottom: none;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Workflow Stage Status Text ── */
.stage-status {
    font-size: 0.85rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0;
    animation: slideInLeft 0.3s ease-out;
}
.stage-status .done { color: var(--accent-green); }
.stage-status .active { color: var(--accent-cyan); }

/* ── Sidebar Logo ── */
.sidebar-logo {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
}

/* ── Example question buttons ── */
.example-btn button {
    text-align: left !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    padding: 0.4rem 0.8rem !important;
}
.example-btn button:hover {
    color: var(--accent-cyan) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
    background: rgba(0, 212, 255, 0.05) !important;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] {
    color: var(--accent-cyan) !important;
}

/* ── Info/Warning/Success boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    backdrop-filter: blur(8px);
}

/* ── Background particle dots (decorative) ── */
.bg-particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.particle {
    position: absolute;
    width: 3px;
    height: 3px;
    border-radius: 50%;
    opacity: 0.15;
    animation: float 8s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Background Particle Decoration
# ============================================================
st.markdown("""
<div class="bg-particles">
    <div class="particle" style="background: var(--accent-cyan); top: 15%; left: 10%; animation-delay: 0s; animation-duration: 9s;"></div>
    <div class="particle" style="background: var(--accent-purple); top: 30%; left: 85%; animation-delay: 2s; animation-duration: 7s;"></div>
    <div class="particle" style="background: var(--accent-green); top: 60%; left: 20%; animation-delay: 4s; animation-duration: 11s;"></div>
    <div class="particle" style="background: var(--accent-amber); top: 80%; left: 70%; animation-delay: 1s; animation-duration: 8s;"></div>
    <div class="particle" style="background: var(--accent-rose); top: 45%; left: 50%; animation-delay: 3s; animation-duration: 10s;"></div>
    <div class="particle" style="background: var(--accent-cyan); top: 70%; left: 40%; animation-delay: 5s; animation-duration: 12s;"></div>
    <div class="particle" style="background: var(--accent-purple); top: 20%; left: 60%; animation-delay: 6s; animation-duration: 9s;"></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialisation
# ============================================================
def init_session():
    defaults = {
        "index_built"        : False,
        "chunks"             : [],
        "sources"            : [],
        "rag_pipeline"       : None,
        "summarizer"         : None,
        "retriever"          : None,
        "embedding_gen"      : None,
        "vector_db"          : None,
        "qa_history"         : [],       # Chat history for Q&A tab
        "bookmarked_papers"  : [],       # Bookmarked papers from agentic workflow
        "last_workflow_state": None,     # Last agentic workflow result for PDF export
        "last_workflow_topic": "",       # Last agentic workflow topic
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
    return '<span class="badge badge-synth">🤖 Synthesized</span>'


# ============================================================
# Helper: Render Pipeline Progress HTML
# ============================================================
PIPELINE_STAGES = [
    ("📋", "Plan"),
    ("🔍", "Search"),
    ("📊", "Rank"),
    ("🧠", "Analyze"),
    ("⚖️", "Compare"),
    ("📝", "Review"),
]

def render_pipeline(current_stage_idx: int, total_stages: int = 6) -> str:
    """
    Generates the HTML for the pipeline progress indicator.
    current_stage_idx: 0-based index of the currently active stage.
                       -1 = all pending, total_stages = all done.
    """
    html = '<div class="pipeline-container">'
    for i, (icon, label) in enumerate(PIPELINE_STAGES):
        if i < current_stage_idx:
            icon_class = "done"
            label_class = "done"
            display_icon = "✓"
        elif i == current_stage_idx:
            icon_class = "active"
            label_class = "active"
            display_icon = icon
        else:
            icon_class = "pending"
            label_class = ""
            display_icon = icon

        html += f'''
        <div class="pipeline-node">
            <div class="pipeline-icon {icon_class}">{display_icon}</div>
            <div class="pipeline-label {label_class}">{label}</div>
        </div>
        '''
        # Connector (not after the last node)
        if i < total_stages - 1:
            if i < current_stage_idx:
                conn_class = "done"
            elif i == current_stage_idx:
                conn_class = "active"
            else:
                conn_class = ""
            html += f'<div class="pipeline-connector {conn_class}"></div>'

    html += '</div>'
    return html


# ============================================================
# Helper: Format Insight Block (Glass Card)
# ============================================================
def render_insight(title: str, content, css_class: str, icon: str):
    if isinstance(content, list):
        items_html = "".join(f"<li>{item}</li>" for item in content if item)
        body = f"<ul>{items_html}</ul>"
    else:
        body = f"<p style='color: var(--text-secondary); margin: 0;'>{content}</p>"

    st.markdown(f"""
    <div class="insight-card {css_class}">
        <h4>{icon} {title}</h4>
        {body}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Helper: Render Paper Result Card
# ============================================================
def render_paper_card(paper: dict, rank: int) -> str:
    title = paper.get('title', 'Unknown Title')
    year = paper.get('year', 'N/A')
    score = paper.get('relevance_score', 0)
    score_pct = min(score * 100, 100)

    rank_class = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-default")

    return f"""
    <div class="result-paper" style="animation-delay: {rank * 0.08}s;">
        <div class="paper-rank {rank_class}">{rank}</div>
        <div class="paper-info">
            <div class="paper-title">{title}</div>
            <div class="paper-meta">
                <span>📅 {year}</span>
                <span>Score: {score:.3f}</span>
                <div class="paper-score-bar">
                    <div class="paper-score-fill" style="width: {score_pct}%;"></div>
                </div>
            </div>
        </div>
    </div>
    """


# ============================================================
# Helper: Build Comparison Table HTML
# ============================================================
def build_comparison_html(analyses: list) -> str:
    if not analyses:
        return "<p>No papers to compare.</p>"

    html = '<table class="comparison-table">'
    html += """<thead><tr>
        <th>Paper</th>
        <th>Proposed Method</th>
        <th>Dataset</th>
        <th>Key Findings</th>
        <th>Limitations</th>
    </tr></thead><tbody>"""

    for p in analyses:
        title = p['title']
        method = str(p.get('proposed_method', '')).replace('\n', ' ')[:180]
        dataset = str(p.get('dataset', '')).replace('\n', ' ')[:120]

        findings = p.get('key_findings', '')
        if isinstance(findings, list):
            findings = findings[0] if findings else "—"
        findings = str(findings).replace('\n', ' ')[:180]

        limitations = p.get('limitations', '')
        if isinstance(limitations, list):
            limitations = limitations[0] if limitations else "—"
        limitations = str(limitations).replace('\n', ' ')[:120]

        html += f"""<tr>
            <td><strong>{title}</strong></td>
            <td>{method}</td>
            <td>{dataset}</td>
            <td>{findings}</td>
            <td>{limitations}</td>
        </tr>"""

    html += "</tbody></table>"
    return html


# ============================================================
# Helper: Safe markdown-to-HTML (for literature review)
# ============================================================
def _md_to_safe_html(md_text: str) -> str:
    """
    Converts basic markdown to HTML for display in the lit-review div.
    Handles headers, bold, lists, horizontal rules.
    """
    import re

    lines = md_text.split('\n')
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Horizontal rule
        if stripped.startswith('---'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<hr/>')
            continue

        # Headers
        if stripped.startswith('### '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3>{stripped[4:]}</h3>')
            continue
        if stripped.startswith('## '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h2>{stripped[3:]}</h2>')
            continue

        # List items
        if stripped.startswith('- '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            content = stripped[2:]
            # Handle bold
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            html_lines.append(f'<li>{content}</li>')
            continue

        # Close list if we're in one
        if in_list and not stripped.startswith('- '):
            html_lines.append('</ul>')
            in_list = False

        # Regular paragraph
        if stripped:
            # Handle bold
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
            # Handle italic
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            html_lines.append(f'<p>{content}</p>')

    if in_list:
        html_lines.append('</ul>')

    return '\n'.join(html_lines)


# ============================================================
# SIDEBAR — Premium Navigation
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 Research Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Powered by RAG + HuggingFace</div>', unsafe_allow_html=True)
    st.divider()

    # ── Upload Section ──
    st.markdown('<div class="sidebar-section">📂 Upload Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label       = "Drop PDF files here",
        type        = ["pdf"],
        accept_multiple_files = True,
        key         = "pdf_uploader",
        help        = "Upload one or more research paper PDFs.",
    )

    # ── Chunking settings ──
    st.divider()
    st.markdown('<div class="sidebar-section">⚙️ Settings</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sidebar-section">📊 Index Stats</div>', unsafe_allow_html=True)
        vdb   = st.session_state.vector_db
        stats = vdb.get_stats() if vdb else {}
        st.metric("Total Chunks",   stats.get("total_chunks",  0))
        st.metric("Total Vectors",  stats.get("total_vectors", 0))
        sources = stats.get("sources", [])
        st.markdown(f"**Papers indexed:** {len(sources)}")
        for s in sources:
            st.markdown(f'<span class="source-chip">📄 {s}</span>', unsafe_allow_html=True)

    # ── Bookmarked Papers ──
    if st.session_state.bookmarked_papers:
        st.divider()
        st.markdown('<div class="sidebar-section">⭐ Bookmarked Papers</div>', unsafe_allow_html=True)
        for bm in st.session_state.bookmarked_papers:
            st.markdown(f"""
            <div class="glass-card" style="padding: 0.6rem; margin-bottom: 0.4rem; font-size: 0.8rem;">
                <strong>{bm.get('title', 'Unknown')[:50]}</strong><br/>
                <span style="color: var(--text-secondary);">{bm.get('year', 'N/A')} · Score: {bm.get('relevance_score', 0):.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.divider()
    st.caption("Agentic AI Research Scientist System")
    st.caption("Built with ❤️ using Streamlit + LangChain + FAISS")


# ============================================================
# MAIN — Animated Header
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>🧠 Research Paper Intelligence Engine</h1>
    <div class="app-subtitle">
        Upload research papers → ask questions → get AI-powered answers, summaries, and structured insights.
    </div>
    <div class="header-decoration">
        <div class="header-dot"></div>
        <div class="header-dot"></div>
        <div class="header-dot"></div>
        <div class="header-dot"></div>
        <div class="header-dot"></div>
    </div>
</div>
""", unsafe_allow_html=True)


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

                    # Extract text directly from memory buffer
                    result = processor.process_single(stream=uf.getvalue(), filename=uf.name)
                    docs[uf.name] = result

                # Step 2: Chunk all documents
                st.write("✂️ Chunking text…")
                all_chunks = chunker.chunk_documents(docs)

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
# TABS: Agentic | Q&A | Summarize | Insights | Context
# ============================================================
tab_agent, tab_qa, tab_summary, tab_insights, tab_context = st.tabs([
    "🤖 Agentic Workflow",
    "💬 Ask a Question",
    "📝 Summarize Paper",
    "🔍 Research Insights",
    "📚 Retrieved Context",
])

# ──────────────────────────────────────────────────────────
# TAB 0: Agentic Workflow — with Pipeline Progress
# ──────────────────────────────────────────────────────────
with tab_agent:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 1.2rem;">
        <h2 style="margin:0;">🤖 Agentic AI Research Assistant</h2>
        <p style="color: var(--text-secondary); margin: 0.3rem 0 0;">
            End-to-end automated research workflow: Discover → Rank → Analyze → Compare → Review
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_topic, col_limit = st.columns([3, 1])
    with col_topic:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Retrieval-Augmented Generation in Healthcare",
            key="agent_topic"
        )
    with col_limit:
        num_papers = st.slider("Number of Papers", 1, 10, 5, key="num_papers_slider")

    agent_btn = st.button("🚀 Run Full Research Workflow", use_container_width=True)

    if agent_btn and topic:
        if not topic.strip():
            st.error("⚠️ Please enter a valid research topic before starting.")
        else:
            # Show initial pipeline (all pending)
            pipeline_placeholder = st.empty()
            pipeline_placeholder.markdown(render_pipeline(-1), unsafe_allow_html=True)

            stage_status = st.empty()

            try:
                from src.agent import (
                    plan_research_node, search_papers_node, rank_papers_node,
                    analyze_papers_node, compare_papers_node,
                    review_papers_node, ResearchState
                )

                state = ResearchState(
                    research_topic=topic,
                    topic=topic,
                    research_questions=[],
                    discovered_papers=[],
                    raw_papers=[],
                    ranked_papers=[],
                    selected_papers=[],
                    paper_analysis=[],
                    analyses=[],
                    comparison="",
                    literature_review="",
                    status="Starting...",
                    errors=[]
                )

                # ── Stage 0: Plan ──
                pipeline_placeholder.markdown(render_pipeline(0), unsafe_allow_html=True)
                stage_status.markdown(
                    '<div class="stage-status"><span class="active">⟳</span> Formulating research plan & scope…</div>',
                    unsafe_allow_html=True
                )
                state = plan_research_node(state)

                # ── Stage 1: Search ──
                pipeline_placeholder.markdown(render_pipeline(1), unsafe_allow_html=True)
                stage_status.markdown(
                    '<div class="stage-status"><span class="active">⟳</span> Searching arXiv for papers on this topic…</div>',
                    unsafe_allow_html=True
                )
                state = search_papers_node(state)
                num_found = len(state.get("discovered_papers", []))

                if num_found == 0:
                    st.warning(f"⚠️ No papers found on arXiv for query '{topic}'. Try broadening your search terms.")
                else:
                    # ── Stage 2: Rank ──
                    pipeline_placeholder.markdown(render_pipeline(2), unsafe_allow_html=True)
                    stage_status.markdown(
                        f'<div class="stage-status"><span class="done">✓</span> Discovered {num_found} papers — <span class="active">⟳</span> Ranking top-{num_papers} by semantic relevance…</div>',
                        unsafe_allow_html=True
                    )
                    # Override Top-K selection with UI slider
                    from src.ranker import SemanticRanker
                    ranker = SemanticRanker(top_k=num_papers)
                    state["ranked_papers"] = ranker.rank_papers(topic, state.get("discovered_papers", []))
                    state["selected_papers"] = state["ranked_papers"]
                    num_ranked = len(state.get("selected_papers", []))

                    # ── Stage 3: Analyze ──
                    pipeline_placeholder.markdown(render_pipeline(3), unsafe_allow_html=True)
                    stage_status.markdown(
                        f'<div class="stage-status"><span class="done">✓</span> Selected top {num_ranked} papers — <span class="active">⟳</span> Ingesting into VT RAG engine & extracting insights…</div>',
                        unsafe_allow_html=True
                    )
                    state = analyze_papers_node(state)

                    # ── Stage 4: Compare ──
                    pipeline_placeholder.markdown(render_pipeline(4), unsafe_allow_html=True)
                    stage_status.markdown(
                        '<div class="stage-status"><span class="done">✓</span> Analysis complete — <span class="active">⟳</span> Generating comparative matrix…</div>',
                        unsafe_allow_html=True
                    )
                    state = compare_papers_node(state)

                    # ── Stage 5: Review ──
                    pipeline_placeholder.markdown(render_pipeline(5), unsafe_allow_html=True)
                    stage_status.markdown(
                        '<div class="stage-status"><span class="done">✓</span> Matrix ready — <span class="active">⟳</span> Synthesizing literature review…</div>',
                        unsafe_allow_html=True
                    )
                    state = review_papers_node(state)

                    # ── All Done ──
                    pipeline_placeholder.markdown(render_pipeline(6), unsafe_allow_html=True)
                    stage_status.markdown(
                        '<div class="stage-status"><span class="done">✓</span> Research Workflow complete! All 6 stages finished successfully.</div>',
                        unsafe_allow_html=True
                    )

            # ════════════════════════════════════════════════
            # Display Results
            # ════════════════════════════════════════════════

            st.markdown("---")

            # ── Top Ranked Papers ──
            st.markdown("### 🏆 Top Ranked Papers")
            papers_html = ""
            for i, p in enumerate(state.get("selected_papers", []), start=1):
                papers_html += render_paper_card(p, i)
            st.markdown(papers_html, unsafe_allow_html=True)

            st.markdown("---")

            # ── Structured Analysis ──
            st.markdown("### 📊 Structured Analysis")
            for idx, p_analysis in enumerate(state.get("analyses", [])):
                with st.expander(f"📄 {p_analysis['title']}", expanded=(idx == 0)):
                    # Display as a nicely formatted glass card
                    fields = [
                        ("🎯 Research Problem", p_analysis.get("research_problem", "")),
                        ("🎯 Objective", p_analysis.get("objective", "")),
                        ("🔬 Proposed Method", p_analysis.get("proposed_method", "")),
                        ("📊 Dataset", p_analysis.get("dataset", "")),
                        ("📏 Evaluation Metrics", p_analysis.get("evaluation_metrics", "")),
                        ("📈 Results", p_analysis.get("experimental_results", "")),
                    ]
                    for label, value in fields:
                        st.markdown(f"**{label}**")
                        st.markdown(f"> {value}")

                    # Key Findings, Limitations, Future Work
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        render_insight("Key Findings", p_analysis.get("key_findings", "—"), "findings", "🟢")
                    with col2:
                        render_insight("Limitations", p_analysis.get("limitations", "—"), "limits", "🟡")
                    with col3:
                        render_insight("Future Work", p_analysis.get("future_work", "—"), "future", "🔵")

            st.markdown("---")

            # ── Multi-Paper Comparison ──
            st.markdown("### ⚖️ Multi-Paper Comparison")
            comparison_html = build_comparison_html(state.get("analyses", []))
            st.markdown(f'<div class="glass-card" style="padding: 0; overflow-x: auto;">{comparison_html}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── Literature Review ──
            lit_review = state.get("literature_review", "")
            if lit_review:
                st.markdown(f'<div class="lit-review">{_md_to_safe_html(lit_review)}</div>', unsafe_allow_html=True)

            # ── Save state for export ──
            st.session_state.last_workflow_state = state
            st.session_state.last_workflow_topic = topic

            st.markdown("---")

            # ── Action buttons row ──
            action_col1, action_col2, action_col3 = st.columns([1, 1, 2])

            with action_col1:
                # PDF Export
                try:
                    from src.export import ResearchReportPDF
                    exporter = ResearchReportPDF()
                    pdf_bytes = exporter.generate(topic, state)
                    st.download_button(
                        label="📥 Download Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"research_report_{topic.replace(' ', '_')[:30]}.pdf",
                        mime="application/pdf",
                        key="download_pdf_report",
                        use_container_width=True,
                    )
                except ImportError:
                    st.warning("PDF export requires fpdf2. Install: pip install fpdf2")
                except Exception as pdf_exc:
                    st.error(f"PDF export failed: {pdf_exc}")

            with action_col2:
                # Bookmark toggle for all papers
                if st.button("⭐ Bookmark All Papers", key="bookmark_all", use_container_width=True):
                    existing_ids = {p.get('id') for p in st.session_state.bookmarked_papers}
                    for p in state.get("selected_papers", []):
                        if p.get('id') not in existing_ids:
                            st.session_state.bookmarked_papers.append(p)
                    st.toast(f"⭐ Bookmarked {len(state.get('selected_papers', []))} papers!")
                    st.rerun()

        except Exception as exc:
            pipeline_placeholder.markdown(render_pipeline(-1), unsafe_allow_html=True)
            stage_status.empty()
            st.error(f"❌ Workflow Error: {exc}")
            logger.exception(exc)

    elif agent_btn and not topic:
        st.warning("Please enter a research topic first.")

    # ── Re-export from last run ──
    elif st.session_state.last_workflow_state is not None:
        st.markdown("---")
        st.info("Previous workflow results are available. Run a new workflow or download the last report.")
        try:
            from src.export import ResearchReportPDF
            exporter = ResearchReportPDF()
            pdf_bytes = exporter.generate(
                st.session_state.last_workflow_topic,
                st.session_state.last_workflow_state,
            )
            st.download_button(
                label="📥 Download Last Report (PDF)",
                data=pdf_bytes,
                file_name=f"research_report_{st.session_state.last_workflow_topic.replace(' ', '_')[:30]}.pdf",
                mime="application/pdf",
                key="download_pdf_last",
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────
# TAB 1: Q&A via RAG
# ──────────────────────────────────────────────────────────
with tab_qa:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 1.2rem;">
        <h2 style="margin:0;">💬 Ask Questions About Your Papers</h2>
        <p style="color: var(--text-secondary); margin: 0.3rem 0 0;">
            Ask anything — the AI retrieves relevant context and answers from your papers.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

                st.markdown(f'### Answer {confidence_badge(1.0)}', unsafe_allow_html=True)
                answer_placeholder = st.empty()

                with st.spinner("🤔 Searching papers & thinking…"):
                    result = rag.answer(question, top_k=top_k, stream=True)

                streamer = result["streamer"]
                thread = result["thread"]

                full_answer = ""
                for new_text in streamer:
                    full_answer += new_text
                    # Display with animated cursor
                    answer_placeholder.markdown(
                        f'<div class="answer-box streaming">{full_answer}<span class="cursor-blink">▌</span></div>',
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

                # ── Save to chat history ──
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": full_answer,
                    "sources": list(unique_sources),
                })

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

        # ── Chat History ──
        if st.session_state.qa_history:
            st.divider()
            col_hist, col_clear = st.columns([4, 1])
            with col_hist:
                st.markdown("### 💬 Chat History")
            with col_clear:
                if st.button("🗑️ Clear", key="clear_history"):
                    st.session_state.qa_history = []
                    st.rerun()

            for i, entry in enumerate(reversed(st.session_state.qa_history)):
                idx = len(st.session_state.qa_history) - i
                st.markdown(f"""
                <div class="glass-card" style="padding: 0.8rem; margin-bottom: 0.5rem;">
                    <div style="color: var(--accent-cyan); font-weight: 600; margin-bottom: 0.3rem;">❓ Q{idx}: {entry['question']}</div>
                    <div style="color: var(--text-primary); margin-bottom: 0.3rem;">{entry['answer']}</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">Sources: {', '.join(entry.get('sources', []))}</div>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# TAB 2: Summarization
# ──────────────────────────────────────────────────────────
with tab_summary:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 1.2rem;">
        <h2 style="margin:0;">📝 Paper Summarization</h2>
        <p style="color: var(--text-secondary); margin: 0.3rem 0 0;">
            Generate a concise abstract-style summary of any indexed paper.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
                        f'<div class="glass-card"><p style="line-height:1.8; color: var(--text-secondary);">{summary}</p></div>',
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
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 1.2rem;">
        <h2 style="margin:0;">🔍 Research Insights Extraction</h2>
        <p style="color: var(--text-secondary); margin: 0.3rem 0 0;">
            Automatically extract <strong>Key Findings</strong>, <strong>Limitations</strong>, and <strong>Future Work</strong> from your research paper.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
                        f'<div class="glass-card"><strong>Summary:</strong><p style="color: var(--text-secondary); line-height: 1.7;">{result["summary"]}</p></div>',
                        unsafe_allow_html=True,
                    )

                    # ── Insight cards ──
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        render_insight(
                            "Key Findings",
                            result["key_findings"],
                            "findings",
                            "🟢",
                        )
                    with col2:
                        render_insight(
                            "Limitations",
                            result["limitations"],
                            "limits",
                            "🟡",
                        )
                    with col3:
                        render_insight(
                            "Future Work",
                            result["future_work"],
                            "future",
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
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 1.2rem;">
        <h2 style="margin:0;">📚 Semantic Search Explorer</h2>
        <p style="color: var(--text-secondary); margin: 0.3rem 0 0;">
            Search the indexed chunks directly to see what's retrieved for any query — useful for debugging and understanding the pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
