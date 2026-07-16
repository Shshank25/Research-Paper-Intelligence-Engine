# ============================================================
# config.py — Central Configuration
# All tunable parameters live here so the rest of the code
# never needs magic numbers.
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR   = os.path.join(BASE_DIR, "documents")
EMBEDDINGS_DIR  = os.path.join(BASE_DIR, "embeddings")
VECTOR_STORE_DIR= os.path.join(BASE_DIR, "vector_store")

# Create directories if they don't exist
for _dir in [DATA_DIR, DOCUMENTS_DIR, EMBEDDINGS_DIR, VECTOR_STORE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE      = 500       # characters per chunk
CHUNK_OVERLAP   = 100       # overlap between consecutive chunks

# ── Embedding Model ──────────────────────────────────────────
# A lightweight, high-quality model that runs well on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── FAISS ────────────────────────────────────────────────────
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
CHUNKS_META_FILE = os.path.join(VECTOR_STORE_DIR, "chunks_meta.json")

# ── Retrieval ────────────────────────────────────────────────
TOP_K_RESULTS   = 5         # number of chunks to retrieve per query

# ── Summarization / QA Model ────────────────────────────────
# facebook/bart-large-cnn is fine for CPU; swap for a larger
# model if running on GPU/Colab A100.
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL            = "google/flan-t5-large"

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = "INFO"
