import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
os.environ["PYTHONUSERBASE"] = r"E:\python_user"
sys.path.insert(0, BASE_DIR)

from src.rag_pipeline import RAGPipeline

class DummyRetriever:
    def is_ready(self): return True
    def retrieve(self, q, top_k):
        return [{"text": "", "source": "dummy.pdf", "score": 0.9}]

print("Initializing RAGPipeline...")
rag = RAGPipeline(retriever=DummyRetriever())
print("Answering question 1...")
res = rag.answer("What problem does this paper solve?")
print("Result 1:", res)

print("Answering question 2...")
res2 = rag.answer("Is it raining?")
print("Result 2:", res2)
