import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
import torch

class DummyRetriever:
    def is_ready(self):
        return True
    def retrieve(self, question, top_k):
        return [{"text": "The problem this paper solves is text classification.", "source": "dummy.pdf", "score": 0.9}]

rag = RAGPipeline(retriever=DummyRetriever(), qa_model="deepset/roberta-base-squad2")
res = rag.answer("What problem does this paper solve?")
print("Result:", res)

res2 = rag.answer("What is the capital of France?")
print("Result2:", res2)
