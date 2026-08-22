import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.arxiv_search import ArxivSearcher

searcher = ArxivSearcher(limit=5)
query = '"Deep Reinforcement Learning for Multi-Agent Path Finding in Warehouse Robotics"'
print(f"Searching with quotes: {query}")
results1 = searcher.search_and_download(query)
print(f"Results with quotes: {len(results1)}")

query2 = 'Deep Reinforcement Learning for Multi-Agent Path Finding in Warehouse Robotics'
print(f"\nSearching without quotes: {query2}")
results2 = searcher.search_and_download(query2)
print(f"Results without quotes: {len(results2)}")
