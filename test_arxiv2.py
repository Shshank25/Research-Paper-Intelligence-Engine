import arxiv

client = arxiv.Client()
search = arxiv.Search(query="Deep Reinforcement Learning", max_results=1)
results = list(client.results(search))
r = results[0]
print(dir(r))
print(f"pdf_url: {r.pdf_url}")
