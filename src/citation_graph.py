# ============================================================
# src/citation_graph.py
# Concept Co-occurrence Graph for Paper Relationships
#
# Extracts key concepts from paper abstracts using TF-IDF
# and builds a concept co-occurrence graph showing how
# papers relate to each other through shared concepts.
# ============================================================

import re
import math
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ConceptGraphBuilder:
    """
    Builds a concept co-occurrence graph from paper abstracts.

    Extracts key concepts using TF-IDF-style scoring and builds
    edges between papers that share significant concepts.

    Usage:
        builder = ConceptGraphBuilder()
        graph = builder.build_graph(papers)
        mermaid = builder.to_mermaid(graph)
    """

    # Common academic stopwords to filter out
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "over", "after",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "this", "that", "these", "those",
        "it", "its", "we", "our", "they", "their", "them", "which", "who",
        "whom", "what", "where", "when", "how", "than", "then", "also",
        "not", "no", "nor", "so", "if", "as", "such", "each", "every",
        "all", "both", "few", "more", "most", "other", "some", "any",
        "new", "used", "using", "based", "paper", "propose", "proposed",
        "method", "approach", "results", "show", "work", "two", "one",
        "first", "use", "however", "well", "still", "even", "also",
        "between", "through", "during", "before", "while", "across",
        "several", "many", "different", "various", "existing", "recent",
    }

    def __init__(self, top_k_concepts: int = 8, min_shared: int = 2):
        """
        Args:
            top_k_concepts: Number of key concepts to extract per paper.
            min_shared: Minimum shared concepts for an edge between papers.
        """
        self.top_k_concepts = top_k_concepts
        self.min_shared = min_shared

    def _extract_ngrams(self, text: str) -> list:
        """Extract unigrams and bigrams from text."""
        # Lowercase and clean
        text = text.lower()
        text = re.sub(r'[^a-z\s\-]', ' ', text)
        words = [w for w in text.split() if len(w) > 2 and w not in self.STOPWORDS]

        # Unigrams
        ngrams = list(words)

        # Bigrams (joined with space)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            ngrams.append(bigram)

        return ngrams

    def _tfidf_keywords(self, abstracts: list) -> list:
        """
        Extract top keywords from each abstract using TF-IDF scoring.

        Args:
            abstracts: List of abstract strings.

        Returns:
            List of sets, each set containing top concepts for that paper.
        """
        # Build document frequency
        doc_ngrams = []
        doc_freq = Counter()

        for abstract in abstracts:
            ngrams = self._extract_ngrams(abstract)
            unique = set(ngrams)
            doc_ngrams.append(Counter(ngrams))
            for ng in unique:
                doc_freq[ng] += 1

        n_docs = len(abstracts)
        results = []

        for tf_counts in doc_ngrams:
            # Compute TF-IDF for each ngram
            scores = {}
            total = sum(tf_counts.values())
            for ngram, count in tf_counts.items():
                tf = count / total
                idf = math.log(n_docs / (1 + doc_freq[ngram]))
                # Boost bigrams
                boost = 1.5 if ' ' in ngram else 1.0
                scores[ngram] = tf * idf * boost

            # Take top-k
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k_concepts]
            results.append(set(kw for kw, _ in top))

        return results

    def build_graph(self, papers: list) -> dict:
        """
        Build a concept co-occurrence graph from papers.

        Args:
            papers: List of paper dicts with 'title' and 'abstract'.

        Returns:
            {
                "nodes": [{"id": str, "title": str, "concepts": list}],
                "edges": [{"source": str, "target": str, "shared": list, "weight": int}],
            }
        """
        if not papers:
            return {"nodes": [], "edges": []}

        abstracts = [p.get("abstract", "") for p in papers]
        paper_concepts = self._tfidf_keywords(abstracts)

        # Build nodes
        nodes = []
        for i, paper in enumerate(papers):
            short_title = paper.get("title", f"Paper {i+1}")
            if len(short_title) > 40:
                short_title = short_title[:37] + "..."
            nodes.append({
                "id": f"P{i+1}",
                "title": short_title,
                "concepts": sorted(paper_concepts[i]),
            })

        # Build edges
        edges = []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                shared = paper_concepts[i] & paper_concepts[j]
                if len(shared) >= self.min_shared:
                    edges.append({
                        "source": f"P{i+1}",
                        "target": f"P{j+1}",
                        "shared": sorted(shared),
                        "weight": len(shared),
                    })

        logger.info(f"Concept graph: {len(nodes)} nodes, {len(edges)} edges")
        return {"nodes": nodes, "edges": edges}

    def to_mermaid(self, graph: dict) -> str:
        """
        Convert the graph to a Mermaid diagram string.

        Args:
            graph: Graph dict from build_graph().

        Returns:
            Mermaid diagram string.
        """
        if not graph["nodes"]:
            return "graph LR\n    empty[No papers to visualize]"

        lines = ["graph LR"]

        # Style definitions
        lines.append("    classDef paper fill:#1a1a2e,stroke:#00d4ff,stroke-width:2px,color:#e0e0e0")
        lines.append("    classDef concept fill:#0f0f23,stroke:#a855f7,stroke-width:1px,color:#c0c0c0,font-size:10px")

        # Node definitions
        for node in graph["nodes"]:
            safe_title = node["title"].replace('"', "'")
            lines.append(f'    {node["id"]}["{safe_title}"]')

        # Edges with shared concept labels
        for edge in graph["edges"]:
            label = ", ".join(edge["shared"][:3])
            if len(edge["shared"]) > 3:
                label += f" +{len(edge['shared'])-3}"
            safe_label = label.replace('"', "'")
            thickness = "==>" if edge["weight"] >= 3 else "-->"
            lines.append(f'    {edge["source"]} {thickness}|"{safe_label}"| {edge["target"]}')

        # Apply styles
        node_ids = " & ".join(n["id"] for n in graph["nodes"])
        if node_ids:
            lines.append(f"    class {node_ids} paper")

        return "\n".join(lines)

    def to_summary(self, graph: dict) -> str:
        """
        Generate a text summary of the concept relationships.

        Args:
            graph: Graph dict from build_graph().

        Returns:
            Human-readable summary string.
        """
        if not graph["edges"]:
            return "No significant concept overlap found between papers."

        lines = ["**Key Concept Relationships:**\n"]
        for edge in sorted(graph["edges"], key=lambda e: e["weight"], reverse=True):
            src = next(n for n in graph["nodes"] if n["id"] == edge["source"])
            tgt = next(n for n in graph["nodes"] if n["id"] == edge["target"])
            concepts = ", ".join(edge["shared"])
            lines.append(f"- **{src['title']}** ↔ **{tgt['title']}** share: _{concepts}_")

        return "\n".join(lines)
