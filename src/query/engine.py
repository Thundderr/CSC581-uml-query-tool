"""GraphRAG query engine using Ollama for LLM-powered UML diagram queries."""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import networkx as nx


OLLAMA_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """You are an expert software architect analyzing UML class diagrams.
You answer questions about classes, their attributes, methods, and relationships
based on the extracted diagram data provided as context.

Rules:
- Only answer based on the provided context. If the information is not in the context, say so.
- Class names may contain OCR artifacts (typos, garbled text). Do your best to interpret them.
- Relationship types are either "inheritance" (is-a) or "association" (has-a / uses).
- Be concise and direct."""


def check_ollama() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def list_ollama_models() -> List[str]:
    """Return list of locally available Ollama model names."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except requests.ConnectionError:
        pass
    return []


class GraphRAGEngine:
    """Query a UML knowledge graph using natural language via Ollama."""

    def __init__(self, graph: nx.DiGraph, model_name: str = "llama3"):
        self.graph = graph
        self.model_name = model_name
        # Build lookup indexes for fast search
        self._build_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build text indexes over node data for keyword search."""
        self._node_texts: Dict[str, str] = {}
        for node_id, data in self.graph.nodes(data=True):
            parts = [data.get("name", "")]
            for attr in data.get("attributes", []):
                parts.append(attr.get("name", ""))
                parts.append(attr.get("type", ""))
            for method in data.get("methods", []):
                parts.append(method.get("name", ""))
                parts.append(method.get("return_type", ""))
            self._node_texts[node_id] = " ".join(parts).lower()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search_graph(self, query: str, max_results: int = 15) -> List[str]:
        """Find node IDs matching query keywords (fuzzy substring match)."""
        keywords = self._extract_keywords(query)
        if not keywords:
            # Fall back: return highest-degree nodes
            degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in degrees[:max_results]]

        scores: Dict[str, int] = {}
        for node_id, text in self._node_texts.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[node_id] = score

        # Keywords existed but matched nothing — fall back to highest-degree nodes
        if not scores:
            degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
            return [n for n, _ in degrees[:max_results]]

        ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
        return ranked[:max_results]

    def retrieve_context(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """Retrieve relevant nodes and edges for a query.

        Returns (nodes_data, edges_data) where each is a list of dicts.
        """
        matched_ids = self.search_graph(query)
        # Expand to 1-hop neighbors
        expanded: Set[str] = set(matched_ids)
        for nid in matched_ids:
            expanded.update(self.graph.predecessors(nid))
            expanded.update(self.graph.successors(nid))

        nodes_data = []
        for nid in expanded:
            if nid in self.graph:
                data = dict(self.graph.nodes[nid])
                data["id"] = nid
                nodes_data.append(data)

        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            if u in expanded and v in expanded:
                edge = dict(data)
                edge["source"] = self.graph.nodes[u].get("name", u)
                edge["target"] = self.graph.nodes[v].get("name", v)
                edges_data.append(edge)

        return nodes_data, edges_data

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Format retrieved graph data as text for the LLM prompt."""
        lines = []
        if nodes:
            lines.append("=== Classes ===")
            for n in nodes:
                name = n.get("name", "Unknown")
                lines.append(f"\nClass: {name}")
                attrs = n.get("attributes", [])
                if attrs:
                    lines.append("  Attributes:")
                    for a in attrs:
                        vis = a.get("visibility", "public")
                        lines.append(f"    {vis} {a.get('name', '?')}: {a.get('type', '?')}")
                methods = n.get("methods", [])
                if methods:
                    lines.append("  Methods:")
                    for m in methods:
                        vis = m.get("visibility", "public")
                        params = ", ".join(
                            f"{p.get('name','?')}: {p.get('type','?')}"
                            for p in m.get("parameters", [])
                        )
                        lines.append(
                            f"    {vis} {m.get('name', '?')}({params}): {m.get('return_type', 'void')}"
                        )

        if edges:
            lines.append("\n=== Relationships ===")
            for e in edges:
                rel = e.get("relationship_type", "association")
                lines.append(f"  {e.get('source','?')} --[{rel}]--> {e.get('target','?')}")

        return "\n".join(lines) if lines else "No relevant classes or relationships found."

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        """Send a prompt to Ollama and return the response text."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure it is running (ollama serve)."
        except requests.Timeout:
            return "Error: Ollama request timed out."
        except Exception as e:
            return f"Error: {e}"

    def _stream_ollama(self, prompt: str, system: str = SYSTEM_PROMPT):
        """Stream tokens from Ollama. Yields (token, done) tuples."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system,
            "stream": True,
        }
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", ""), chunk.get("done", False)
        except requests.ConnectionError:
            yield "Error: Cannot connect to Ollama.", True
        except requests.Timeout:
            yield "Error: Ollama request timed out.", True
        except Exception as e:
            yield f"Error: {e}", True

    def query_stream(self, question: str):
        """Streaming version of query. Yields (token, done) tuples.

        Call retrieve_context and format_context first, then stream LLM tokens.
        Returns a generator plus the context info.
        """
        nodes, edges = self.retrieve_context(question)
        context_text = self.format_context(nodes, edges)

        prompt = (
            f"Context from the UML diagram:\n{context_text}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        return self._stream_ollama(prompt), context_text, len(nodes), len(edges)

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """Run the full GraphRAG pipeline: retrieve → format → ask LLM.

        Returns dict with 'answer', 'context', 'nodes_used', 'edges_used'.
        """
        nodes, edges = self.retrieve_context(question)
        context_text = self.format_context(nodes, edges)

        prompt = (
            f"Context from the UML diagram:\n{context_text}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        answer = self._call_ollama(prompt)

        return {
            "answer": answer,
            "context": context_text,
            "nodes_used": len(nodes),
            "edges_used": len(edges),
        }

    def query_without_llm(self, question: str) -> str:
        """Return just the retrieved context (no LLM) for debugging."""
        nodes, edges = self.retrieve_context(question)
        return self.format_context(nodes, edges)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract meaningful keywords from a query string."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "which",
            "who", "how", "does", "do", "did", "can", "could", "would",
            "should", "will", "shall", "has", "have", "had", "be", "been",
            "being", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "as", "into", "about", "between", "through", "and",
            "or", "but", "not", "all", "any", "each", "every", "this",
            "that", "these", "those", "it", "its", "my", "your", "our",
            "their", "me", "him", "her", "us", "them", "show", "list",
            "tell", "give", "find", "get", "describe", "explain",
            "class", "classes", "relationship", "relationships",
            "attribute", "attributes", "method", "methods",
        }
        words = re.findall(r"[a-zA-Z_]\w*", text.lower())
        return [w for w in words if w not in stop_words and len(w) > 1]
