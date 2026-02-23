"""
Rule-based question answering over UML knowledge graphs.

This module provides a lightweight QA engine for typical UML questions
without relying on an LLM. It returns a structured answer plus evidence
from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


@dataclass
class QAResult:
    """Result of a QA query over a graph."""
    answer: str
    evidence: Dict[str, Any]
    matched_intent: str
    confidence: float
    warnings: List[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.strip().lower())


def _unique_preserve(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _index_names(graph: nx.DiGraph) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for node_id, data in graph.nodes(data=True):
        name = str(data.get("name", "")).strip()
        if not name:
            continue
        key = _normalize_name(name)
        if not key:
            continue
        index.setdefault(key, []).append(node_id)
    return index


def _extract_quoted_names(question: str) -> List[str]:
    quoted = []
    quoted.extend(re.findall(r'"([^"]+)"', question))
    quoted.extend(re.findall(r"'([^']+)'", question))
    return [q.strip() for q in quoted if q.strip()]


def _match_names_in_question(question: str, graph: nx.DiGraph) -> List[str]:
    q = question.lower()
    matches = []
    for _, data in graph.nodes(data=True):
        name = str(data.get("name", "")).strip()
        if not name:
            continue
        if name.lower() in q:
            matches.append(name)
    return _unique_preserve(matches)


def _resolve_class_nodes(
    graph: nx.DiGraph,
    name: str,
    name_index: Dict[str, List[str]]
) -> List[str]:
    if not name:
        return []
    key = _normalize_name(name)
    if not key:
        return []
    if key in name_index:
        return name_index[key]

    # Fallback: partial match
    for indexed_key, node_ids in name_index.items():
        if key in indexed_key or indexed_key in key:
            return node_ids
    return []


def _class_names_for_nodes(graph: nx.DiGraph, node_ids: List[str]) -> List[str]:
    names = []
    for node_id in node_ids:
        data = graph.nodes.get(node_id, {})
        name = str(data.get("name", "Unknown")).strip()
        names.append(name or "Unknown")
    return _unique_preserve(names)


def _edge_evidence(edges: List[Tuple[str, str, Dict[str, Any]]], graph: nx.DiGraph) -> List[Dict[str, Any]]:
    evidence = []
    for source, target, data in edges:
        evidence.append({
            "source": graph.nodes[source].get("name", source),
            "target": graph.nodes[target].get("name", target),
            "relationship_type": data.get("relationship_type", "unknown"),
            "match_confidence": data.get("match_confidence", 0.0),
        })
    return evidence


class GraphQAEngine:
    """Simple QA engine for UML knowledge graphs."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._name_index = _index_names(graph)

    def refresh_index(self) -> None:
        self._name_index = _index_names(self.graph)

    def answer(self, question: str) -> QAResult:
        warnings: List[str] = []
        if self.graph.number_of_nodes() == 0:
            return QAResult(
                answer="The graph is empty. I cannot answer questions yet.",
                evidence={"node_count": 0, "edge_count": 0},
                matched_intent="empty_graph",
                confidence=0.0,
                warnings=[],
            )

        q_raw = question.strip()
        q = _normalize(q_raw)

        # Intent: count classes
        if re.search(r"\bhow many\b.*\bclasses\b", q) or "number of classes" in q:
            count = sum(1 for _, data in self.graph.nodes(data=True)
                        if data.get("type") in ("class", None))
            return QAResult(
                answer=f"Detected {count} classes.",
                evidence={"class_count": count},
                matched_intent="count_classes",
                confidence=0.9,
                warnings=[],
            )

        # Intent: list classes
        if re.search(r"\blist\b.*\bclasses\b", q) or re.search(r"\bshow\b.*\bclasses\b", q):
            names = _unique_preserve([
                str(data.get("name", "Unknown")).strip()
                for _, data in self.graph.nodes(data=True)
                if str(data.get("name", "")).strip()
            ])
            if not names:
                warnings.append("No class names found in graph nodes.")
            return QAResult(
                answer="Classes: " + (", ".join(names) if names else "(none)"),
                evidence={"classes": names},
                matched_intent="list_classes",
                confidence=0.85,
                warnings=warnings,
            )

        # Intent: list inheritance relationships
        if "inheritance" in q and ("list" in q or "show" in q):
            edges = [
                (u, v, data)
                for u, v, data in self.graph.edges(data=True)
                if data.get("relationship_type") == "inheritance"
            ]
            evidence = _edge_evidence(edges, self.graph)
            if edges:
                answer = "Inheritance relationships: " + "; ".join(
                    f"{e['source']} -> {e['target']}" for e in evidence
                )
            else:
                answer = "No inheritance relationships found."
            return QAResult(
                answer=answer,
                evidence={"relationships": evidence},
                matched_intent="list_inheritance",
                confidence=0.8,
                warnings=[],
            )

        # Intent: list relationships of a type
        match_rel = re.search(r"\b(list|show)\b.*\b(association|composition|aggregation|dependency)\b", q)
        if match_rel:
            rel_type = match_rel.group(2)
            edges = [
                (u, v, data)
                for u, v, data in self.graph.edges(data=True)
                if data.get("relationship_type") == rel_type
            ]
            evidence = _edge_evidence(edges, self.graph)
            if edges:
                answer = f"{rel_type.title()} relationships: " + "; ".join(
                    f"{e['source']} -> {e['target']}" for e in evidence
                )
            else:
                answer = f"No {rel_type} relationships found."
            return QAResult(
                answer=answer,
                evidence={"relationships": evidence},
                matched_intent=f"list_{rel_type}",
                confidence=0.8,
                warnings=[],
            )

        # Intent: is A connected to B
        m = re.search(r"\bis\b\s+(.+?)\s+connected to\s+(.+?)(\?|$)", q_raw, flags=re.IGNORECASE)
        if m:
            name_a = m.group(1).strip()
            name_b = m.group(2).strip()
            nodes_a = _resolve_class_nodes(self.graph, name_a, self._name_index)
            nodes_b = _resolve_class_nodes(self.graph, name_b, self._name_index)
            if not nodes_a or not nodes_b:
                return QAResult(
                    answer="I could not resolve one or both class names in the graph.",
                    evidence={"name_a": name_a, "name_b": name_b},
                    matched_intent="connected_check",
                    confidence=0.4,
                    warnings=["Missing class name match."],
                )

            direct_edges = []
            for a in nodes_a:
                for b in nodes_b:
                    if self.graph.has_edge(a, b):
                        direct_edges.append((a, b, self.graph.edges[a, b]))
                    if self.graph.has_edge(b, a):
                        direct_edges.append((b, a, self.graph.edges[b, a]))

            if direct_edges:
                edge_evidence = _edge_evidence(direct_edges, self.graph)
                answer = "Yes. Direct relationship(s) found."
                confidence = 0.9
                evidence = {"direct_edges": edge_evidence}
            else:
                # Check undirected connectivity as a weaker signal
                undirected = self.graph.to_undirected()
                connected = any(
                    nx.has_path(undirected, a, b) for a in nodes_a for b in nodes_b
                )
                if connected:
                    answer = "There is an indirect path between those classes."
                    confidence = 0.6
                else:
                    answer = "No relationship found between those classes."
                    confidence = 0.7
                evidence = {"direct_edges": []}

            return QAResult(
                answer=answer,
                evidence=evidence,
                matched_intent="connected_check",
                confidence=confidence,
                warnings=[],
            )

        # Intent: classes connected to X
        m = re.search(r"\b(classes|class) connected to\s+(.+)$", q_raw, flags=re.IGNORECASE)
        if m:
            name = m.group(2).strip().rstrip("?")
            node_ids = _resolve_class_nodes(self.graph, name, self._name_index)
            if not node_ids:
                return QAResult(
                    answer="I could not find that class in the graph.",
                    evidence={"name": name},
                    matched_intent="neighbors",
                    confidence=0.4,
                    warnings=["Missing class name match."],
                )
            neighbors = set()
            edges = []
            for node_id in node_ids:
                for _, target, data in self.graph.out_edges(node_id, data=True):
                    neighbors.add(target)
                    edges.append((node_id, target, data))
                for source, _, data in self.graph.in_edges(node_id, data=True):
                    neighbors.add(source)
                    edges.append((source, node_id, data))

            names = _class_names_for_nodes(self.graph, list(neighbors))
            evidence = _edge_evidence(edges, self.graph)
            return QAResult(
                answer="Connected classes: " + (", ".join(names) if names else "(none)"),
                evidence={"classes": names, "relationships": evidence},
                matched_intent="neighbors",
                confidence=0.8,
                warnings=[],
            )

        # Intent: what does X connect to
        m = re.search(r"\bwhat does\s+(.+?)\s+connect to\b", q_raw, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            node_ids = _resolve_class_nodes(self.graph, name, self._name_index)
            if not node_ids:
                return QAResult(
                    answer="I could not find that class in the graph.",
                    evidence={"name": name},
                    matched_intent="outgoing",
                    confidence=0.4,
                    warnings=["Missing class name match."],
                )
            edges = []
            targets = set()
            for node_id in node_ids:
                for _, target, data in self.graph.out_edges(node_id, data=True):
                    targets.add(target)
                    edges.append((node_id, target, data))
            names = _class_names_for_nodes(self.graph, list(targets))
            return QAResult(
                answer="Outgoing connections: " + (", ".join(names) if names else "(none)"),
                evidence={"classes": names, "relationships": _edge_evidence(edges, self.graph)},
                matched_intent="outgoing",
                confidence=0.8,
                warnings=[],
            )

        # Intent: which classes depend on X
        m = re.search(r"\bdepend on\s+(.+)$", q_raw, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip("?")
            node_ids = _resolve_class_nodes(self.graph, name, self._name_index)
            if not node_ids:
                return QAResult(
                    answer="I could not find that class in the graph.",
                    evidence={"name": name},
                    matched_intent="incoming",
                    confidence=0.4,
                    warnings=["Missing class name match."],
                )
            edges = []
            sources = set()
            for node_id in node_ids:
                for source, _, data in self.graph.in_edges(node_id, data=True):
                    sources.add(source)
                    edges.append((source, node_id, data))
            names = _class_names_for_nodes(self.graph, list(sources))
            return QAResult(
                answer="Classes that depend on it: " + (", ".join(names) if names else "(none)"),
                evidence={"classes": names, "relationships": _edge_evidence(edges, self.graph)},
                matched_intent="incoming",
                confidence=0.8,
                warnings=[],
            )

        # Intent: relationships for class
        m = re.search(r"\brelationships for\s+(.+)$", q_raw, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip("?")
            node_ids = _resolve_class_nodes(self.graph, name, self._name_index)
            if not node_ids:
                return QAResult(
                    answer="I could not find that class in the graph.",
                    evidence={"name": name},
                    matched_intent="relationships_for_class",
                    confidence=0.4,
                    warnings=["Missing class name match."],
                )
            edges = []
            for node_id in node_ids:
                for _, target, data in self.graph.out_edges(node_id, data=True):
                    edges.append((node_id, target, data))
                for source, _, data in self.graph.in_edges(node_id, data=True):
                    edges.append((source, node_id, data))
            evidence = _edge_evidence(edges, self.graph)
            answer = "Relationships: " + ("; ".join(
                f"{e['source']} -> {e['target']} ({e['relationship_type']})" for e in evidence
            ) if evidence else "(none)")
            return QAResult(
                answer=answer,
                evidence={"relationships": evidence},
                matched_intent="relationships_for_class",
                confidence=0.75,
                warnings=[],
            )

        # Fallback: try name inference and show neighbors
        quoted = _extract_quoted_names(q_raw)
        if quoted:
            name = quoted[0]
            node_ids = _resolve_class_nodes(self.graph, name, self._name_index)
            if node_ids:
                neighbors = set()
                for node_id in node_ids:
                    neighbors.update(self.graph.successors(node_id))
                    neighbors.update(self.graph.predecessors(node_id))
                names = _class_names_for_nodes(self.graph, list(neighbors))
                return QAResult(
                    answer="Related classes: " + (", ".join(names) if names else "(none)"),
                    evidence={"classes": names},
                    matched_intent="fallback_neighbors",
                    confidence=0.5,
                    warnings=["Matched via quoted name fallback."],
                )

        # Final fallback
        return QAResult(
            answer="I could not match that question to a known query pattern yet.",
            evidence={"question": q_raw},
            matched_intent="no_match",
            confidence=0.2,
            warnings=["Consider rephrasing with a class name or a list/count query."],
        )


def answer_question(graph: nx.DiGraph, question: str) -> QAResult:
    """Convenience helper for one-off queries."""
    engine = GraphQAEngine(graph)
    return engine.answer(question)
