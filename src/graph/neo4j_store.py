"""Neo4j storage utilities for UML knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

try:
    from neo4j import GraphDatabase
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("neo4j driver not installed. Run: py -m pip install neo4j") from exc


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_config() -> Neo4jConfig:
    _load_env_file(ROOT / ".env")
    uri = os.getenv("NEO4J_URI", "")
    username = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not uri or not username or not password:
        raise ValueError("Missing Neo4j env vars. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD.")
    return Neo4jConfig(uri=uri, username=username, password=password, database=database)


def get_driver(config: Optional[Neo4jConfig] = None):
    if config is None:
        config = get_config()
    return GraphDatabase.driver(config.uri, auth=(config.username, config.password))


def ensure_schema(driver, database: str = "neo4j") -> None:
    with driver.session(database=database) as session:
        session.run(
            "CREATE CONSTRAINT uml_graph_id IF NOT EXISTS "
            "FOR (g:UMLGraph) REQUIRE g.graph_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT uml_class_key IF NOT EXISTS "
            "FOR (c:UMLClass) REQUIRE (c.graph_id, c.node_id) IS UNIQUE"
        )


def graph_exists(driver, graph_id: str, database: str = "neo4j") -> bool:
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (g:UMLGraph {graph_id: $graph_id}) RETURN g.graph_id AS id",
            graph_id=graph_id,
        )
        return result.single() is not None


def list_graphs(driver, database: str = "neo4j") -> List[Dict[str, Any]]:
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (g:UMLGraph) "
            "OPTIONAL MATCH (g)-[:CONTAINS]->(c:UMLClass) "
            "WITH g, count(c) AS node_count "
            "OPTIONAL MATCH (c:UMLClass {graph_id: g.graph_id})-[r:UML_REL]->(:UMLClass {graph_id: g.graph_id}) "
            "WITH g, node_count, count(r) AS edge_count "
            "RETURN g.graph_id AS graph_id, g.created_at AS created_at, "
            "node_count, edge_count "
            "ORDER BY g.created_at DESC"
        )
        return [dict(record) for record in result]


def _to_json(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def upsert_graph(
    driver,
    graph_id: str,
    graph: nx.DiGraph,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    database: str = "neo4j",
) -> None:
    ensure_schema(driver, database=database)

    if graph_exists(driver, graph_id, database=database) and not overwrite:
        raise ValueError(
            f"Graph '{graph_id}' already exists. Use overwrite=True or a new graph_id."
        )

    created_at = datetime.utcnow().isoformat()
    meta = metadata or {}

    with driver.session(database=database) as session:
        session.run(
            "MERGE (g:UMLGraph {graph_id: $graph_id}) "
            "SET g.created_at = coalesce(g.created_at, $created_at) "
            "SET g += $meta",
            graph_id=graph_id,
            created_at=created_at,
            meta=meta,
        )

        for node_id, data in graph.nodes(data=True):
            props = {k: _to_json(v) for k, v in data.items()}
            props.update({"graph_id": graph_id, "node_id": node_id})
            session.run(
                "MERGE (c:UMLClass {graph_id: $graph_id, node_id: $node_id}) "
                "SET c += $props",
                graph_id=graph_id,
                node_id=node_id,
                props=props,
            )
            session.run(
                "MATCH (g:UMLGraph {graph_id: $graph_id}), "
                "(c:UMLClass {graph_id: $graph_id, node_id: $node_id}) "
                "MERGE (g)-[:CONTAINS]->(c)",
                graph_id=graph_id,
                node_id=node_id,
            )

        for source, target, data in graph.edges(data=True):
            props = {k: _to_json(v) for k, v in data.items()}
            props.update({
                "graph_id": graph_id,
                "source_id": source,
                "target_id": target,
            })
            session.run(
                "MATCH (s:UMLClass {graph_id: $graph_id, node_id: $source_id}), "
                "(t:UMLClass {graph_id: $graph_id, node_id: $target_id}) "
                "MERGE (s)-[r:UML_REL {graph_id: $graph_id, source_id: $source_id, target_id: $target_id}]->(t) "
                "SET r += $props",
                graph_id=graph_id,
                source_id=source,
                target_id=target,
                props=props,
            )


def load_graph(driver, graph_id: str, database: str = "neo4j") -> nx.DiGraph:
    graph = nx.DiGraph()
    with driver.session(database=database) as session:
        nodes = session.run(
            "MATCH (c:UMLClass {graph_id: $graph_id}) RETURN c",
            graph_id=graph_id,
        )
        for record in nodes:
            props = dict(record["c"])
            node_id = props.pop("node_id")
            props.pop("graph_id", None)
            props = {k: _maybe_json(v) for k, v in props.items()}
            graph.add_node(node_id, **props)

        edges = session.run(
            "MATCH (s:UMLClass {graph_id: $graph_id})-[r:UML_REL {graph_id: $graph_id}]->"
            "(t:UMLClass {graph_id: $graph_id}) "
            "RETURN s.node_id AS source_id, t.node_id AS target_id, r",
            graph_id=graph_id,
        )
        for record in edges:
            source = record["source_id"]
            target = record["target_id"]
            props = dict(record["r"])
            props.pop("graph_id", None)
            props.pop("source_id", None)
            props.pop("target_id", None)
            props = {k: _maybe_json(v) for k, v in props.items()}
            graph.add_edge(source, target, **props)

    return graph
