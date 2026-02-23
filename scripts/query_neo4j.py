"""Query a UML graph stored in Neo4j using rule-based QA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.neo4j_store import get_config, get_driver, load_graph
from src.query import GraphQAEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a Neo4j-stored UML graph")
    parser.add_argument("--graph-id", required=True, help="Graph ID stored in Neo4j")
    parser.add_argument("--question", required=True, help="Question to ask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = get_config()
    driver = get_driver(config)
    try:
        graph = load_graph(driver, args.graph_id, database=config.database)
    finally:
        driver.close()

    engine = GraphQAEngine(graph)
    result = engine.answer(args.question)
    print("Question:", args.question)
    print("Answer:", result.answer)
    if result.warnings:
        print("Warnings:", "; ".join(result.warnings))


if __name__ == "__main__":
    main()
