"""Run a question against an exported UML knowledge graph JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph import load_from_json
from src.query import GraphQAEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UML Query Tool graph QA")
    parser.add_argument("--graph", required=True, help="Path to knowledge_graph.json")
    parser.add_argument("--question", required=True, help="Question to ask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise SystemExit(f"Graph file not found: {graph_path}")

    graph = load_from_json(graph_path)
    engine = GraphQAEngine(graph)
    result = engine.answer(args.question)

    print("Question:", args.question)
    print("Answer:", result.answer)
    if result.warnings:
        print("Warnings:", "; ".join(result.warnings))


if __name__ == "__main__":
    main()
