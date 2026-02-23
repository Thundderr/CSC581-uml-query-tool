"""List UML graphs stored in Neo4j."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.neo4j_store import get_config, get_driver, list_graphs


def main() -> None:
    config = get_config()
    driver = get_driver(config)
    try:
        graphs = list_graphs(driver, database=config.database)
    finally:
        driver.close()

    if not graphs:
        print("No graphs found in Neo4j.")
        return

    for g in graphs:
        print(
            f"{g.get('graph_id')} | nodes={g.get('node_count')} | "
            f"edges={g.get('edge_count')} | created_at={g.get('created_at')}"
        )


if __name__ == "__main__":
    main()
