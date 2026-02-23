"""Run the UML extraction pipeline on a single image."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import uuid

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph import UMLKnowledgeGraphBuilder, export_to_json, export_to_graphml, export_statistics
from src.graph.arrow_matching import detect_arrows
from src.ocr import UMLTextExtractor
from src.query import GraphQAEngine
from src.utils import draw_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UML Query Tool pipeline runner")
    parser.add_argument("--image", required=True, help="Path to UML diagram image")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 model weights")
    parser.add_argument("--output-dir", default="data/processed/run", help="Output directory")
    parser.add_argument("--class-conf", type=float, default=0.5, help="Class detection confidence")
    parser.add_argument("--arrow-conf", type=float, default=0.3, help="Arrow detection confidence")
    parser.add_argument("--match-distance", type=float, default=100.0, help="Arrow to class match distance")
    parser.add_argument("--bbox-padding", type=int, default=10, help="OCR bbox padding")
    parser.add_argument("--export-graphml", action="store_true", help="Export GraphML")
    parser.add_argument("--export-json", action="store_true", help="Export JSON graph")
    parser.add_argument("--export-stats", action="store_true", help="Export statistics JSON")
    parser.add_argument("--overlay", action="store_true", help="Save overlay image")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs in output-dir")
    parser.add_argument("--push-neo4j", action="store_true", help="Store graph in Neo4j using .env credentials")
    parser.add_argument("--graph-id", default="", help="Graph ID for Neo4j storage (auto if empty)")
    parser.add_argument("--overwrite-neo4j", action="store_true", help="Overwrite Neo4j graph with same ID")
    parser.add_argument("--question", default="", help="Ask a question after building the graph")
    return parser.parse_args()


def planned_outputs(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if args.export_json:
        outputs.append(output_dir / "knowledge_graph.json")
    if args.export_graphml:
        outputs.append(output_dir / "knowledge_graph.graphml")
    if args.export_stats:
        outputs.append(output_dir / "statistics.json")
    if args.overlay:
        outputs.append(output_dir / "overlay.png")
    return outputs


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = [p for p in planned_outputs(args, output_dir) if p.exists()]
        if existing:
            files = "\n  - " + "\n  - ".join(str(p) for p in existing)
            raise SystemExit(
                "Output files already exist. Use --overwrite or choose a new --output-dir."
                f"{files}"
            )

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    extractor = UMLTextExtractor(
        model_path=str(model_path),
        confidence_threshold=args.class_conf,
        bbox_padding=args.bbox_padding,
        gpu=False,
        multi_strategy=True,
        use_pytesseract_fallback=True,
    )

    ocr_result = extractor.extract(str(image_path))

    builder = UMLKnowledgeGraphBuilder(
        model_path=str(model_path),
        arrow_confidence=args.arrow_conf,
        match_distance=args.match_distance,
    )

    graph = builder.build_from_ocr_results([ocr_result], verbose=False)

    print(f"Classes detected: {ocr_result.get('num_classes_detected', 0)}")
    print(f"Relationships detected: {graph.number_of_edges()}")

    graph_id = args.graph_id.strip()
    if args.push_neo4j and not graph_id:
        graph_id = f"uml_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    if args.export_json:
        export_to_json(graph, output_dir / "knowledge_graph.json")
    if args.export_graphml:
        export_to_graphml(graph, output_dir / "knowledge_graph.graphml")
    if args.export_stats:
        export_statistics(builder.get_statistics(), output_dir / "statistics.json")

    if args.overlay:
        image = cv2.imread(str(image_path))
        arrows = detect_arrows(builder.detector, image, args.arrow_conf)
        overlay = draw_overlay(
            str(image_path),
            ocr_result.get("classes", []),
            arrows,
        )
        overlay_path = output_dir / "overlay.png"
        overlay.save(overlay_path)
        print(f"Saved overlay: {overlay_path}")

    if args.push_neo4j:
        try:
            from src.graph.neo4j_store import get_config, get_driver, upsert_graph
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc

        config = get_config()
        driver = get_driver(config)
        try:
            upsert_graph(
                driver,
                graph_id=graph_id,
                graph=graph,
                metadata={"source_image": str(image_path)},
                overwrite=args.overwrite_neo4j,
                database=config.database,
            )
        finally:
            driver.close()

        print(f"Neo4j graph_id: {graph_id}")

    if args.question.strip():
        engine = GraphQAEngine(graph)
        result = engine.answer(args.question)
        print("\nQuestion:", args.question)
        print("Answer:", result.answer)
        if result.warnings:
            print("Warnings:", "; ".join(result.warnings))


if __name__ == "__main__":
    main()
