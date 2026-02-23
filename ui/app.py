"""Streamlit UI for UML diagram parsing and querying."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph import UMLKnowledgeGraphBuilder
from src.graph.arrow_matching import detect_arrows
from src.ocr import UMLTextExtractor
from src.query import GraphQAEngine
from src.utils import draw_overlay


st.set_page_config(page_title="UML Query Tool", layout="wide")


@st.cache_resource(show_spinner=False)
def load_text_extractor(model_path: str, confidence: float, padding: int) -> UMLTextExtractor:
    return UMLTextExtractor(
        model_path=model_path,
        confidence_threshold=confidence,
        bbox_padding=padding,
        gpu=False,
        multi_strategy=True,
        use_pytesseract_fallback=True,
    )


@st.cache_resource(show_spinner=False)
def load_graph_builder(model_path: str, arrow_confidence: float, match_distance: float) -> UMLKnowledgeGraphBuilder:
    return UMLKnowledgeGraphBuilder(
        model_path=model_path,
        arrow_confidence=arrow_confidence,
        match_distance=match_distance,
    )


def process_diagram(image_path: str, model_path: str,
                    class_conf: float, arrow_conf: float,
                    match_distance: float, padding: int) -> Tuple[Dict[str, Any], Any, List[Dict]]:
    extractor = load_text_extractor(model_path, class_conf, padding)
    builder = load_graph_builder(model_path, arrow_conf, match_distance)

    ocr_result = extractor.extract(image_path)
    graph = builder.build_from_ocr_results([ocr_result], verbose=False)

    image = cv2.imread(image_path)
    arrows = detect_arrows(builder.detector, image, arrow_conf)

    return ocr_result, graph, arrows


st.title("UML Class Diagram Query Tool")

with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input(
        "YOLO model path",
        value=os.getenv("UML_MODEL_PATH", ""),
        help="Path to trained YOLOv8 weights for UML classes/arrows.",
    )
    class_conf = st.slider("Class detection confidence", 0.1, 0.9, 0.5, 0.05)
    arrow_conf = st.slider("Arrow detection confidence", 0.1, 0.9, 0.3, 0.05)
    match_distance = st.slider("Arrow-to-class match distance", 20, 300, 100, 10)
    bbox_padding = st.slider("OCR bbox padding", 0, 30, 10, 2)

st.write("Upload a UML class diagram and run the pipeline to extract classes, relationships, and query the graph.")

uploaded = st.file_uploader("UML Diagram Image", type=["png", "jpg", "jpeg"])
run = st.button("Process Diagram")

if run:
    if not uploaded:
        st.error("Please upload a diagram image first.")
        st.stop()
    if not model_path:
        st.error("Please provide a YOLO model path in the sidebar.")
        st.stop()
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    with st.spinner("Processing diagram..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            ocr_result, graph, arrows = process_diagram(
                tmp_path,
                model_path,
                class_conf,
                arrow_conf,
                match_distance,
                bbox_padding,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {exc}")
            st.stop()

        st.session_state["ocr_result"] = ocr_result
        st.session_state["graph"] = graph
        st.session_state["arrows"] = arrows
        st.session_state["image_path"] = tmp_path

if "graph" in st.session_state:
    graph = st.session_state["graph"]
    ocr_result = st.session_state.get("ocr_result", {})
    arrows = st.session_state.get("arrows", [])
    image_path = st.session_state.get("image_path")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Diagram Overlay")
        if image_path:
            overlay = draw_overlay(image_path, ocr_result.get("classes", []), arrows)
            st.image(overlay, use_container_width=True)

    with col2:
        st.subheader("Summary")
        st.write(f"Classes detected: {ocr_result.get('num_classes_detected', 0)}")
        st.write(f"Relationships detected: {graph.number_of_edges()}")

        class_names = [c.get("class_name", "Unknown") for c in ocr_result.get("classes", [])]
        if class_names:
            st.write("Class list:")
            st.code("\n".join(class_names))

    st.divider()
    st.subheader("Ask a Question")
    question = st.text_input("Question", placeholder="Which classes depend on OrderManager?")
    ask = st.button("Ask")

    if ask and question.strip():
        engine = GraphQAEngine(graph)
        result = engine.answer(question)
        st.markdown("**Answer**")
        st.write(result.answer)
        st.markdown("**Evidence**")
        st.code(json.dumps(result.evidence, indent=2))
        if result.warnings:
            st.warning("; ".join(result.warnings))
