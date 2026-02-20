"""
UML Diagram Query Tool — Streamlit App

Upload a UML class diagram image, run the full analysis pipeline
(detection → OCR → knowledge graph), and ask questions via Ollama LLM.

Launch:  streamlit run app.py
"""

import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import networkx as nx
import torch

# Add project root so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device
from src.ocr.extractor import UMLTextExtractor, parse_uml_class
from src.graph.arrow_matching import detect_arrows, match_arrows_to_classes
from src.query.engine import GraphRAGEngine, check_ollama, list_ollama_models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "models" / "uml_detector_best.pt"
DEVICE = get_device()

CLASS_COLORS = {
    "arrow": (0, 200, 0),
    "class": (255, 100, 0),
    "cross": (200, 0, 0),
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_extractor(model_path: str, gpu: bool = False) -> UMLTextExtractor:
    return UMLTextExtractor(
        model_path=model_path,
        confidence_threshold=0.5,
        bbox_padding=10,
        gpu=gpu,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_np, extractor, arrow_conf, match_dist):
    h, w = image_np.shape[:2]

    class_detections = extractor.detect_classes(image_np)
    arrows = detect_arrows(extractor.detector, image_np, confidence_threshold=arrow_conf)

    classes = []
    for det in class_detections:
        x1, y1, x2, y2 = det["bbox"]
        padded = (max(0, x1 - 10), max(0, y1 - 10), min(w, x2 + 10), min(h, y2 + 10))
        cropped = image_np[padded[1]:padded[3], padded[0]:padded[2]]
        raw_text, ocr_conf, strategy = extractor.extract_text(cropped)
        parsed = parse_uml_class(raw_text)
        classes.append({
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "detection_confidence": det["confidence"],
            "class_name": parsed["class_name"],
            "attributes": parsed["attributes"],
            "methods": parsed["methods"],
            "raw_text": raw_text,
            "ocr_confidence": ocr_conf,
        })

    relationships = match_arrows_to_classes(arrows, classes, max_distance=match_dist, image=image_np)

    graph = nx.DiGraph()
    bbox_to_node = {}
    for i, cls in enumerate(classes):
        nid = f"class_{i}"
        cls["_node_id"] = nid
        graph.add_node(nid, name=cls["class_name"], type="class",
                       attributes=cls["attributes"], methods=cls["methods"],
                       bbox=cls["bbox"],
                       detection_confidence=cls["detection_confidence"],
                       ocr_confidence=cls["ocr_confidence"])
        key = (cls["bbox"]["x1"], cls["bbox"]["y1"], cls["bbox"]["x2"], cls["bbox"]["y2"])
        bbox_to_node[key] = nid

    for rel in relationships:
        sb, tb = rel["source_bbox"], rel["target_bbox"]
        sn = bbox_to_node.get((sb["x1"], sb["y1"], sb["x2"], sb["y2"]))
        tn = bbox_to_node.get((tb["x1"], tb["y1"], tb["x2"], tb["y2"]))
        if sn and tn and sn != tn:
            graph.add_edge(sn, tn,
                           relationship_type=rel.get("relationship_type", "association"),
                           arrow_confidence=rel.get("arrow_confidence", 0),
                           match_confidence=rel.get("match_confidence", 0))

    annotated = image_np.copy()
    for cls in classes:
        b = cls["bbox"]
        cv2.rectangle(annotated, (b["x1"], b["y1"]), (b["x2"], b["y2"]), CLASS_COLORS["class"], 2)
        cv2.putText(annotated, cls["class_name"][:30],
                    (b["x1"], b["y1"] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, CLASS_COLORS["class"], 1, cv2.LINE_AA)
    for arrow in arrows:
        b = arrow["bbox"]
        cv2.rectangle(annotated, (b["x1"], b["y1"]), (b["x2"], b["y2"]), CLASS_COLORS["arrow"], 2)

    # Draw traced arrow path and endpoint markers for each relationship
    ENDPOINT_COLOR = (0, 0, 255)   # red (BGR)
    TRACE_COLOR = (200, 100, 255)  # pink/magenta dashed trace
    DASH_ON = 10   # pixels of drawn dash
    DASH_OFF = 6   # pixels of gap
    for rel in relationships:
        # Draw the traced skeleton path as a dashed line
        trace_pts = rel.get("trace_points", [])
        if trace_pts and len(trace_pts) >= 2:
            pts_arr = np.array(trace_pts, dtype=np.int32)
            i = 0
            while i < len(pts_arr) - 1:
                end = min(i + DASH_ON, len(pts_arr))
                seg = pts_arr[i:end]
                if len(seg) >= 2:
                    cv2.polylines(annotated, [seg], False, TRACE_COLOR, 2, cv2.LINE_AA)
                i += DASH_ON + DASH_OFF

        # Filled circles at each endpoint
        pixel_eps = rel.get("pixel_endpoints")
        if pixel_eps is not None:
            ep_a, ep_b = pixel_eps
            cv2.circle(annotated, ep_a, 6, ENDPOINT_COLOR, -1)
            cv2.circle(annotated, ep_b, 6, ENDPOINT_COLOR, -1)

    return classes, graph, relationships, annotated, arrows


# ---------------------------------------------------------------------------
# Session state helpers for multi-conversation chat
# ---------------------------------------------------------------------------

def init_session_state():
    if "conversations" not in st.session_state:
        first_id = str(uuid.uuid4())[:8]
        st.session_state.conversations = {first_id: {"name": "Chat 1", "messages": []}}
        st.session_state.active_convo = first_id
    if "active_convo" not in st.session_state:
        st.session_state.active_convo = list(st.session_state.conversations.keys())[0]
    # Pipeline results cached across reruns
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None


def new_conversation():
    cid = str(uuid.uuid4())[:8]
    n = len(st.session_state.conversations) + 1
    st.session_state.conversations[cid] = {"name": f"Chat {n}", "messages": []}
    st.session_state.active_convo = cid


def delete_conversation(cid):
    if len(st.session_state.conversations) <= 1:
        return
    del st.session_state.conversations[cid]
    if st.session_state.active_convo == cid:
        st.session_state.active_convo = list(st.session_state.conversations.keys())[0]


# ---------------------------------------------------------------------------
# Chat fragment — reruns independently so the rest of the page stays active
# ---------------------------------------------------------------------------

@st.fragment
def _chat_fragment(graph, model_name):
    # Init streaming state
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    convo_ids = list(st.session_state.conversations.keys())
    convo_names = [st.session_state.conversations[c]["name"] for c in convo_ids]

    header_cols = st.columns([6, 1, 1])
    with header_cols[0]:
        selected_name = st.selectbox(
            "Conversation",
            convo_names,
            index=convo_ids.index(st.session_state.active_convo),
            label_visibility="collapsed",
        )
        st.session_state.active_convo = convo_ids[convo_names.index(selected_name)]
    with header_cols[1]:
        if st.button("＋", help="New conversation", use_container_width=True):
            new_conversation()
            st.rerun()
    with header_cols[2]:
        if st.button("🗑", help="Delete conversation", use_container_width=True):
            delete_conversation(st.session_state.active_convo)
            st.rerun()

    active = st.session_state.conversations[st.session_state.active_convo]

    # Chat history — sized by CSS on .st-key-chat_history
    chat_container = st.container(key="chat_history")
    with chat_container:
        if not active["messages"]:
            st.markdown(
                "_Ask a question about the uploaded UML diagram. "
                "The LLM will use the extracted classes and relationships as context._"
            )
        for msg in active["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Stop button row — only visible while streaming
    if st.session_state.is_streaming:
        if st.button("Stop generating", type="primary", use_container_width=True):
            st.session_state.stop_requested = True

    if prompt := st.chat_input("Ask about this diagram..."):
        active["messages"].append({"role": "user", "content": prompt})

        # Auto-name conversation from first question
        if len(active["messages"]) == 1:
            active["name"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

        engine = GraphRAGEngine(graph, model_name=model_name)
        st.session_state.is_streaming = True
        st.session_state.stop_requested = False

        # Show user message + streaming assistant response inside the chat box
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Thinking...")

                # Stream tokens from Ollama
                token_stream, context_text, n_nodes, n_edges = engine.query_stream(prompt)
                full_response = ""
                for token, done in token_stream:
                    if st.session_state.stop_requested:
                        full_response += "\n\n*(generation stopped)*"
                        break
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                    if done:
                        break
                placeholder.markdown(full_response)

        st.session_state.is_streaming = False
        st.session_state.stop_requested = False

        active["messages"].append({
            "role": "assistant",
            "content": full_response,
        })

        st.rerun(scope="fragment")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="UML Query Tool", layout="wide")
    init_session_state()

    # All panel heights use vh via CSS on .st-key-* classes.
    # No height= parameter on containers, so no inline styles to fight.
    st.markdown("""
    <style>
    .block-container {
        padding: 3.5rem 2.5rem 0.5rem 2.5rem;
        max-height: 100vh;
        overflow: hidden;
    }
    [data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 1.5rem; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    header[data-testid="stHeader"] { height: 1.5rem; }

    /* Diagram panel — 60% of viewport, scrollable */
    .st-key-diagram {
        height: 60vh;
        min-height: 60vh;
        max-height: 60vh;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        flex: none !important;
    }
    /* Chat history — 55% of viewport, scrollable, holds its size */
    .st-key-chat_history {
        height: 55vh;
        min-height: 55vh;
        max-height: 55vh;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        flex: none !important;
    }
    /* Bottom panels — 25% of viewport each, scrollable */
    .st-key-classes_panel {
        height: 25vh;
        min-height: 25vh;
        max-height: 25vh;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        flex: none !important;
    }
    .st-key-rels_panel {
        height: 25vh;
        min-height: 25vh;
        max-height: 25vh;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        flex: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Upload UML diagram", type=["jpg", "jpeg", "png", "bmp"])
        arrow_conf = st.slider("Arrow confidence", 0.1, 0.9, 0.3, 0.05)
        match_dist = st.slider("Match distance (px)", 50, 300, 200, 10)

        st.divider()
        st.header("Ollama LLM")
        ollama_ok = check_ollama()
        if ollama_ok:
            st.success("Ollama connected")
            models = list_ollama_models()
            model_name = st.selectbox("Model", models) if models else st.text_input("Model name", "llama3")
        else:
            st.error("Ollama not reachable — run `ollama serve`")
            model_name = st.text_input("Model name", "llama3")

        st.divider()
        st.caption(f"Device: **{DEVICE}**")

    # ---- Check model ----
    if not MODEL_PATH.exists():
        st.error(f"Model not found at `{MODEL_PATH}`. Run **Notebook 02** first.")
        return

    extractor = load_extractor(str(MODEL_PATH), gpu=(DEVICE == "cuda"))

    # ---- Process uploaded image (cache in session state) ----
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_np is not None:
            current_name = uploaded.name
            prev = st.session_state.pipeline_results
            if prev is None or prev.get("filename") != current_name:
                with st.spinner("Running detection + OCR pipeline..."):
                    classes, graph, rels, annotated, arrows = run_pipeline(
                        image_np, extractor, arrow_conf, match_dist
                    )
                st.session_state.pipeline_results = {
                    "filename": current_name,
                    "classes": classes,
                    "graph": graph,
                    "relationships": rels,
                    "annotated": annotated,
                    "arrows": arrows,
                }

    results = st.session_state.pipeline_results

    if results is None:
        st.info("Upload a UML diagram image in the sidebar to get started.")
        return

    # ---- TOP ROW: Diagram (left) | Chat (right) ----
    top_left, top_right = st.columns([1, 1])

    with top_left:
        annotated_rgb = cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB)
        with st.container(key="diagram"):
            st.image(annotated_rgb, width="stretch")

    with top_right:
        _chat_fragment(results["graph"], model_name)

    # ---- BOTTOM ROW: Classes (left) | Relationships (right) ----
    bot_left, bot_right = st.columns(2)

    with bot_left:
        st.markdown("**Classes**")
        with st.container(key="classes_panel"):
            for cls in results["classes"]:
                name = cls["class_name"] or "(unnamed)"
                with st.expander(f"{name}  —  {cls['detection_confidence']:.0%}"):
                    if cls["attributes"]:
                        st.markdown("**Attributes**")
                        for a in cls["attributes"]:
                            st.text(f"  {a.get('visibility','public')} {a['name']}: {a['type']}")
                    if cls["methods"]:
                        st.markdown("**Methods**")
                        for m in cls["methods"]:
                            params = ", ".join(
                                f"{p['name']}: {p['type']}" for p in m.get("parameters", [])
                            )
                            st.text(f"  {m.get('visibility','public')} {m['name']}({params}): {m['return_type']}")
                    if not cls["attributes"] and not cls["methods"]:
                        st.text(f"Raw: {cls['raw_text'][:120]}")

    with bot_right:
        st.markdown("**Relationships**")
        with st.container(key="rels_panel"):
            if results["relationships"]:
                for r in results["relationships"]:
                    st.markdown(
                        f"{r['source_class'][:35]} → {r['target_class'][:35]}  \n"
                        f"`{r.get('relationship_type', 'association')}`  "
                        f"_{r.get('match_confidence', 0):.0%}_"
                    )
            else:
                st.info("No relationships detected.")


if __name__ == "__main__":
    main()
