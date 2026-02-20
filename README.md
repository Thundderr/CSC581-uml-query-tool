# CSC581 UML Query Tool

A tool for analyzing UML class diagrams using computer vision, OCR, and knowledge graph querying. Upload a UML diagram and ask questions about classes, relationships, and software architecture.

## Requirements

- **Python 3.12**
- **Ollama** (for LLM queries) — [Install Ollama](https://ollama.com/download)
- **NVIDIA GPU** (optional, but recommended) — CUDA-capable GPU for faster training and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CSC581-uml-query-tool.git
cd CSC581-uml-query-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Setup (CUDA)

The default `pip install torch` gives you **CPU-only** PyTorch. For NVIDIA GPU support:

```bash
# Windows/Linux with NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Mac users with Apple Silicon get MPS acceleration automatically with the default install.

Verify your GPU is detected:

```python
import torch
print(torch.cuda.is_available())       # True for NVIDIA GPU
print(torch.backends.mps.is_available()) # True for Apple Silicon
```

### Ollama Setup

Install Ollama from [ollama.com](https://ollama.com/download), then pull a model:

```bash
ollama pull llama3
ollama serve  # start the server (may already be running)
```

## Quick Start

### Streamlit App (recommended)

Upload any UML diagram image and query it with natural language:

```bash
streamlit run app.py
```

### Jupyter Notebooks

Run the pipeline step by step:

```bash
jupyter lab
# Open notebooks/01_dataset_exploration.ipynb to get started
```

## Pipeline

1. **Object Detection** (Notebook 02) — YOLOv8 detects class boxes, arrows, and crosses
2. **OCR Extraction** (Notebook 03) — EasyOCR extracts text from detected class boxes
3. **Knowledge Graph** (Notebook 04) — NetworkX graph built from classes and arrow relationships
4. **GraphRAG Query** (`app.py`) — Natural language queries via Ollama LLM over the graph

## Project Structure

```
CSC581-uml-query-tool/
├── app.py                  # Streamlit GUI — upload & query UML diagrams
├── notebooks/              # Jupyter notebooks for each pipeline stage
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_object_detection.ipynb
│   ├── 03_ocr_extraction.ipynb
│   └── 04_knowledge_graph.ipynb
├── src/                    # Source code modules
│   ├── detection/          # Object detection
│   ├── ocr/                # Text extraction (EasyOCR + PyTesseract)
│   ├── graph/              # Knowledge graph construction & arrow matching
│   ├── query/              # GraphRAG engine (Ollama LLM)
│   └── utils/              # Device selection, data loading
├── data/                   # Dataset & processed outputs (git-ignored)
├── models/                 # Trained YOLO weights (git-ignored)
├── configs/                # Configuration files
└── requirements.txt
```

## Dataset

[UML Class Diagram Dataset](https://www.kaggle.com/datasets/domenicoarm/uml-class-diagram-dataset-bounded-box-rating) from Kaggle:

- ~650 UML class diagram images
- Annotations in YOLO and PASCAL VOC formats
- Three classes: arrow, class, cross
- Quality ratings for each diagram

## License

This project is for educational purposes (CSC581). The dataset is licensed under CC BY-NC-SA 4.0.
