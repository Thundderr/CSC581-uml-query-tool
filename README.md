# CSC581 UML Query Tool

A tool for analyzing UML class diagrams using computer vision, OCR, and knowledge graph querying. Upload a UML diagram and ask questions about classes, relationships, and software architecture.

## Project Overview

This project builds a 5-stage pipeline to make UML diagrams queryable:

1. **Object Detection**: Use YOLOv8/SAM 2 to detect and segment classes, arrows, and relationships
2. **OCR Extraction**: Extract text from detected class boxes (names, attributes, methods)
3. **Knowledge Graph**: Convert extracted information into a queryable graph structure
4. **GraphRAG Query**: Enable natural language queries about the UML diagrams
5. **Interactive UI**: Streamlit/Gradio interface for uploading and querying diagrams

## Dataset

We use the [UML Class Diagram Dataset](https://www.kaggle.com/datasets/domenicoarm/uml-class-diagram-dataset-bounded-box-rating) from Kaggle:

- ~650 UML class diagram images (JPG/PNG)
- 134 MB total size
- Annotations in YOLO (.txt) and Faster R-CNN (.xml) formats
- Three annotation classes: Classes, Arrows, Crosses
- Quality ratings for each diagram

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

## Quick Start

```bash
# Launch Jupyter Lab
jupyter lab

# Open notebooks/01_dataset_exploration.ipynb to get started
```

## Project Structure

```
CSC581-uml-query-tool/
├── notebooks/              # Jupyter notebooks for each pipeline stage
│   └── 01_dataset_exploration.ipynb
├── src/                    # Source code modules
│   ├── detection/          # Object detection (YOLOv8, SAM)
│   ├── ocr/                # Text extraction
│   ├── graph/              # Knowledge graph construction
│   ├── query/              # GraphRAG implementation
│   └── utils/              # Shared utilities
├── data/                   # Dataset (git-ignored)
│   ├── raw/                # Downloaded dataset
│   ├── processed/          # Processed outputs
│   └── sample/             # Small sample for testing
├── models/                 # Trained models (git-ignored)
├── configs/                # Configuration files
└── ui/                     # User interface
```

## Pipeline Stages

### Stage 1: Object Detection
Detect bounding boxes for classes and arrows in UML diagrams using YOLOv8 or SAM 2.

### Stage 2: OCR Extraction
Read text content from detected class boxes using Tesseract or EasyOCR.

### Stage 3: Knowledge Graph
Build a graph representation of classes and their relationships.

### Stage 4: GraphRAG Query
Query the knowledge graph using natural language with an LLM.

### Stage 5: User Interface
Web-based interface for uploading diagrams and asking questions.

## License

This project is for educational purposes (CSC581). The dataset is licensed under CC BY-NC-SA 4.0.
