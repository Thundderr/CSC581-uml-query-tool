"""
Knowledge Graph Builder for UML Class Diagrams.

This module provides the UMLKnowledgeGraphBuilder class that orchestrates
the construction of a knowledge graph from OCR extraction results.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import networkx as nx

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .arrow_matching import (
    detect_arrows,
    match_arrows_to_classes,
    extract_relationships_from_image
)


class UMLKnowledgeGraphBuilder:
    """
    Builds a knowledge graph from UML class diagrams.

    Pipeline:
    1. Load OCR results (classes with attributes/methods)
    2. Detect arrows using YOLO for each image
    3. Match arrows to source/target classes
    4. Construct NetworkX directed graph

    Attributes:
        detector: Trained YOLO model for arrow detection
        arrow_confidence: Minimum confidence for arrow detections
        match_distance: Maximum distance for arrow-class matching
        graph: The constructed NetworkX DiGraph
    """

    def __init__(self,
                 model_path: str,
                 arrow_confidence: float = 0.3,
                 match_distance: float = 100.0):
        """
        Initialize the graph builder.

        Args:
            model_path: Path to trained YOLO model weights
            arrow_confidence: Minimum confidence for arrow detections
            match_distance: Maximum distance for arrow-class matching

        Raises:
            ImportError: If ultralytics is not installed
            FileNotFoundError: If model_path doesn't exist
        """
        if YOLO is None:
            raise ImportError("ultralytics is required: pip install ultralytics")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.detector = YOLO(str(model_path))
        self.arrow_confidence = arrow_confidence
        self.match_distance = match_distance
        self.graph = nx.DiGraph()

        # Statistics
        self.stats = {
            'images_processed': 0,
            'total_classes': 0,
            'total_arrows_detected': 0,
            'total_relationships': 0,
            'failed_images': []
        }

    def build_from_ocr_results(self,
                                ocr_results: List[Dict],
                                verbose: bool = True) -> nx.DiGraph:
        """
        Build complete knowledge graph from OCR extraction results.

        Args:
            ocr_results: List of OCR extraction results (from Notebook 03)
            verbose: Whether to print progress

        Returns:
            NetworkX DiGraph with classes as nodes and relationships as edges
        """
        self.graph = nx.DiGraph()
        self.stats = {
            'images_processed': 0,
            'total_classes': 0,
            'total_arrows_detected': 0,
            'total_relationships': 0,
            'failed_images': []
        }

        for i, result in enumerate(ocr_results):
            if verbose:
                img_name = Path(result.get('image_path', f'image_{i}')).name
                print(f"Processing {i + 1}/{len(ocr_results)}: {img_name}", end='\r')

            self._process_image(result)

        if verbose:
            print(f"\nBuilt graph with {self.graph.number_of_nodes()} nodes "
                  f"and {self.graph.number_of_edges()} edges")

        return self.graph

    def _process_image(self, ocr_result: Dict) -> None:
        """
        Process a single image to extract classes and relationships.

        Args:
            ocr_result: OCR extraction result for one image
        """
        image_path = ocr_result.get('image_path', '')
        classes = ocr_result.get('classes', [])

        if not classes:
            return

        # Add class nodes first
        for cls in classes:
            self._add_class_node(cls, image_path)

        self.stats['total_classes'] += len(classes)

        # Try to load image for arrow detection
        image = cv2.imread(image_path) if image_path else None

        if image is not None:
            # Detect arrows
            arrows = detect_arrows(self.detector, image, self.arrow_confidence)
            self.stats['total_arrows_detected'] += len(arrows)

            # Match arrows to classes
            relationships = match_arrows_to_classes(
                arrows, classes, self.match_distance
            )

            # Add relationship edges
            for rel in relationships:
                self._add_relationship_edge(rel, image_path)

            self.stats['total_relationships'] += len(relationships)
        else:
            self.stats['failed_images'].append(image_path)

        self.stats['images_processed'] += 1

    def _add_class_node(self, cls: Dict, source_image: str) -> str:
        """
        Add a class as a node in the graph.

        Args:
            cls: Class dictionary from OCR extraction
            source_image: Path to source image

        Returns:
            Node ID for the added class
        """
        node_id = self._make_node_id(cls.get('class_name', 'Unknown'), source_image)

        self.graph.add_node(
            node_id,
            name=cls.get('class_name', 'Unknown'),
            type='class',
            attributes=cls.get('attributes', []),
            methods=cls.get('methods', []),
            bbox=cls.get('bbox', {}),
            source_image=source_image,
            detection_confidence=cls.get('detection_confidence', 0.0),
            ocr_confidence=cls.get('ocr_confidence', 0.0),
            attribute_count=len(cls.get('attributes', [])),
            method_count=len(cls.get('methods', []))
        )

        return node_id

    def _add_relationship_edge(self, rel: Dict, source_image: str) -> bool:
        """
        Add a relationship as an edge in the graph.

        Args:
            rel: Relationship dictionary from arrow matching
            source_image: Path to source image

        Returns:
            True if edge was added, False otherwise
        """
        source_id = self._make_node_id(rel['source_class'], source_image)
        target_id = self._make_node_id(rel['target_class'], source_image)

        # Only add edge if both nodes exist
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(
                source_id,
                target_id,
                relationship_type=rel.get('relationship_type', 'association'),
                arrow_bbox=rel.get('arrow_bbox', {}),
                arrow_confidence=rel.get('arrow_confidence', 0.0),
                match_confidence=rel.get('match_confidence', 0.0),
                source_image=source_image
            )
            return True

        return False

    def _make_node_id(self, class_name: str, source_image: str) -> str:
        """
        Create unique node ID from class name and source image.

        Uses a hash of the source image to ensure classes from different
        images don't collide (same class name could appear in multiple diagrams).

        Args:
            class_name: Name of the class
            source_image: Path to source image

        Returns:
            Unique node ID string
        """
        # Sanitize class name (remove non-alphanumeric chars)
        safe_name = re.sub(r'[^\w]', '_', class_name or 'Unknown')

        # Include image hash for uniqueness across images
        img_hash = hashlib.md5(source_image.encode()).hexdigest()[:8]

        return f"{safe_name}_{img_hash}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the built graph.

        Returns:
            Dictionary with graph statistics
        """
        if self.graph.number_of_nodes() == 0:
            return self.stats

        # Basic counts
        stats = {
            **self.stats,
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
        }

        # Connected components
        undirected = self.graph.to_undirected()
        stats['connected_components'] = nx.number_connected_components(undirected)

        # Relationship type counts
        rel_types = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relationship_type', 'unknown')
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        stats['relationship_types'] = rel_types

        # Degree statistics
        degrees = [d for _, d in self.graph.degree()]
        if degrees:
            stats['avg_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['isolated_nodes'] = sum(1 for d in degrees if d == 0)

        return stats

    def get_subgraph_for_image(self, image_path: str) -> nx.DiGraph:
        """
        Get the subgraph containing only nodes from a specific image.

        Args:
            image_path: Path to the source image

        Returns:
            Subgraph containing only nodes from that image
        """
        nodes = [
            n for n, data in self.graph.nodes(data=True)
            if data.get('source_image') == image_path
        ]
        return self.graph.subgraph(nodes).copy()

    def find_class_by_name(self, class_name: str) -> List[str]:
        """
        Find all node IDs matching a class name.

        Args:
            class_name: Class name to search for

        Returns:
            List of matching node IDs
        """
        return [
            n for n, data in self.graph.nodes(data=True)
            if data.get('name', '').lower() == class_name.lower()
        ]

    def get_relationships_for_class(self, node_id: str) -> Dict[str, List]:
        """
        Get all relationships involving a specific class.

        Args:
            node_id: Node ID of the class

        Returns:
            Dictionary with 'outgoing' and 'incoming' relationship lists
        """
        if node_id not in self.graph:
            return {'outgoing': [], 'incoming': []}

        outgoing = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            target_data = self.graph.nodes[target]
            outgoing.append({
                'target': target_data.get('name', 'Unknown'),
                'relationship_type': data.get('relationship_type', 'unknown'),
                'confidence': data.get('match_confidence', 0.0)
            })

        incoming = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            source_data = self.graph.nodes[source]
            incoming.append({
                'source': source_data.get('name', 'Unknown'),
                'relationship_type': data.get('relationship_type', 'unknown'),
                'confidence': data.get('match_confidence', 0.0)
            })

        return {'outgoing': outgoing, 'incoming': incoming}
