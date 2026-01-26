"""
Arrow detection and class matching algorithms for UML diagrams.

This module provides functions for detecting arrows in UML diagrams
and matching them to source/target classes based on spatial proximity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Class IDs from YOLO model
ARROW_ID = 0
CLASS_BOX_ID = 1
CROSS_ID = 2


@dataclass
class ArrowEndpoints:
    """Estimated endpoints of an arrow from its bounding box."""
    head: Tuple[float, float]  # (x, y) - arrowhead position (target)
    tail: Tuple[float, float]  # (x, y) - arrow origin (source)
    confidence: float
    orientation: str  # 'horizontal', 'vertical', 'diagonal'


def estimate_arrow_endpoints(bbox: Dict[str, int]) -> ArrowEndpoints:
    """
    Estimate arrow head and tail positions from bounding box.

    Since YOLO only provides bounding boxes, we estimate endpoints
    based on the box geometry and typical UML arrow patterns.

    Args:
        bbox: Dictionary with x1, y1, x2, y2 coordinates

    Returns:
        ArrowEndpoints with estimated head and tail positions
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    width = x2 - x1
    height = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # Determine orientation based on aspect ratio
    if width > height * 1.5:
        # Horizontal arrow: endpoints at left and right
        orientation = 'horizontal'
        # Assume left-to-right direction (most common)
        tail = (x1, cy)
        head = (x2, cy)
    elif height > width * 1.5:
        # Vertical arrow: endpoints at top and bottom
        orientation = 'vertical'
        # In UML, inheritance arrows typically point UP (subclass at bottom)
        tail = (cx, y2)  # bottom is tail (source/subclass)
        head = (cx, y1)  # top is head (target/superclass)
    else:
        # Diagonal or square arrow
        orientation = 'diagonal'
        # Assume bottom-left to top-right
        tail = (x1, y2)
        head = (x2, y1)

    return ArrowEndpoints(
        head=head,
        tail=tail,
        confidence=1.0,
        orientation=orientation
    )


def distance_to_bbox_edge(point: Tuple[float, float],
                          bbox: Dict[str, int]) -> float:
    """
    Calculate distance from a point to the nearest edge of a bounding box.

    If the point is inside the box, returns 0.

    Args:
        point: (x, y) coordinate
        bbox: Dictionary with x1, y1, x2, y2 coordinates

    Returns:
        Distance to nearest edge (0 if inside)
    """
    px, py = point
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

    # Calculate distance to each edge
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)

    return np.sqrt(dx * dx + dy * dy)


def distance_to_bbox_center(point: Tuple[float, float],
                            bbox: Dict[str, int]) -> float:
    """
    Calculate distance from a point to the center of a bounding box.

    Args:
        point: (x, y) coordinate
        bbox: Dictionary with x1, y1, x2, y2 coordinates

    Returns:
        Distance to center
    """
    px, py = point
    cx = (bbox['x1'] + bbox['x2']) / 2
    cy = (bbox['y1'] + bbox['y2']) / 2

    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def find_closest_class(point: Tuple[float, float],
                       classes: List[Dict],
                       max_distance: float = 100.0) -> Optional[Dict]:
    """
    Find the class box closest to a given point.

    Uses distance to class box edge (not center) for more accurate matching,
    since arrows typically connect to the edges of class boxes.

    Args:
        point: (x, y) coordinate
        classes: List of class dicts with 'bbox' key
        max_distance: Maximum distance to consider a match

    Returns:
        Closest class dict or None if no match within threshold
    """
    best_class = None
    best_distance = float('inf')

    for cls in classes:
        bbox = cls['bbox']
        distance = distance_to_bbox_edge(point, bbox)

        if distance < best_distance and distance <= max_distance:
            best_distance = distance
            best_class = cls

    return best_class


def detect_arrows(detector: Any,
                  image: np.ndarray,
                  confidence_threshold: float = 0.3) -> List[Dict]:
    """
    Detect arrows in the image using YOLO.

    Args:
        detector: Trained YOLO model
        image: BGR image
        confidence_threshold: Minimum confidence for detections

    Returns:
        List of arrow detections with bbox and confidence
    """
    results = detector(image, device='cpu', verbose=False)

    arrows = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            if cls_id == ARROW_ID and conf >= confidence_threshold:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                arrows.append({
                    'bbox': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2)
                    },
                    'confidence': conf
                })

    return arrows


def match_arrows_to_classes(arrows: List[Dict],
                            classes: List[Dict],
                            max_distance: float = 100.0) -> List[Dict]:
    """
    Match detected arrows to source and target classes.

    For each arrow, estimates the head (target) and tail (source) positions,
    then finds the closest class to each endpoint.

    Args:
        arrows: List of arrow detections with bbox
        classes: List of class extractions with bbox
        max_distance: Maximum distance for endpoint-to-class matching

    Returns:
        List of relationship dicts with source, target, and metadata
    """
    relationships = []

    for arrow in arrows:
        # Estimate endpoints from bounding box
        endpoints = estimate_arrow_endpoints(arrow['bbox'])

        # Find source class (near tail)
        source = find_closest_class(endpoints.tail, classes, max_distance)

        # Find target class (near head)
        target = find_closest_class(endpoints.head, classes, max_distance)

        # Only create relationship if both endpoints match different classes
        if source and target and source != target:
            # Calculate match confidence based on distance
            source_dist = distance_to_bbox_edge(endpoints.tail, source['bbox'])
            target_dist = distance_to_bbox_edge(endpoints.head, target['bbox'])

            match_conf = 1.0 - (source_dist + target_dist) / (2 * max_distance)
            match_conf = max(0.0, min(1.0, match_conf))

            relationships.append({
                'source_class': source.get('class_name', 'Unknown'),
                'target_class': target.get('class_name', 'Unknown'),
                'source_bbox': source['bbox'],
                'target_bbox': target['bbox'],
                'arrow_bbox': arrow['bbox'],
                'arrow_confidence': arrow['confidence'],
                'arrow_orientation': endpoints.orientation,
                'relationship_type': infer_relationship_type(endpoints),
                'match_confidence': match_conf
            })

    return relationships


def infer_relationship_type(endpoints: ArrowEndpoints) -> str:
    """
    Infer UML relationship type based on arrow orientation.

    This is a heuristic based on common UML conventions:
    - Vertical arrows pointing up often indicate inheritance
    - Horizontal arrows often indicate association

    Note: Without analyzing the actual arrow head style (filled diamond,
    empty triangle, etc.), we can only make educated guesses.

    Args:
        endpoints: Arrow endpoint information

    Returns:
        Relationship type string
    """
    if endpoints.orientation == 'vertical':
        # Vertical arrows often indicate inheritance in UML
        # (subclass at bottom, superclass at top)
        return 'inheritance'
    else:
        # Default to association for horizontal/diagonal
        return 'association'


def extract_relationships_from_image(detector: Any,
                                     image_path: str,
                                     classes: List[Dict],
                                     arrow_confidence: float = 0.3,
                                     max_distance: float = 100.0) -> List[Dict]:
    """
    Complete pipeline to extract relationships from a single image.

    Args:
        detector: Trained YOLO model
        image_path: Path to UML diagram image
        classes: List of class extractions (from OCR) with bboxes
        arrow_confidence: Minimum confidence for arrow detection
        max_distance: Maximum distance for arrow-class matching

    Returns:
        List of extracted relationships
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    # Detect arrows
    arrows = detect_arrows(detector, image, arrow_confidence)

    # Match to classes
    relationships = match_arrows_to_classes(arrows, classes, max_distance)

    return relationships
