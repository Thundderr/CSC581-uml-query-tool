"""
Arrow detection and class matching algorithms for UML diagrams.

This module provides functions for detecting arrows in UML diagrams
and matching them to source/target classes based on spatial proximity.
Uses skeletonization to trace the exact arrow line path and find true endpoints.
"""

from collections import deque
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


# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------

def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeletonization — reduce binary mask to 1-pixel-wide lines."""
    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask.copy()
    for _ in range(max(mask.shape) * 2):  # safety limit
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def _find_skeleton_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Find endpoints of a skeleton — pixels with exactly 1 neighbor (8-connected)."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    skel_binary = (skeleton > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel)
    endpoint_mask = (skel_binary > 0) & (neighbor_count == 1)
    coords = np.column_stack(np.where(endpoint_mask))  # (row, col)
    return [(int(c), int(r)) for r, c in coords]  # convert to (x, y)


def _trace_skeleton_path(
    skeleton: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """BFS along skeleton pixels from *start* to *end*. Returns ordered (x, y) path."""
    sx, sy = start
    ex, ey = end
    rows, cols = skeleton.shape

    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}
    queue = deque([(sx, sy)])

    found = False
    while queue:
        cx, cy = queue.popleft()
        if cx == ex and cy == ey:
            found = True
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= ny < rows and 0 <= nx < cols and (nx, ny) not in parent and skeleton[ny, nx] > 0:
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    if not found:
        return [start, end]

    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = (ex, ey)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Main pixel-based endpoint + trace detection
# ---------------------------------------------------------------------------

def find_arrow_endpoints_from_pixels(
    image: np.ndarray,
    arrow_bbox: Dict[str, int],
    class_bboxes: Optional[List[Dict[str, int]]] = None,
    color_diff_threshold: float = 30.0,
    min_foreground_pixels: int = 10,
) -> Optional[Dict]:
    """
    Trace the arrow line within its bounding box using pixel analysis.

    Pipeline:
      1. Crop the arrow bbox region.
      2. Estimate background colour from the centre 50 % of the crop.
      3. Build foreground mask via colour distance.
      4. Mask out known class-box regions (removes box borders that
         overlap with the arrow bbox).
      5. Connected-component analysis — pick the component nearest the
         crop centre (= the arrow line).
      6. Skeletonize to a 1-pixel-wide path.
      7. Find skeleton endpoints and BFS-trace the path between them.
      8. Convert to full-image coordinates.
      9. Validate that endpoints straddle a bbox midline.

    Args:
        image: Full BGR uint8 image
        arrow_bbox: Arrow bounding box dict with x1, y1, x2, y2
        class_bboxes: Optional list of class box bbox dicts to mask out
        color_diff_threshold: Min colour distance to be considered foreground
        min_foreground_pixels: Minimum foreground pixels required

    Returns:
        Dict with keys ``endpoint_a``, ``endpoint_b`` (each ``(x, y)``),
        and ``trace_points`` (ordered list of ``(x, y)`` skeleton pixels
        in original image coordinates), or ``None``.
    """
    x1, y1, x2, y2 = arrow_bbox['x1'], arrow_bbox['y1'], arrow_bbox['x2'], arrow_bbox['y2']
    h_img, w_img = image.shape[:2]

    # Clamp to image bounds
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)

    crop = image[y1c:y2c, x1c:x2c]
    ch, cw = crop.shape[:2]
    if ch < 5 or cw < 5:
        return None

    cy, cx = ch // 2, cw // 2

    # --- Step 1: Background from centre 50 % of crop ---
    q1y, q3y = ch // 4, 3 * ch // 4
    q1x, q3x = cw // 4, 3 * cw // 4
    center_region = crop[q1y:q3y, q1x:q3x].reshape(-1, 3)
    bg_color = np.median(center_region, axis=0).astype(np.float32)

    # --- Step 2: Foreground mask via colour distance ---
    diff = crop.astype(np.float32) - bg_color[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    fg_mask = (dist >= color_diff_threshold).astype(np.uint8) * 255

    # --- Step 3: Morphological cleanup ---
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

    # --- Step 4: Mask out known class-box regions ---
    # This surgically removes class box borders that overlap with the
    # arrow bbox, leaving only the arrow line pixels.  We pad the class
    # bbox by a few pixels to also catch border pixels at the box edge.
    if class_bboxes:
        pad = 4
        for cb in class_bboxes:
            cb_x1 = max(0, cb['x1'] - pad - x1c)
            cb_y1 = max(0, cb['y1'] - pad - y1c)
            cb_x2 = min(cw, cb['x2'] + pad - x1c)
            cb_y2 = min(ch, cb['y2'] + pad - y1c)
            if cb_x1 < cb_x2 and cb_y1 < cb_y2:
                fg_mask[cb_y1:cb_y2, cb_x1:cb_x2] = 0

    # --- Step 5: Connected component nearest to centre ---
    num_labels, labels = cv2.connectedComponents(fg_mask)
    if num_labels <= 1:
        return None

    best_label = -1
    best_center_dist = float('inf')
    for label in range(1, num_labels):
        component_mask = (labels == label)
        component_coords = np.column_stack(np.where(component_mask))
        if len(component_coords) < min_foreground_pixels:
            continue
        dists_to_center = np.sqrt(
            (component_coords[:, 0] - cy) ** 2 + (component_coords[:, 1] - cx) ** 2
        )
        min_dist = dists_to_center.min()
        if min_dist < best_center_dist:
            best_center_dist = min_dist
            best_label = label

    if best_label < 0:
        return None

    arrow_mask = (labels == best_label).astype(np.uint8) * 255
    if cv2.countNonZero(arrow_mask) < min_foreground_pixels:
        return None

    # --- Step 6: Skeletonize ---
    skeleton = _skeletonize(arrow_mask)
    if cv2.countNonZero(skeleton) < 2:
        return None

    # --- Step 7: Find skeleton endpoints & trace path ---
    skel_eps = _find_skeleton_endpoints(skeleton)

    if len(skel_eps) >= 2:
        # Pick the two most-distant skeleton endpoints
        if len(skel_eps) == 2:
            ep_a_local, ep_b_local = skel_eps[0], skel_eps[1]
        else:
            best_d = 0
            ep_a_local, ep_b_local = skel_eps[0], skel_eps[1]
            for i in range(len(skel_eps)):
                for j in range(i + 1, len(skel_eps)):
                    dx = skel_eps[i][0] - skel_eps[j][0]
                    dy = skel_eps[i][1] - skel_eps[j][1]
                    d = dx * dx + dy * dy
                    if d > best_d:
                        best_d = d
                        ep_a_local, ep_b_local = skel_eps[i], skel_eps[j]
    else:
        # Fallback: convex hull max-distance pair on skeleton pixels
        skel_coords = np.column_stack(np.where(skeleton > 0))[:, ::-1].astype(np.int32)
        if len(skel_coords) < 2:
            return None
        hull = cv2.convexHull(skel_coords.reshape(-1, 1, 2))
        hull_pts = hull.reshape(-1, 2)
        if len(hull_pts) < 2:
            return None
        best_d = 0
        pt_a, pt_b = hull_pts[0], hull_pts[-1]
        for i in range(len(hull_pts)):
            for j in range(i + 1, len(hull_pts)):
                dx = int(hull_pts[i][0]) - int(hull_pts[j][0])
                dy = int(hull_pts[i][1]) - int(hull_pts[j][1])
                d = dx * dx + dy * dy
                if d > best_d:
                    best_d = d
                    pt_a, pt_b = hull_pts[i], hull_pts[j]
        ep_a_local = (int(pt_a[0]), int(pt_a[1]))
        ep_b_local = (int(pt_b[0]), int(pt_b[1]))

    trace_local = _trace_skeleton_path(skeleton, ep_a_local, ep_b_local)

    # If skeleton was disconnected (morphological thinning can create gaps),
    # retry the trace on a slightly dilated version to bridge 1-2px gaps.
    if len(trace_local) <= 2:
        bridge = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
        trace_local = _trace_skeleton_path(bridge, ep_a_local, ep_b_local)

    # --- Step 8: Convert to original image coordinates ---
    endpoint_a = (ep_a_local[0] + x1c, ep_a_local[1] + y1c)
    endpoint_b = (ep_b_local[0] + x1c, ep_b_local[1] + y1c)
    trace_points = [(x + x1c, y + y1c) for x, y in trace_local]

    # --- Step 9: Midline validation ---
    x_mid = (x1 + x2) / 2.0
    y_mid = (y1 + y2) / 2.0
    crosses_vertical = (endpoint_a[0] - x_mid) * (endpoint_b[0] - x_mid) <= 0
    crosses_horizontal = (endpoint_a[1] - y_mid) * (endpoint_b[1] - y_mid) <= 0
    if not crosses_vertical and not crosses_horizontal:
        return None

    return {
        'endpoint_a': endpoint_a,
        'endpoint_b': endpoint_b,
        'trace_points': trace_points,
    }


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


def bbox_to_bbox_distance(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
    """
    Calculate the minimum distance between two bounding boxes.

    Returns 0 if the boxes overlap.

    Args:
        bbox1: Dictionary with x1, y1, x2, y2 coordinates
        bbox2: Dictionary with x1, y1, x2, y2 coordinates

    Returns:
        Minimum distance between the two bounding boxes (0 if overlapping)
    """
    dx = max(bbox1['x1'] - bbox2['x2'], 0, bbox2['x1'] - bbox1['x2'])
    dy = max(bbox1['y1'] - bbox2['y2'], 0, bbox2['y1'] - bbox1['y2'])
    return np.sqrt(dx * dx + dy * dy)


def determine_direction(arrow_bbox: Dict[str, int],
                        cls1: Dict, cls2: Dict,
                        pixel_endpoints: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
                        ) -> Tuple[Dict, Dict]:
    """
    Determine which class is source and which is target for an arrow.

    Uses the estimated arrow head/tail positions as a heuristic.
    The class closer to the estimated head is the target.

    When pixel_endpoints are provided, first assigns each endpoint to the
    nearer class, then uses the bbox-based head heuristic to decide direction.

    Args:
        arrow_bbox: Arrow bounding box dict
        cls1: First class dict (one of the two closest classes)
        cls2: Second class dict (the other closest class)
        pixel_endpoints: Optional tuple of two (x, y) pixel-detected endpoints

    Returns:
        Tuple of (source_class, target_class)
    """
    endpoints = estimate_arrow_endpoints(arrow_bbox)

    if pixel_endpoints is not None:
        ep_a, ep_b = pixel_endpoints

        # Assign each endpoint to the nearer class (minimize total distance)
        dist_a_cls1 = distance_to_bbox_edge(ep_a, cls1['bbox'])
        dist_a_cls2 = distance_to_bbox_edge(ep_a, cls2['bbox'])
        dist_b_cls1 = distance_to_bbox_edge(ep_b, cls1['bbox'])
        dist_b_cls2 = distance_to_bbox_edge(ep_b, cls2['bbox'])

        if (dist_a_cls1 + dist_b_cls2) <= (dist_a_cls2 + dist_b_cls1):
            cls1_ep, cls2_ep = ep_a, ep_b
        else:
            cls1_ep, cls2_ep = ep_b, ep_a

        # Which endpoint is closer to the estimated head? That class is target.
        d_cls1ep_head = np.sqrt(
            (cls1_ep[0] - endpoints.head[0]) ** 2 + (cls1_ep[1] - endpoints.head[1]) ** 2
        )
        d_cls2ep_head = np.sqrt(
            (cls2_ep[0] - endpoints.head[0]) ** 2 + (cls2_ep[1] - endpoints.head[1]) ** 2
        )

        if d_cls1ep_head <= d_cls2ep_head:
            return cls2, cls1  # cls1 is target (its endpoint closer to head)
        else:
            return cls1, cls2  # cls2 is target

    # Fallback: use class centers vs estimated head
    cls1_cx = (cls1['bbox']['x1'] + cls1['bbox']['x2']) / 2
    cls1_cy = (cls1['bbox']['y1'] + cls1['bbox']['y2']) / 2
    cls2_cx = (cls2['bbox']['x1'] + cls2['bbox']['x2']) / 2
    cls2_cy = (cls2['bbox']['y1'] + cls2['bbox']['y2']) / 2

    dist_cls1_to_head = np.sqrt(
        (cls1_cx - endpoints.head[0]) ** 2 + (cls1_cy - endpoints.head[1]) ** 2
    )
    dist_cls2_to_head = np.sqrt(
        (cls2_cx - endpoints.head[0]) ** 2 + (cls2_cy - endpoints.head[1]) ** 2
    )

    if dist_cls1_to_head <= dist_cls2_to_head:
        return cls2, cls1  # cls2=source, cls1=target (cls1 closer to head)
    else:
        return cls1, cls2  # cls1=source, cls2=target (cls2 closer to head)


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
    from src.utils.device import get_device
    results = detector(image, device=get_device(), verbose=False)

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
                            max_distance: float = 200.0,
                            image: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Match detected arrows to source and target classes using pixel analysis.

    Analyzes arrow pixels to find true endpoints, then matches each endpoint
    to the nearest class within max_distance. No bbox-based fallback — only
    pixel-verified matches are accepted.

    Args:
        arrows: List of arrow detections with bbox
        classes: List of class extractions with bbox
        max_distance: Maximum distance from endpoint to class edge
        image: BGR image for pixel-based endpoint detection (required)

    Returns:
        List of relationship dicts with source, target, and metadata
    """
    PIXEL_MAX_DIST = max_distance
    relationships = []

    for arrow in arrows:
        if image is None:
            continue

        class_bboxes = [cls['bbox'] for cls in classes]
        pixel_result = find_arrow_endpoints_from_pixels(
            image, arrow['bbox'], class_bboxes=class_bboxes
        )
        if pixel_result is None:
            continue

        ep_a = pixel_result['endpoint_a']
        ep_b = pixel_result['endpoint_b']
        trace_pts = pixel_result['trace_points']

        cls_a = find_closest_class(ep_a, classes, max_distance=PIXEL_MAX_DIST)
        cls_b = find_closest_class(ep_b, classes, max_distance=PIXEL_MAX_DIST)

        if cls_a is None or cls_b is None or cls_a is cls_b:
            continue

        pixel_eps = (ep_a, ep_b)
        source, target = determine_direction(
            arrow['bbox'], cls_a, cls_b, pixel_endpoints=pixel_eps
        )
        dist_a = distance_to_bbox_edge(ep_a, cls_a['bbox'])
        dist_b = distance_to_bbox_edge(ep_b, cls_b['bbox'])
        match_conf = 1.0 - (dist_a + dist_b) / (2 * PIXEL_MAX_DIST)
        match_conf = max(0.0, min(1.0, match_conf))

        endpoints = estimate_arrow_endpoints(arrow['bbox'])
        relationships.append({
            'source_class': source.get('class_name', 'Unknown'),
            'target_class': target.get('class_name', 'Unknown'),
            'source_bbox': source['bbox'],
            'target_bbox': target['bbox'],
            'arrow_bbox': arrow['bbox'],
            'arrow_confidence': arrow['confidence'],
            'arrow_orientation': endpoints.orientation,
            'relationship_type': infer_relationship_type(endpoints),
            'match_confidence': match_conf,
            'match_method': 'pixel',
            'pixel_endpoints': pixel_eps,
            'trace_points': trace_pts,
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
                                     max_distance: float = 200.0) -> List[Dict]:
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
    relationships = match_arrows_to_classes(arrows, classes, max_distance, image=image)

    return relationships
