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
# Core: extract endpoints from a foreground mask
# ---------------------------------------------------------------------------

def _extract_endpoints_from_mask(
    fg_mask: np.ndarray,
    class_bboxes: Optional[List[Dict[str, int]]],
    x1c: int, y1c: int,
    ch: int, cw: int,
    min_foreground_pixels: int,
) -> Optional[Dict]:
    """
    Given a binary foreground mask (crop-local), mask class boxes, select
    the best component, skeletonize, and return endpoints + trace.

    Returns dict with crop-local ``ep_a``, ``ep_b``, ``trace`` keys, or None.
    """
    cy, cx = ch // 2, cw // 2

    # Mask out class-box regions
    if class_bboxes:
        fg_before = fg_mask.copy()
        for pad in (2, 0):
            fg_mask = fg_before.copy()
            for cb in class_bboxes:
                cb_x1 = max(0, cb['x1'] - pad - x1c)
                cb_y1 = max(0, cb['y1'] - pad - y1c)
                cb_x2 = min(cw, cb['x2'] + pad - x1c)
                cb_y2 = min(ch, cb['y2'] + pad - y1c)
                if cb_x1 < cb_x2 and cb_y1 < cb_y2:
                    fg_mask[cb_y1:cb_y2, cb_x1:cb_x2] = 0
            if cv2.countNonZero(fg_mask) >= min_foreground_pixels:
                break
            if pad == 0 and cv2.countNonZero(fg_mask) < min_foreground_pixels:
                fg_mask = fg_before

    # Bridge dashed lines if fragmented
    num_labels_pre, _ = cv2.connectedComponents(fg_mask)
    if num_labels_pre > 3:
        fg_mask = cv2.dilate(
            fg_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    # Line filter: open along BOTH axes and union results to remove text
    # blobs while preserving L-shaped arrows.  A single-direction kernel
    # would destroy the perpendicular leg of an L-shaped arrow.
    # After the union, close small gaps to reconnect corners where both
    # openings eroded the turn of an L-shape.
    fg_total = cv2.countNonZero(fg_mask)
    if fg_total > max(cw, ch) * 3 and cw >= 20 and ch >= 20:
        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, cw // 15), 1))
        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, ch // 15)))
        fg_h = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, h_kern)
        fg_v = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, v_kern)
        fg_filtered = cv2.bitwise_or(fg_h, fg_v)
        # Bridge the corner gap: each opening erodes ~half-kernel pixels
        # at L-shaped turns.  The gap is diagonal so use Pythagorean distance.
        h_half = h_kern.shape[1] // 2
        v_half = v_kern.shape[0] // 2
        gap = int(np.sqrt(h_half ** 2 + v_half ** 2)) + 2
        close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
        fg_filtered = cv2.morphologyEx(fg_filtered, cv2.MORPH_CLOSE, close_kern)
        if cv2.countNonZero(fg_filtered) >= min_foreground_pixels:
            fg_mask = fg_filtered

    # Largest connected component near centre, then merge nearby components
    # to reassemble L-shaped arrows whose legs were split by the line filter.
    num_labels, labels = cv2.connectedComponents(fg_mask)
    if num_labels <= 1:
        return None

    center_radius = max(ch, cw) * 0.4
    best_label = -1
    best_size = 0
    for label in range(1, num_labels):
        coords = np.column_stack(np.where(labels == label))
        if len(coords) < min_foreground_pixels:
            continue
        dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
        if dists.min() > center_radius:
            continue
        if len(coords) > best_size:
            best_size = len(coords)
            best_label = label

    if best_label < 0:
        return None

    arrow_mask = (labels == best_label).astype(np.uint8) * 255

    # Merge nearby components: dilate the selected component and absorb
    # any other *significant* component it touches.  This reconnects legs
    # of L-shaped arrows that were split at the corner by the line filter.
    # Only merge components that are at least 25% of the main component's
    # size to avoid absorbing small text/noise blobs.
    merge_radius = max(ch, cw) // 8
    merge_min_size = max(min_foreground_pixels, best_size // 4)
    merged_any = False
    if merge_radius > 0 and num_labels > 2:
        merge_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * merge_radius + 1, 2 * merge_radius + 1))
        dilated = cv2.dilate(arrow_mask, merge_kern)
        for label in range(1, num_labels):
            if label == best_label:
                continue
            comp = (labels == label).astype(np.uint8) * 255
            if cv2.countNonZero(comp) < merge_min_size:
                continue
            if cv2.countNonZero(cv2.bitwise_and(dilated, comp)) > 0:
                arrow_mask = cv2.bitwise_or(arrow_mask, comp)
                merged_any = True

    # If we merged disconnected components, bridge the gaps between them
    # so the skeleton will be a single connected curve.  Use progressive
    # dilation to find the smallest bridge, avoiding over-thickening.
    if merged_any:
        n_comp, _ = cv2.connectedComponents(arrow_mask)
        if n_comp > 2:  # more than one component
            for r in range(1, merge_radius + 1):
                bridged = cv2.dilate(
                    arrow_mask, np.ones((3, 3), np.uint8), iterations=r)
                n_comp2, _ = cv2.connectedComponents(bridged)
                if n_comp2 <= 2:
                    arrow_mask = bridged
                    break

    if cv2.countNonZero(arrow_mask) < min_foreground_pixels:
        return None

    # Skeletonize
    skeleton = _skeletonize(arrow_mask)
    if cv2.countNonZero(skeleton) < 2:
        return None

    # Endpoints via convex hull of skeleton pixels (most-distant pair)
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
    ep_a = (int(pt_a[0]), int(pt_a[1]))
    ep_b = (int(pt_b[0]), int(pt_b[1]))

    # Trace path — progressively dilate the skeleton to bridge micro-gaps
    # left by skeletonization of thick/merged masks.
    trace = _trace_skeleton_path(skeleton, ep_a, ep_b)
    if len(trace) <= 2:
        for bridge_iter in range(1, 6):
            bridge = cv2.dilate(skeleton, np.ones((3, 3), np.uint8),
                                iterations=bridge_iter)
            trace = _trace_skeleton_path(bridge, ep_a, ep_b)
            if len(trace) > 2:
                break

    return {'ep_a': ep_a, 'ep_b': ep_b, 'trace': trace}


# ---------------------------------------------------------------------------
# Main pixel-based endpoint + trace detection
# ---------------------------------------------------------------------------

def find_arrow_endpoints_from_pixels(
    image: np.ndarray,
    arrow_bbox: Dict[str, int],
    class_bboxes: Optional[List[Dict[str, int]]] = None,
    color_diff_threshold: float = 50.0,
    min_foreground_pixels: int = 15,
) -> Optional[Dict]:
    """
    Trace the arrow line within its bounding box using pixel analysis.

    Tries two detection strategies:
      1. **Colour distance** — estimate background, threshold, skeletonize.
      2. **Edge detection** (Canny) — fallback for low-contrast or noisy images.

    Args:
        image: Full BGR uint8 image
        arrow_bbox: Arrow bounding box dict with x1, y1, x2, y2
        class_bboxes: Optional list of class box bbox dicts to mask out
        color_diff_threshold: Min colour distance to be considered foreground
        min_foreground_pixels: Minimum foreground pixels required

    Returns:
        Dict with ``endpoint_a``, ``endpoint_b`` (each ``(x, y)``), and
        ``trace_points`` (ordered skeleton pixels in image coords), or None.
    """
    x1, y1, x2, y2 = arrow_bbox['x1'], arrow_bbox['y1'], arrow_bbox['x2'], arrow_bbox['y2']
    h_img, w_img = image.shape[:2]

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)

    crop = image[y1c:y2c, x1c:x2c]
    ch, cw = crop.shape[:2]
    if ch < 5 or cw < 5:
        return None

    # ---- Strategy 1: Colour distance (try primary + alternate thresholds) ----
    q1y, q3y = ch // 4, 3 * ch // 4
    q1x, q3x = cw // 4, 3 * cw // 4
    center_region = crop[q1y:q3y, q1x:q3x].reshape(-1, 3)
    bg_color = np.median(center_region, axis=0).astype(np.float32)

    diff = crop.astype(np.float32) - bg_color[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    result = None
    for thresh in (color_diff_threshold, color_diff_threshold * 0.5, color_diff_threshold * 1.5):
        fg_color = (dist >= thresh).astype(np.uint8) * 255
        fg_color = cv2.morphologyEx(fg_color, cv2.MORPH_CLOSE, kernel_close)
        fg_color = cv2.morphologyEx(fg_color, cv2.MORPH_OPEN, kernel_open)
        result = _extract_endpoints_from_mask(
            fg_color, class_bboxes, x1c, y1c, ch, cw, min_foreground_pixels,
        )
        if result is not None:
            break

    # Fallback strategies use a lower pixel threshold since edge/adaptive
    # methods produce thinner foreground.
    fallback_min_fg = max(5, min_foreground_pixels // 2)

    # Grayscale needed for fallback strategies
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ---- Strategy 2: Edge detection (Canny) fallback ----
    if result is None:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        result = _extract_endpoints_from_mask(
            edges, class_bboxes, x1c, y1c, ch, cw, fallback_min_fg,
        )

    # ---- Strategy 3: Adaptive threshold fallback ----
    if result is None:
        block_size = max(11, (min(ch, cw) // 4) | 1)  # ensure odd
        fg_adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 5,
        )
        result = _extract_endpoints_from_mask(
            fg_adapt, class_bboxes, x1c, y1c, ch, cw, fallback_min_fg,
        )

    # ---- Strategy 4: Otsu threshold fallback ----
    if result is None:
        _, fg_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        result = _extract_endpoints_from_mask(
            fg_otsu, class_bboxes, x1c, y1c, ch, cw, fallback_min_fg,
        )

    if result is None:
        return None

    # Convert to image coordinates
    ep_a = (result['ep_a'][0] + x1c, result['ep_a'][1] + y1c)
    ep_b = (result['ep_b'][0] + x1c, result['ep_b'][1] + y1c)
    trace = [(x + x1c, y + y1c) for x, y in result['trace']]

    return {
        'endpoint_a': ep_a,
        'endpoint_b': ep_b,
        'trace_points': trace,
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
