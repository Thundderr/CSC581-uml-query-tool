"""
Tests for arrow line tracing algorithm.

Creates synthetic UML-like images with known arrow configurations and verifies
that the skeleton-based tracing finds correct endpoints and paths.

Run:  python -m tests.test_arrow_tracing
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.graph.arrow_matching import (
    find_arrow_endpoints_from_pixels,
    match_arrows_to_classes,
    _skeletonize,
    _find_skeleton_endpoints,
    _trace_skeleton_path,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic image with two class boxes and an arrow line
# ---------------------------------------------------------------------------

def make_test_image(
    size=(600, 400),
    box_a=((30, 80), (150, 220)),
    box_b=((400, 80), (560, 220)),
    line_points=None,
    bg_color=(255, 255, 255),
    box_color=(0, 0, 0),
    line_color=(0, 160, 0),
    line_thickness=2,
    box_thickness=2,
):
    """Create a synthetic image with two rectangles (class boxes) and a line (arrow)."""
    w, h = size
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    cv2.rectangle(img, box_a[0], box_a[1], box_color, box_thickness)
    cv2.rectangle(img, box_b[0], box_b[1], box_color, box_thickness)
    if line_points is not None and len(line_points) >= 2:
        for i in range(len(line_points) - 1):
            cv2.line(img, line_points[i], line_points[i + 1], line_color, line_thickness)
    return img


def assert_near(actual, expected, tol, label=""):
    """Assert that actual is within tol of expected."""
    dist = np.sqrt((actual[0] - expected[0]) ** 2 + (actual[1] - expected[1]) ** 2)
    assert dist <= tol, (
        f"{label}: expected ~{expected}, got {actual} (distance={dist:.1f}, tol={tol})"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_horizontal_line():
    """Straight horizontal arrow between two boxes."""
    print("Test: horizontal line ... ", end="")
    box_a = ((30, 80), (150, 220))
    box_b = ((400, 80), (560, 220))
    img = make_test_image(
        box_a=box_a, box_b=box_b,
        line_points=[(150, 150), (400, 150)],
    )
    # Arrow bbox tightly around the line (with some padding that clips box edges)
    bbox = {'x1': 140, 'y1': 135, 'x2': 410, 'y2': 165}
    class_bboxes = [
        {'x1': box_a[0][0], 'y1': box_a[0][1], 'x2': box_a[1][0], 'y2': box_a[1][1]},
        {'x1': box_b[0][0], 'y1': box_b[0][1], 'x2': box_b[1][0], 'y2': box_b[1][1]},
    ]

    result = find_arrow_endpoints_from_pixels(img, bbox, class_bboxes=class_bboxes)
    assert result is not None, "Expected a result, got None"

    ep_a, ep_b = result['endpoint_a'], result['endpoint_b']
    trace = result['trace_points']

    # Sort by x so we know which is left/right
    left, right = sorted([ep_a, ep_b], key=lambda p: p[0])

    # Endpoints should be near the line ends (within margin tolerance)
    assert_near(left, (150, 150), 30, "left endpoint")
    assert_near(right, (400, 150), 30, "right endpoint")

    # Trace should have many points (roughly the line length)
    assert len(trace) > 50, f"Trace too short: {len(trace)} points"

    # Trace should be roughly horizontal (all y values near 150)
    y_vals = [p[1] for p in trace]
    assert max(y_vals) - min(y_vals) <= 5, f"Trace not horizontal: y range = {max(y_vals) - min(y_vals)}"

    print(f"PASS  (endpoints: {left} -> {right}, trace: {len(trace)} pts)")


def test_vertical_line():
    """Straight vertical arrow between two boxes."""
    print("Test: vertical line ... ", end="")
    box_a = ((80, 20), (220, 120))
    box_b = ((80, 320), (220, 420))
    img = make_test_image(
        size=(300, 500),
        box_a=box_a, box_b=box_b,
        line_points=[(150, 120), (150, 320)],
    )
    bbox = {'x1': 135, 'y1': 110, 'x2': 165, 'y2': 330}
    class_bboxes = [
        {'x1': box_a[0][0], 'y1': box_a[0][1], 'x2': box_a[1][0], 'y2': box_a[1][1]},
        {'x1': box_b[0][0], 'y1': box_b[0][1], 'x2': box_b[1][0], 'y2': box_b[1][1]},
    ]

    result = find_arrow_endpoints_from_pixels(img, bbox, class_bboxes=class_bboxes)
    assert result is not None, "Expected a result, got None"

    ep_a, ep_b = result['endpoint_a'], result['endpoint_b']
    trace = result['trace_points']

    top, bottom = sorted([ep_a, ep_b], key=lambda p: p[1])
    assert_near(top, (150, 120), 30, "top endpoint")
    assert_near(bottom, (150, 320), 30, "bottom endpoint")

    assert len(trace) > 50, f"Trace too short: {len(trace)} points"

    x_vals = [p[0] for p in trace]
    assert max(x_vals) - min(x_vals) <= 5, f"Trace not vertical: x range = {max(x_vals) - min(x_vals)}"

    print(f"PASS  (endpoints: {top} -> {bottom}, trace: {len(trace)} pts)")


def test_l_shaped_line():
    """L-shaped arrow (horizontal then vertical)."""
    print("Test: L-shaped line ... ", end="")
    box_a = ((30, 80), (150, 220))
    box_b = ((300, 300), (460, 420))
    img = make_test_image(
        size=(600, 500),
        box_a=box_a, box_b=box_b,
        line_points=[(150, 150), (380, 150), (380, 300)],
        line_thickness=3,
    )
    # Bbox around the full L shape
    bbox = {'x1': 140, 'y1': 135, 'x2': 395, 'y2': 310}
    class_bboxes = [
        {'x1': box_a[0][0], 'y1': box_a[0][1], 'x2': box_a[1][0], 'y2': box_a[1][1]},
        {'x1': box_b[0][0], 'y1': box_b[0][1], 'x2': box_b[1][0], 'y2': box_b[1][1]},
    ]

    result = find_arrow_endpoints_from_pixels(img, bbox, class_bboxes=class_bboxes)
    assert result is not None, "Expected a result, got None"

    ep_a, ep_b = result['endpoint_a'], result['endpoint_b']
    trace = result['trace_points']

    # One endpoint near the left box exit, other near the bottom box entry
    # Sort so we know which is which
    pts = sorted([ep_a, ep_b], key=lambda p: p[0] + p[1])
    assert_near(pts[0], (150, 150), 35, "top-left endpoint")
    assert_near(pts[1], (380, 300), 35, "bottom-right endpoint")

    assert len(trace) > 100, f"Trace too short for L-shape: {len(trace)} points"

    # Trace should contain the corner — some points near (380, 150)
    corner_dists = [np.sqrt((p[0] - 380) ** 2 + (p[1] - 150) ** 2) for p in trace]
    assert min(corner_dists) < 20, f"No trace point near L-corner (380, 150); closest = {min(corner_dists):.0f}"

    print(f"PASS  (endpoints: {pts[0]} -> {pts[1]}, trace: {len(trace)} pts)")


def test_diagonal_line():
    """Diagonal arrow between two boxes."""
    print("Test: diagonal line ... ", end="")
    box_a = ((30, 30), (150, 150))
    box_b = ((400, 300), (560, 430))
    img = make_test_image(
        size=(600, 500),
        box_a=box_a, box_b=box_b,
        line_points=[(150, 150), (400, 300)],
        line_thickness=2,
    )
    bbox = {'x1': 140, 'y1': 140, 'x2': 410, 'y2': 310}
    class_bboxes = [
        {'x1': box_a[0][0], 'y1': box_a[0][1], 'x2': box_a[1][0], 'y2': box_a[1][1]},
        {'x1': box_b[0][0], 'y1': box_b[0][1], 'x2': box_b[1][0], 'y2': box_b[1][1]},
    ]

    result = find_arrow_endpoints_from_pixels(img, bbox, class_bboxes=class_bboxes)
    assert result is not None, "Expected a result, got None"

    ep_a, ep_b = result['endpoint_a'], result['endpoint_b']
    trace = result['trace_points']

    tl, br = sorted([ep_a, ep_b], key=lambda p: p[0] + p[1])
    assert_near(tl, (150, 150), 35, "top-left endpoint")
    assert_near(br, (400, 300), 35, "bottom-right endpoint")

    assert len(trace) > 50, f"Trace too short: {len(trace)} points"

    print(f"PASS  (endpoints: {tl} -> {br}, trace: {len(trace)} pts)")


def test_arrow_with_class_box_overlap():
    """Arrow bbox overlaps significantly with class box borders."""
    print("Test: arrow bbox overlaps class box ... ", end="")
    box_a = ((30, 50), (180, 250))
    box_b = ((400, 50), (560, 250))
    img = make_test_image(
        size=(600, 300),
        box_a=box_a, box_b=box_b,
        line_points=[(180, 150), (400, 150)],
        box_color=(0, 0, 0),
        line_color=(0, 160, 0),
        box_thickness=3,
        line_thickness=2,
    )
    # Arrow bbox extends 20px into each class box (overlapping the box borders)
    bbox = {'x1': 160, 'y1': 130, 'x2': 420, 'y2': 170}
    class_bboxes = [
        {'x1': box_a[0][0], 'y1': box_a[0][1], 'x2': box_a[1][0], 'y2': box_a[1][1]},
        {'x1': box_b[0][0], 'y1': box_b[0][1], 'x2': box_b[1][0], 'y2': box_b[1][1]},
    ]

    result = find_arrow_endpoints_from_pixels(img, bbox, class_bboxes=class_bboxes)
    assert result is not None, "Expected a result, got None (box overlap not handled)"

    ep_a, ep_b = result['endpoint_a'], result['endpoint_b']
    left, right = sorted([ep_a, ep_b], key=lambda p: p[0])

    # Endpoints should be on the arrow line, NOT on the class box borders
    assert_near(left, (180, 150), 35, "left endpoint (should not be on box border)")
    assert_near(right, (400, 150), 35, "right endpoint (should not be on box border)")

    print(f"PASS  (endpoints: {left} -> {right})")


def test_full_matching_pipeline():
    """Test match_arrows_to_classes with synthetic data."""
    print("Test: full matching pipeline ... ", end="")
    img = make_test_image(
        line_points=[(150, 150), (400, 150)],
    )
    arrows = [{'bbox': {'x1': 140, 'y1': 135, 'x2': 410, 'y2': 165}, 'confidence': 0.9}]
    classes = [
        {'class_name': 'ClassA', 'bbox': {'x1': 30, 'y1': 80, 'x2': 150, 'y2': 220}},
        {'class_name': 'ClassB', 'bbox': {'x1': 400, 'y1': 80, 'x2': 560, 'y2': 220}},
    ]

    rels = match_arrows_to_classes(arrows, classes, max_distance=200.0, image=img)
    assert len(rels) == 1, f"Expected 1 relationship, got {len(rels)}"

    rel = rels[0]
    names = {rel['source_class'], rel['target_class']}
    assert names == {'ClassA', 'ClassB'}, f"Wrong classes matched: {names}"
    assert rel['match_method'] == 'pixel'
    assert 'trace_points' in rel
    assert len(rel['trace_points']) > 50

    print(f"PASS  ({rel['source_class']} -> {rel['target_class']}, "
          f"trace: {len(rel['trace_points'])} pts)")


def test_skeletonize_simple():
    """Unit test for _skeletonize on a simple thick line."""
    print("Test: skeletonize thick line ... ", end="")
    mask = np.zeros((50, 200), dtype=np.uint8)
    cv2.line(mask, (10, 25), (190, 25), 255, 6)  # thick horizontal line

    skel = _skeletonize(mask)
    assert cv2.countNonZero(skel) > 0, "Skeleton is empty"

    # Skeleton should be thin — no pixel should have >2 neighbours
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    skel_bin = (skel > 0).astype(np.uint8)
    nbrs = cv2.filter2D(skel_bin, -1, kernel)
    max_nbrs = nbrs[skel > 0].max()
    # Allow up to 3 neighbours at turns/junctions
    assert max_nbrs <= 3, f"Skeleton has a pixel with {max_nbrs} neighbours — not thin"

    endpoints = _find_skeleton_endpoints(skel)
    assert len(endpoints) >= 2, f"Expected >=2 endpoints, got {len(endpoints)}"

    # Endpoints should be near (10, 25) and (190, 25)
    left, right = sorted(endpoints, key=lambda p: p[0])[:2]  # take first two by x
    if len(endpoints) > 2:
        # Take the two most distant
        best_d = 0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                dx = endpoints[i][0] - endpoints[j][0]
                dy = endpoints[i][1] - endpoints[j][1]
                d = dx * dx + dy * dy
                if d > best_d:
                    best_d = d
                    left, right = endpoints[i], endpoints[j]
        left, right = sorted([left, right], key=lambda p: p[0])

    assert abs(left[0] - 10) < 5, f"Left endpoint x={left[0]}, expected ~10"
    assert abs(right[0] - 190) < 5, f"Right endpoint x={right[0]}, expected ~190"

    print(f"PASS  (skeleton pixels: {cv2.countNonZero(skel)}, endpoints: {left}, {right})")


def test_trace_path_l_shape():
    """Unit test for _trace_skeleton_path on an L-shaped skeleton."""
    print("Test: trace path L-shape ... ", end="")
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Horizontal segment then vertical
    cv2.line(mask, (10, 30), (150, 30), 255, 1)
    cv2.line(mask, (150, 30), (150, 90), 255, 1)

    # Already 1px wide, but skeletonize to be safe
    skel = _skeletonize(mask)
    if cv2.countNonZero(skel) < 2:
        skel = mask  # if already thin, use original

    endpoints = _find_skeleton_endpoints(skel)
    assert len(endpoints) >= 2, f"Expected >=2 endpoints, got {len(endpoints)}"

    # Find the two most distant
    best_d = 0
    ep_a, ep_b = endpoints[0], endpoints[-1]
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dx = endpoints[i][0] - endpoints[j][0]
            dy = endpoints[i][1] - endpoints[j][1]
            d = dx * dx + dy * dy
            if d > best_d:
                best_d = d
                ep_a, ep_b = endpoints[i], endpoints[j]

    path = _trace_skeleton_path(skel, ep_a, ep_b)
    assert len(path) > 50, f"Path too short: {len(path)} points"

    # Path should include the corner area
    corner_dists = [np.sqrt((p[0] - 150) ** 2 + (p[1] - 30) ** 2) for p in path]
    assert min(corner_dists) < 10, f"Path doesn't pass through corner; closest = {min(corner_dists):.0f}"

    print(f"PASS  (path length: {len(path)}, corner closest: {min(corner_dists):.0f}px)")


def test_different_arrow_colors():
    """Arrows of different colors on white background should all be detected."""
    print("Test: different arrow colors ... ", end="")
    box_a = ((30, 80), (150, 220))
    box_b = ((400, 80), (560, 220))
    class_bboxes = [
        {'x1': box_a[0][0], 'y1': box_a[0][1], 'x2': box_a[1][0], 'y2': box_a[1][1]},
        {'x1': box_b[0][0], 'y1': box_b[0][1], 'x2': box_b[1][0], 'y2': box_b[1][1]},
    ]
    colors = [
        ((0, 160, 0), "green"),
        ((0, 0, 0), "black"),
        ((200, 0, 0), "blue"),
        ((0, 0, 200), "red"),
        ((128, 0, 128), "purple"),
    ]
    results = []
    for color, name in colors:
        img = make_test_image(
            box_a=box_a, box_b=box_b,
            line_points=[(150, 150), (400, 150)], line_color=color,
        )
        bbox = {'x1': 140, 'y1': 135, 'x2': 410, 'y2': 165}
        r = find_arrow_endpoints_from_pixels(img, bbox, class_bboxes=class_bboxes)
        results.append((name, r is not None))
        if r is None:
            print(f"\n  WARNING: {name} arrow not detected")

    passed = sum(1 for _, ok in results if ok)
    assert passed >= 4, f"Only {passed}/{len(colors)} colors detected"
    print(f"PASS  ({passed}/{len(colors)} colors detected)")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Arrow Tracing Algorithm Tests")
    print("=" * 60)
    print()

    tests = [
        test_skeletonize_simple,
        test_trace_path_l_shape,
        test_horizontal_line,
        test_vertical_line,
        test_l_shaped_line,
        test_diagonal_line,
        test_arrow_with_class_box_overlap,
        test_full_matching_pipeline,
        test_different_arrow_colors,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"FAIL  ({e})")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
