"""
Comprehensive arrow tracing evaluation on real UML diagrams.

Uses ground-truth YOLO annotations to test find_arrow_endpoints_from_pixels()
on real arrow bounding boxes from the dataset.

Usage:
    python tests/test_real_arrows.py                    # default params, quick eval
    python tests/test_real_arrows.py --sweep            # full parameter sweep
    python tests/test_real_arrows.py --debug            # save debug images for failures
    python tests/test_real_arrows.py --threshold 25     # test specific threshold
    python tests/test_real_arrows.py --max-images 20    # limit images for quick check
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import (
    download_dataset,
    load_yolo_split,
    parse_yolo_annotation,
)
from src.graph.arrow_matching import (
    find_arrow_endpoints_from_pixels,
    _skeletonize,
    _find_skeleton_endpoints,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARROW_CLASS_ID = 0
CLASS_BOX_ID = 1

DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_output"

SWEEP_THRESHOLDS = [15, 20, 25, 30, 35, 40, 50]
SWEEP_MIN_FG = [5, 10, 15, 20]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_images(max_images: int = 100) -> List[Dict]:
    """
    Load up to *max_images* image+annotation pairs from test and valid splits.

    Returns list of dicts:
        {
            'image_path': Path,
            'arrows': [{'x1': int, 'y1': int, 'x2': int, 'y2': int}, ...],
            'class_boxes': [{'x1': int, 'y1': int, 'x2': int, 'y2': int}, ...],
        }
    """
    dataset_path = download_dataset()
    pairs: List[Dict] = []

    # Test split first (63 images), then valid (61 images)
    for split in ['test', 'valid']:
        split_pairs = load_yolo_split(dataset_path, split)
        for p in split_pairs:
            if len(pairs) >= max_images:
                break
            img = cv2.imread(str(p['image']))
            if img is None:
                continue
            h, w = img.shape[:2]
            annotations = parse_yolo_annotation(p['label'], w, h)

            arrows = []
            class_boxes = []
            for a in annotations:
                bbox = {
                    'x1': int(round(a['x1'])),
                    'y1': int(round(a['y1'])),
                    'x2': int(round(a['x2'])),
                    'y2': int(round(a['y2'])),
                }
                if a['class_id'] == ARROW_CLASS_ID:
                    arrows.append(bbox)
                elif a['class_id'] == CLASS_BOX_ID:
                    class_boxes.append(bbox)

            if arrows:  # only include images that have arrows
                pairs.append({
                    'image_path': p['image'],
                    'arrows': arrows,
                    'class_boxes': class_boxes,
                })
        if len(pairs) >= max_images:
            break

    return pairs[:max_images]


# ---------------------------------------------------------------------------
# Failure diagnosis
# ---------------------------------------------------------------------------

def diagnose_failure(
    image: np.ndarray,
    arrow_bbox: Dict[str, int],
    class_bboxes: List[Dict[str, int]],
    color_diff_threshold: float,
    min_foreground_pixels: int,
) -> str:
    """
    Re-run pipeline steps to categorize why tracing failed.
    Returns a failure category string.
    """
    x1, y1, x2, y2 = arrow_bbox['x1'], arrow_bbox['y1'], arrow_bbox['x2'], arrow_bbox['y2']
    h_img, w_img = image.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)

    crop = image[y1c:y2c, x1c:x2c]
    ch, cw = crop.shape[:2]
    if ch < 5 or cw < 5:
        return 'bbox_too_small'

    # Background estimation
    q1y, q3y = ch // 4, 3 * ch // 4
    q1x, q3x = cw // 4, 3 * cw // 4
    center_region = crop[q1y:q3y, q1x:q3x].reshape(-1, 3)
    if len(center_region) == 0:
        return 'bbox_too_small'
    bg_color = np.median(center_region, axis=0).astype(np.float32)

    # Foreground mask
    diff = crop.astype(np.float32) - bg_color[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    fg_mask = (dist >= color_diff_threshold).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

    fg_before_masking = cv2.countNonZero(fg_mask)

    # Mask out class boxes
    if class_bboxes:
        pad = 4
        for cb in class_bboxes:
            cb_x1 = max(0, cb['x1'] - pad - x1c)
            cb_y1 = max(0, cb['y1'] - pad - y1c)
            cb_x2 = min(cw, cb['x2'] + pad - x1c)
            cb_y2 = min(ch, cb['y2'] + pad - y1c)
            if cb_x1 < cb_x2 and cb_y1 < cb_y2:
                fg_mask[cb_y1:cb_y2, cb_x1:cb_x2] = 0

    fg_after_masking = cv2.countNonZero(fg_mask)

    if fg_before_masking < min_foreground_pixels:
        return 'no_foreground'
    if fg_after_masking < min_foreground_pixels and fg_before_masking >= min_foreground_pixels:
        return 'over_masked'
    if fg_after_masking < min_foreground_pixels:
        return 'no_foreground'

    # Connected component
    num_labels, labels = cv2.connectedComponents(fg_mask)
    if num_labels <= 1:
        return 'no_foreground'

    cy_c, cx_c = ch // 2, cw // 2
    best_label = -1
    best_center_dist = float('inf')
    for label in range(1, num_labels):
        component_mask = (labels == label)
        component_coords = np.column_stack(np.where(component_mask))
        if len(component_coords) < min_foreground_pixels:
            continue
        dists_to_center = np.sqrt(
            (component_coords[:, 0] - cy_c) ** 2 + (component_coords[:, 1] - cx_c) ** 2
        )
        min_dist = dists_to_center.min()
        if min_dist < best_center_dist:
            best_center_dist = min_dist
            best_label = label

    if best_label < 0:
        return 'no_foreground'

    arrow_mask = (labels == best_label).astype(np.uint8) * 255

    # Skeleton
    skeleton = _skeletonize(arrow_mask)
    if cv2.countNonZero(skeleton) < 2:
        return 'no_skeleton_endpoints'

    skel_eps = _find_skeleton_endpoints(skeleton)
    if len(skel_eps) < 2:
        # Check convex hull fallback
        skel_coords = np.column_stack(np.where(skeleton > 0))[:, ::-1].astype(np.int32)
        if len(skel_coords) < 2:
            return 'no_skeleton_endpoints'
        hull = cv2.convexHull(skel_coords.reshape(-1, 1, 2))
        hull_pts = hull.reshape(-1, 2)
        if len(hull_pts) < 2:
            return 'no_skeleton_endpoints'
        # If we got here, hull gave us endpoints but the main function still failed
        # — likely midline validation
        return 'midline_validation_failed'

    # If skeleton endpoints exist, the failure is likely midline validation
    return 'midline_validation_failed'


# ---------------------------------------------------------------------------
# Debug image output
# ---------------------------------------------------------------------------

def save_debug_image(
    image: np.ndarray,
    arrow_bbox: Dict[str, int],
    class_bboxes: List[Dict[str, int]],
    color_diff_threshold: float,
    output_path: Path,
):
    """Save a debug visualization for a failed arrow trace."""
    x1, y1, x2, y2 = arrow_bbox['x1'], arrow_bbox['y1'], arrow_bbox['x2'], arrow_bbox['y2']
    h_img, w_img = image.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)

    crop = image[y1c:y2c, x1c:x2c]
    ch, cw = crop.shape[:2]
    if ch < 5 or cw < 5:
        return

    # Resize crop for visibility (min 100px tall)
    scale = max(1, 100 // max(ch, 1))
    crop_resized = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_NEAREST)

    # Build foreground mask
    q1y, q3y = ch // 4, 3 * ch // 4
    q1x, q3x = cw // 4, 3 * cw // 4
    center_region = crop[q1y:q3y, q1x:q3x].reshape(-1, 3)
    if len(center_region) == 0:
        return
    bg_color = np.median(center_region, axis=0).astype(np.float32)
    diff = crop.astype(np.float32) - bg_color[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    fg_mask = (dist >= color_diff_threshold).astype(np.uint8) * 255
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

    fg_before = fg_mask.copy()

    # Mask out class boxes
    if class_bboxes:
        pad = 4
        for cb in class_bboxes:
            cb_x1 = max(0, cb['x1'] - pad - x1c)
            cb_y1 = max(0, cb['y1'] - pad - y1c)
            cb_x2 = min(cw, cb['x2'] + pad - x1c)
            cb_y2 = min(ch, cb['y2'] + pad - y1c)
            if cb_x1 < cb_x2 and cb_y1 < cb_y2:
                fg_mask[cb_y1:cb_y2, cb_x1:cb_x2] = 0

    fg_after = fg_mask.copy()

    # Skeleton
    skeleton = _skeletonize(fg_after)
    skel_eps = _find_skeleton_endpoints(skeleton)

    # Build debug panels
    panels = []

    # Panel 1: Original crop with class boxes outlined
    p1 = crop.copy()
    if class_bboxes:
        for cb in class_bboxes:
            cb_x1 = max(0, cb['x1'] - x1c)
            cb_y1 = max(0, cb['y1'] - y1c)
            cb_x2 = min(cw, cb['x2'] - x1c)
            cb_y2 = min(ch, cb['y2'] - y1c)
            cv2.rectangle(p1, (cb_x1, cb_y1), (cb_x2, cb_y2), (0, 255, 0), 1)
    panels.append(p1)

    # Panel 2: FG mask before class masking
    panels.append(cv2.cvtColor(fg_before, cv2.COLOR_GRAY2BGR))

    # Panel 3: FG mask after class masking
    panels.append(cv2.cvtColor(fg_after, cv2.COLOR_GRAY2BGR))

    # Panel 4: Skeleton with endpoints
    skel_vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for ep in skel_eps:
        cv2.circle(skel_vis, ep, 3, (0, 0, 255), -1)
    panels.append(skel_vis)

    # Resize all panels to same height
    target_h = max(p.shape[0] for p in panels)
    target_h = max(target_h, 80)  # minimum 80px
    resized_panels = []
    for p in panels:
        if p.shape[0] != target_h:
            s = target_h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * s), target_h), interpolation=cv2.INTER_NEAREST)
        resized_panels.append(p)

    debug_img = cv2.hconcat(resized_panels)
    cv2.imwrite(str(output_path), debug_img)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_trace_evaluation(
    test_images: List[Dict],
    color_diff_threshold: float = 30.0,
    min_foreground_pixels: int = 10,
    save_debug: bool = False,
    max_debug_saves: int = 50,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate arrow tracing on all test images.

    Returns a result dict with aggregate and per-image stats.
    """
    total_arrows = 0
    traced_arrows = 0
    failure_counts = defaultdict(int)
    per_image_results = []
    debug_saves = 0

    for img_data in test_images:
        image = cv2.imread(str(img_data['image_path']))
        if image is None:
            continue

        img_name = Path(img_data['image_path']).stem
        arrows = img_data['arrows']
        class_bboxes = img_data['class_boxes']

        img_total = 0
        img_traced = 0
        img_failures = []

        for i, arrow_bbox in enumerate(arrows):
            total_arrows += 1
            img_total += 1

            result = find_arrow_endpoints_from_pixels(
                image, arrow_bbox, class_bboxes=class_bboxes,
                color_diff_threshold=color_diff_threshold,
                min_foreground_pixels=min_foreground_pixels,
            )

            if result is not None and len(result.get('trace_points', [])) >= 3:
                traced_arrows += 1
                img_traced += 1
            else:
                # Diagnose failure
                category = diagnose_failure(
                    image, arrow_bbox, class_bboxes,
                    color_diff_threshold, min_foreground_pixels,
                )
                failure_counts[category] += 1
                img_failures.append({
                    'arrow_index': i,
                    'bbox': arrow_bbox,
                    'category': category,
                    'bbox_area': (arrow_bbox['x2'] - arrow_bbox['x1']) * (arrow_bbox['y2'] - arrow_bbox['y1']),
                })

                # Save debug image
                if save_debug and debug_saves < max_debug_saves:
                    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    debug_path = DEBUG_OUTPUT_DIR / f"{img_name}_arrow{i}_{category}.png"
                    save_debug_image(
                        image, arrow_bbox, class_bboxes,
                        color_diff_threshold, debug_path,
                    )
                    debug_saves += 1

        per_image_results.append({
            'image': img_name,
            'total_arrows': img_total,
            'traced_arrows': img_traced,
            'success_rate': img_traced / img_total if img_total > 0 else 1.0,
            'failures': img_failures,
        })

        if verbose:
            rate = img_traced / img_total if img_total > 0 else 1.0
            print(f"  {img_name}: {img_traced}/{img_total} ({rate:.0%})")

    success_rate = traced_arrows / total_arrows if total_arrows > 0 else 0.0

    # Breakdown by arrow area
    area_buckets = {'small': [0, 0], 'medium': [0, 0], 'large': [0, 0]}
    for img_data in test_images:
        image = cv2.imread(str(img_data['image_path']))
        if image is None:
            continue
        for arrow_bbox in img_data['arrows']:
            area = (arrow_bbox['x2'] - arrow_bbox['x1']) * (arrow_bbox['y2'] - arrow_bbox['y1'])
            if area < 500:
                bucket = 'small'
            elif area < 5000:
                bucket = 'medium'
            else:
                bucket = 'large'

            result = find_arrow_endpoints_from_pixels(
                image, arrow_bbox, class_bboxes=img_data['class_boxes'],
                color_diff_threshold=color_diff_threshold,
                min_foreground_pixels=min_foreground_pixels,
            )
            area_buckets[bucket][0] += 1  # total
            if result is not None and len(result.get('trace_points', [])) >= 3:
                area_buckets[bucket][1] += 1  # success

    area_rates = {}
    for bucket, (total, success) in area_buckets.items():
        area_rates[bucket] = {
            'total': total,
            'success': success,
            'rate': success / total if total > 0 else 0.0,
        }

    return {
        'total_arrows': total_arrows,
        'traced_arrows': traced_arrows,
        'success_rate': success_rate,
        'failure_distribution': dict(failure_counts),
        'area_breakdown': area_rates,
        'per_image_results': per_image_results,
        'params': {
            'color_diff_threshold': color_diff_threshold,
            'min_foreground_pixels': min_foreground_pixels,
        },
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(
    test_images: List[Dict],
    thresholds: List[float] = None,
    min_fgs: List[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Grid search over color_diff_threshold and min_foreground_pixels.
    Returns dict with all results and best params.
    """
    if thresholds is None:
        thresholds = SWEEP_THRESHOLDS
    if min_fgs is None:
        min_fgs = SWEEP_MIN_FG

    results = []
    best_rate = -1.0
    best_params = {}

    total_combos = len(thresholds) * len(min_fgs)
    combo_idx = 0

    for thresh in thresholds:
        for min_fg in min_fgs:
            combo_idx += 1
            if verbose:
                print(f"\n[{combo_idx}/{total_combos}] threshold={thresh}, min_fg={min_fg}")

            result = run_trace_evaluation(
                test_images,
                color_diff_threshold=thresh,
                min_foreground_pixels=min_fg,
                verbose=False,
            )

            results.append({
                'color_diff_threshold': thresh,
                'min_foreground_pixels': min_fg,
                'success_rate': result['success_rate'],
                'total_arrows': result['total_arrows'],
                'traced_arrows': result['traced_arrows'],
                'failure_distribution': result['failure_distribution'],
                'area_breakdown': result['area_breakdown'],
            })

            if verbose:
                print(f"  => {result['traced_arrows']}/{result['total_arrows']} "
                      f"({result['success_rate']:.1%})")
                for cat, cnt in sorted(result['failure_distribution'].items()):
                    print(f"     {cat}: {cnt}")

            if result['success_rate'] > best_rate:
                best_rate = result['success_rate']
                best_params = {
                    'color_diff_threshold': thresh,
                    'min_foreground_pixels': min_fg,
                }

    return {
        'sweep_results': results,
        'best_params': best_params,
        'best_success_rate': best_rate,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(result: Dict):
    """Print a formatted console report."""
    print("\n" + "=" * 60)
    print("ARROW TRACING EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal arrows:  {result['total_arrows']}")
    print(f"Traced arrows: {result['traced_arrows']}")
    print(f"Success rate:  {result['success_rate']:.1%}")

    print(f"\nParameters:")
    for k, v in result['params'].items():
        print(f"  {k}: {v}")

    if result['failure_distribution']:
        print(f"\nFailure distribution:")
        for cat, cnt in sorted(result['failure_distribution'].items(),
                               key=lambda x: -x[1]):
            pct = cnt / result['total_arrows'] * 100 if result['total_arrows'] > 0 else 0
            print(f"  {cat:30s}: {cnt:4d} ({pct:.1f}%)")

    print(f"\nArea breakdown:")
    for bucket, stats in result['area_breakdown'].items():
        rate_str = f"{stats['rate']:.1%}" if stats['total'] > 0 else "N/A"
        print(f"  {bucket:8s}: {stats['success']}/{stats['total']} ({rate_str})")

    # Worst images
    worst = sorted(result['per_image_results'],
                   key=lambda x: x['success_rate'])[:5]
    if worst:
        print(f"\nWorst 5 images:")
        for img in worst:
            print(f"  {img['image']}: {img['traced_arrows']}/{img['total_arrows']} "
                  f"({img['success_rate']:.0%})")

    print("=" * 60)


def print_sweep_report(sweep: Dict):
    """Print a formatted sweep report."""
    print("\n" + "=" * 60)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 60)

    # Table header
    print(f"\n{'Threshold':>10s} {'MinFG':>6s} {'Rate':>8s} {'Traced':>8s} {'Total':>8s}")
    print("-" * 46)

    for r in sorted(sweep['sweep_results'], key=lambda x: -x['success_rate']):
        marker = " <-- BEST" if (
            r['color_diff_threshold'] == sweep['best_params']['color_diff_threshold'] and
            r['min_foreground_pixels'] == sweep['best_params']['min_foreground_pixels']
        ) else ""
        print(f"{r['color_diff_threshold']:>10.0f} {r['min_foreground_pixels']:>6d} "
              f"{r['success_rate']:>7.1%} {r['traced_arrows']:>8d} {r['total_arrows']:>8d}{marker}")

    print(f"\nBest parameters: {sweep['best_params']}")
    print(f"Best success rate: {sweep['best_success_rate']:.1%}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Arrow tracing evaluation on real UML diagrams")
    parser.add_argument('--max-images', type=int, default=100,
                        help='Max number of images to load (default: 100)')
    parser.add_argument('--threshold', type=float, default=50.0,
                        help='Color diff threshold (default: 50.0)')
    parser.add_argument('--min-fg', type=int, default=15,
                        help='Min foreground pixels (default: 15)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run full parameter sweep')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images for failures')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose per-image output')
    parser.add_argument('--output', type=str, default='tests/real_arrow_results.json',
                        help='JSON output path')
    args = parser.parse_args()

    print(f"Loading up to {args.max_images} test images...")
    t0 = time.time()
    test_images = load_test_images(args.max_images)
    total_arrows = sum(len(img['arrows']) for img in test_images)
    print(f"Loaded {len(test_images)} images with {total_arrows} arrow bboxes "
          f"in {time.time() - t0:.1f}s")

    if args.sweep:
        print("\nStarting parameter sweep...")
        t0 = time.time()
        sweep = parameter_sweep(test_images, verbose=True)
        print(f"\nSweep completed in {time.time() - t0:.1f}s")
        print_sweep_report(sweep)

        # Run detailed eval with best params + debug
        print(f"\nRunning detailed evaluation with best params: {sweep['best_params']}")
        result = run_trace_evaluation(
            test_images,
            color_diff_threshold=sweep['best_params']['color_diff_threshold'],
            min_foreground_pixels=sweep['best_params']['min_foreground_pixels'],
            save_debug=args.debug,
            verbose=args.verbose,
        )
        print_report(result)

        # Save full results
        output = {
            'sweep': sweep,
            'best_detailed_result': {
                k: v for k, v in result.items() if k != 'per_image_results'
            },
            'per_image_results': result['per_image_results'],
        }
    else:
        print(f"\nEvaluating with threshold={args.threshold}, min_fg={args.min_fg}...")
        t0 = time.time()
        result = run_trace_evaluation(
            test_images,
            color_diff_threshold=args.threshold,
            min_foreground_pixels=args.min_fg,
            save_debug=args.debug,
            verbose=args.verbose,
        )
        print(f"Evaluation completed in {time.time() - t0:.1f}s")
        print_report(result)
        output = result

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
