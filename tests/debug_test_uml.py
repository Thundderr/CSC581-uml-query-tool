"""Quick test: run find_arrow_endpoints_from_pixels on each arrow and report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
from src.graph.arrow_matching import find_arrow_endpoints_from_pixels, ARROW_ID, CLASS_BOX_ID
from src.utils.device import get_device

IMAGE_PATH = Path(r"C:\Users\Aiden Smith\Downloads\test_uml.png")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "uml_detector_best.pt"
OUTPUT_DIR = Path(__file__).resolve().parent / "debug_output"

image = cv2.imread(str(IMAGE_PATH))
h_img, w_img = image.shape[:2]
model = YOLO(str(MODEL_PATH))
results = model(image, device=get_device(), verbose=False)

arrows, class_boxes = [], []
for r in results:
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].item())
        conf = float(r.boxes.conf[i].item())
        x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[i].tolist()]
        bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        if cls_id == ARROW_ID:
            arrows.append({'bbox': bbox, 'conf': conf})
        elif cls_id == CLASS_BOX_ID:
            class_boxes.append(bbox)

arrows_main = [a for a in arrows if a['conf'] >= 0.3]
print(f"{len(arrows_main)} arrows (conf >= 0.3)\n")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Draw all results on full image
vis = image.copy()

for idx, arrow in enumerate(arrows_main):
    ab = arrow['bbox']
    x1c, y1c = max(0, ab['x1']), max(0, ab['y1'])
    print(f"Arrow {idx}: ({ab['x1']},{ab['y1']})-({ab['x2']},{ab['y2']})  "
          f"{ab['x2']-ab['x1']}x{ab['y2']-ab['y1']}  conf={arrow['conf']:.2f}")
    try:
        result = find_arrow_endpoints_from_pixels(image, ab, class_bboxes=class_boxes)
        if result:
            trace = result['trace_points']
            ep_a, ep_b = result['endpoint_a'], result['endpoint_b']
            if len(trace) >= 3:
                p0 = np.array(trace[0], dtype=float)
                p1 = np.array(trace[-1], dtype=float)
                lv = p1 - p0
                ll = np.linalg.norm(lv)
                max_dev = 0
                if ll > 0:
                    ld = lv / ll
                    for pt in trace:
                        v = np.array(pt, dtype=float) - p0
                        perp = v - np.dot(v, ld) * ld
                        max_dev = max(max_dev, np.linalg.norm(perp))
                status = "L-SHAPE" if max_dev > 10 else "straight"
                print(f"  {status}: {ep_a} -> {ep_b}, {len(trace)} pts, dev={max_dev:.1f}px")

                # Draw trace on full image
                color = (0, 0, 255)  # red
                for i in range(len(trace) - 1):
                    cv2.line(vis, trace[i], trace[i+1], color, 2)
                cv2.circle(vis, ep_a, 6, (255, 0, 0), -1)
                cv2.circle(vis, ep_b, 6, (0, 255, 255), -1)

                # Also save individual crop
                crop = image[y1c:max(0, ab['y2']), x1c:max(0, ab['x2'])].copy()
                for i in range(len(trace) - 1):
                    pt1 = (trace[i][0] - x1c, trace[i][1] - y1c)
                    pt2 = (trace[i+1][0] - x1c, trace[i+1][1] - y1c)
                    cv2.line(crop, pt1, pt2, (0, 0, 255), 2)
                cv2.circle(crop, (ep_a[0]-x1c, ep_a[1]-y1c), 5, (255, 0, 0), -1)
                cv2.circle(crop, (ep_b[0]-x1c, ep_b[1]-y1c), 5, (0, 255, 255), -1)
                cv2.imwrite(str(OUTPUT_DIR / f"A{idx}_final_trace.png"), crop)
            else:
                print(f"  PARTIAL: only {len(trace)} trace points")
        else:
            print(f"  FAILED (None)")
    except Exception as e:
        print(f"  ERROR: {e}")

# Draw arrow bboxes
for idx, arrow in enumerate(arrows_main):
    ab = arrow['bbox']
    cv2.rectangle(vis, (ab['x1'], ab['y1']), (ab['x2'], ab['y2']), (0, 255, 0), 1)
    cv2.putText(vis, f"A{idx}", (ab['x1'], ab['y1']-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite(str(OUTPUT_DIR / "full_image_traces.png"), vis)
print(f"\nFull visualization: {OUTPUT_DIR / 'full_image_traces.png'}")
