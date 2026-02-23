"""Visualization utilities for UML diagram overlays."""

from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


CLASS_COLOR = (30, 144, 255)   # DodgerBlue
ARROW_COLOR = (50, 205, 50)    # LimeGreen
TEXT_COLOR = (10, 10, 10)


def _draw_box(draw: ImageDraw.ImageDraw, bbox: Dict[str, int], color: Tuple[int, int, int],
              label: str | None = None) -> None:
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    if label:
        # Simple label box
        text_x, text_y = x1 + 2, max(0, y1 - 14)
        draw.rectangle([text_x - 1, text_y - 1, text_x + 120, text_y + 12], fill=(255, 255, 255))
        draw.text((text_x, text_y), label, fill=TEXT_COLOR)


def draw_overlay(
    image_path: str,
    class_boxes: List[Dict],
    arrow_boxes: List[Dict],
) -> Image.Image:
    """
    Draw class and arrow bounding boxes on a UML diagram.

    Args:
        image_path: Path to the image
        class_boxes: List of dicts with bbox {x1,y1,x2,y2}
        arrow_boxes: List of dicts with bbox {x1,y1,x2,y2}

    Returns:
        PIL Image with overlay
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for idx, cls in enumerate(class_boxes):
        bbox = cls.get("bbox", {})
        label = cls.get("class_name") or f"class_{idx + 1}"
        _draw_box(draw, bbox, CLASS_COLOR, label=label)

    for idx, arrow in enumerate(arrow_boxes):
        bbox = arrow.get("bbox", {})
        _draw_box(draw, bbox, ARROW_COLOR, label=f"arrow_{idx + 1}")

    return image
