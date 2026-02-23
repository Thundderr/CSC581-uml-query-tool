"""
OCR Text Extraction for UML Class Diagrams.

This module provides classes and functions for extracting text from
detected UML class boxes using EasyOCR and parsing the text into
structured class information.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


# Visibility modifiers in UML notation
VISIBILITY_MAP = {
    '+': 'public',
    '-': 'private',
    '#': 'protected',
    '~': 'package'
}

# Class ID for UML class boxes in our YOLO model
CLASS_BOX_ID = 1

# Minimum OCR confidence to include a result
MIN_OCR_CONFIDENCE = 0.3

# Character allowlist for UML text (prevents # being detected instead of :)
OCR_ALLOWLIST = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:_+-<>()[] .'


def expand_bbox(x1: int, y1: int, x2: int, y2: int,
                img_h: int, img_w: int, padding: int = 10) -> Tuple[int, int, int, int]:
    """
    Expand bounding box with padding while staying within image bounds.

    Args:
        x1, y1, x2, y2: Original bounding box coordinates
        img_h, img_w: Image dimensions
        padding: Pixels to add on each side

    Returns:
        Expanded bounding box coordinates (x1, y1, x2, y2)
    """
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(img_w, x2 + padding),
        min(img_h, y2 + padding)
    )


def scale_if_needed(image: np.ndarray, min_height: int = 80) -> np.ndarray:
    """
    Scale small images for better OCR accuracy.

    Args:
        image: Input image (grayscale or BGR)
        min_height: Minimum height before scaling is applied

    Returns:
        Scaled image if too small, otherwise original
    """
    h = image.shape[0]
    if h < min_height:
        scale = max(3.0, min_height / h)
        return cv2.resize(image, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return image


def preprocess_strategies(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate multiple preprocessed versions of an image for OCR.

    Different preprocessing strategies work better for different image qualities.
    We try multiple and pick the one with best OCR confidence.

    Args:
        image: BGR image from cv2

    Returns:
        List of (strategy_name, processed_image) tuples
    """
    results = []

    # Convert to grayscale first
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Also extract HSV Value channel (better for colored backgrounds)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
    else:
        gray = image.copy()
        v_channel = image.copy()

    # Strategy 1: HSV Value channel (BEST for colored backgrounds like yellow)
    # This removes color information while preserving text contrast
    results.append(('hsv_value', v_channel))

    # Strategy 2: Grayscale only (let EasyOCR handle it)
    results.append(('grayscale', gray))

    # Strategy 3: CLAHE enhancement on V-channel (no thresholding)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v_channel)
    results.append(('hsv_clahe', v_enhanced))

    # Strategy 4: CLAHE on grayscale
    enhanced = clahe.apply(gray)
    results.append(('clahe', enhanced))

    # Strategy 5: Gentle denoising + CLAHE
    denoised = cv2.fastNlMeansDenoising(v_channel, h=10)
    denoised_enhanced = clahe.apply(denoised)
    results.append(('denoised_clahe', denoised_enhanced))

    return results


def preprocess_for_ocr(image: np.ndarray, scale_factor: float = 2.0,
                       min_height: int = 100) -> np.ndarray:
    """
    Simple preprocessing for OCR (legacy function, kept for compatibility).

    Now uses HSV Value channel for better handling of colored backgrounds.

    Args:
        image: BGR image (from cv2)
        scale_factor: Factor to scale small images
        min_height: Minimum height before scaling is applied

    Returns:
        Preprocessed grayscale image
    """
    # Use HSV Value channel instead of grayscale
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
    else:
        v_channel = image.copy()

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(v_channel)

    # Scale up small images for better OCR
    if enhanced.shape[0] < min_height:
        enhanced = cv2.resize(
            enhanced, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_CUBIC
        )

    return enhanced


def crop_detection(image: np.ndarray, bbox: Tuple[int, int, int, int],
                   padding: int = 10) -> np.ndarray:
    """
    Crop a detected region from the image with padding.

    Args:
        image: Full image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding to add around the box

    Returns:
        Cropped image region
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = expand_bbox(*bbox, h, w, padding)
    return image[y1:y2, x1:x2]


def parse_visibility(text: str) -> Tuple[str, str]:
    """
    Extract visibility modifier from the beginning of a line.

    Args:
        text: Line of text from UML class

    Returns:
        Tuple of (visibility, remaining_text)
    """
    text = text.strip()
    if text and text[0] in VISIBILITY_MAP:
        return VISIBILITY_MAP[text[0]], text[1:].strip()
    return 'public', text


def parse_attribute(line: str) -> Optional[Dict[str, str]]:
    """
    Parse a UML attribute line.

    Format: [visibility] name: type

    Args:
        line: Attribute line from UML class

    Returns:
        Dictionary with name, type, visibility or None if not an attribute
    """
    visibility, text = parse_visibility(line)

    # Attributes have ':' but no '()'
    if ':' in text and '(' not in text:
        parts = text.split(':', 1)
        if len(parts) == 2:
            return {
                'name': parts[0].strip(),
                'type': parts[1].strip(),
                'visibility': visibility
            }
    return None


def parse_method(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a UML method line.

    Format: [visibility] name(params): return_type

    Args:
        line: Method line from UML class

    Returns:
        Dictionary with name, parameters, return_type, visibility or None
    """
    visibility, text = parse_visibility(line)

    # Methods have '()' or '('
    match = re.match(r'([\w]+)\s*\(([^)]*)\)(?:\s*:\s*(.+))?', text)
    if match:
        name = match.group(1)
        params_str = match.group(2).strip()
        return_type = match.group(3).strip() if match.group(3) else 'void'

        # Parse parameters
        parameters = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if ':' in param:
                    p_parts = param.split(':', 1)
                    parameters.append({
                        'name': p_parts[0].strip(),
                        'type': p_parts[1].strip()
                    })
                elif param:
                    parameters.append({'name': param, 'type': 'unknown'})

        return {
            'name': name,
            'parameters': parameters,
            'return_type': return_type,
            'visibility': visibility
        }
    return None


def parse_uml_class(text: str) -> Dict[str, Any]:
    """
    Parse OCR text from a UML class box into structured data.

    UML class structure:
    - First line: Class name
    - Lines with ':' but no '()': Attributes
    - Lines with '()': Methods

    Args:
        text: Raw OCR text from class box

    Returns:
        Dictionary with class_name, attributes, methods
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    result = {
        'class_name': '',
        'attributes': [],
        'methods': [],
        'unparsed_lines': []
    }

    if not lines:
        return result

    # First non-empty line is typically the class name
    result['class_name'] = lines[0].strip()

    # Process remaining lines
    for line in lines[1:]:
        # Skip separator lines
        if re.match(r'^[-_=]+$', line):
            continue

        # Try parsing as method first (more specific pattern)
        method = parse_method(line)
        if method:
            result['methods'].append(method)
            continue

        # Try parsing as attribute
        attribute = parse_attribute(line)
        if attribute:
            result['attributes'].append(attribute)
            continue

        # Line couldn't be parsed
        result['unparsed_lines'].append(line)

    return result


class UMLTextExtractor:
    """
    Complete pipeline for extracting text from UML class diagrams.

    Pipeline:
    1. Detect class boxes using YOLO
    2. Crop each detected box with padding
    3. Try multiple preprocessing strategies
    4. Extract text using EasyOCR with beamsearch decoder
    5. Fall back to PyTesseract for namespace text
    6. Parse UML structure
    """

    def __init__(self, model_path: str,
                 confidence_threshold: float = 0.5,
                 bbox_padding: int = 10,
                 gpu: bool = False,
                 multi_strategy: bool = True,
                 use_pytesseract_fallback: bool = True):
        """
        Initialize the extractor.

        Args:
            model_path: Path to trained YOLO model weights
            confidence_threshold: Minimum detection confidence
            bbox_padding: Padding around detected boxes
            gpu: Whether to use GPU for EasyOCR
            multi_strategy: Whether to try multiple preprocessing strategies
            use_pytesseract_fallback: Whether to use PyTesseract as fallback

        Raises:
            ImportError: If required dependencies are not installed
        """
        if YOLO is None:
            raise ImportError("ultralytics is required: pip install ultralytics")
        if easyocr is None:
            raise ImportError("easyocr is required: pip install easyocr")

        self.detector = YOLO(str(model_path))
        self.ocr_reader = easyocr.Reader(['en'], gpu=gpu)
        self.confidence_threshold = confidence_threshold
        self.bbox_padding = bbox_padding
        self.multi_strategy = multi_strategy
        self.use_pytesseract_fallback = use_pytesseract_fallback and pytesseract is not None

    def detect_classes(self, image: np.ndarray) -> List[Dict]:
        """
        Detect class boxes in the image using YOLO.

        Args:
            image: BGR image

        Returns:
            List of detections with bbox and confidence
        """
        from src.utils.device import get_device
        results = self.detector(image, device=get_device(), verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                if cls_id == CLASS_BOX_ID and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf
                    })

        return detections

    def _ocr_single_strategy(self, image: np.ndarray) -> Tuple[List, float]:
        """
        Run OCR on a single preprocessed image using beamsearch decoder.

        Args:
            image: Preprocessed grayscale image

        Returns:
            Tuple of (ocr_results, average_confidence)
        """
        # Use beamsearch decoder for better accuracy on unusual patterns like ::
        # Use allowlist to prevent # being detected instead of :
        results = self.ocr_reader.readtext(
            image,
            paragraph=False,
            decoder='beamsearch',
            beamWidth=10,
            allowlist=OCR_ALLOWLIST
        )

        if not results:
            return [], 0.0

        # Filter by confidence
        filtered = [r for r in results if r[2] >= MIN_OCR_CONFIDENCE]

        if not filtered:
            return [], 0.0

        # Sort by y-coordinate (top to bottom) to preserve reading order
        filtered.sort(key=lambda r: r[0][0][1])

        avg_conf = np.mean([r[2] for r in filtered])
        return filtered, avg_conf

    def _try_pytesseract(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Fallback OCR using PyTesseract - better for monospace code-like text.

        Args:
            image: BGR or grayscale image

        Returns:
            Tuple of (extracted_text, confidence)
        """
        if pytesseract is None:
            return '', 0.0

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binarize for Tesseract (it works better with binary images)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # PSM 6 = Assume single uniform block of text
        try:
            text = pytesseract.image_to_string(binary, config='--psm 6')
            return text.strip(), 0.7  # Fixed confidence for Tesseract
        except Exception:
            # Tesseract binary not installed or not in PATH.
            return '', 0.0

    def extract_text(self, image: np.ndarray) -> Tuple[str, float, str]:
        """
        Extract text from a cropped class box image.

        Tries multiple preprocessing strategies and returns the best result.
        Falls back to PyTesseract if EasyOCR misses namespace separators.

        Args:
            image: Cropped class box image (BGR)

        Returns:
            Tuple of (extracted_text, average_confidence, strategy_used)
        """
        # Scale up small images first
        scaled = scale_if_needed(image)

        if self.multi_strategy:
            # Try multiple preprocessing strategies
            best_text = ''
            best_conf = 0.0
            best_strategy = 'none'

            for strategy_name, processed in preprocess_strategies(scaled):
                results, avg_conf = self._ocr_single_strategy(processed)

                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_strategy = strategy_name
                    # Extract text from results
                    best_text = '\n'.join([r[1] for r in results])

            # PyTesseract fallback: if text looks like namespace but missing ::
            if self.use_pytesseract_fallback:
                # Check if text might have namespace separators that were missed
                if '::' not in best_text and any(c.isupper() for c in best_text):
                    tess_text, tess_conf = self._try_pytesseract(scaled)
                    if '::' in tess_text:
                        return tess_text, tess_conf, 'pytesseract_fallback'

            return best_text, best_conf, best_strategy
        else:
            # Simple preprocessing (CLAHE only)
            processed = preprocess_for_ocr(scaled)
            results, avg_conf = self._ocr_single_strategy(processed)
            text = '\n'.join([r[1] for r in results])
            return text, avg_conf, 'clahe'

    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        Full extraction pipeline for a single image.

        Args:
            image_path: Path to UML diagram image

        Returns:
            Dictionary with image info and extracted classes
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'Could not load image: {image_path}'}

        h, w = image.shape[:2]
        detections = self.detect_classes(image)

        classes = []
        for det in detections:
            bbox = det['bbox']
            cropped = crop_detection(image, bbox, self.bbox_padding)
            raw_text, ocr_conf, strategy = self.extract_text(cropped)
            parsed = parse_uml_class(raw_text)

            classes.append({
                'bbox': {
                    'x1': bbox[0], 'y1': bbox[1],
                    'x2': bbox[2], 'y2': bbox[3]
                },
                'detection_confidence': det['confidence'],
                'class_name': parsed['class_name'],
                'attributes': parsed['attributes'],
                'methods': parsed['methods'],
                'raw_text': raw_text,
                'ocr_confidence': ocr_conf,
                'ocr_strategy': strategy,
                'unparsed_lines': parsed['unparsed_lines']
            })

        return {
            'image_path': str(image_path),
            'image_size': {'width': w, 'height': h},
            'num_classes_detected': len(classes),
            'classes': classes
        }

    def batch_extract(self, image_paths: List[str], verbose: bool = True) -> List[Dict]:
        """
        Process multiple images and collect results.

        Args:
            image_paths: List of image paths
            verbose: Print progress

        Returns:
            List of extraction results
        """
        results = []

        for i, path in enumerate(image_paths):
            if verbose:
                print(f"Processing {i+1}/{len(image_paths)}: {Path(path).name}", end='\r')

            result = self.extract(path)
            results.append(result)

        if verbose:
            print(f"\nProcessed {len(results)} images")

        return results
