"""
Data loading utilities for UML Class Diagram Dataset.
"""
import kagglehub
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import yaml


DATASET_ID = "domenicoarm/uml-class-diagram-dataset-bounded-box-rating"

# Class names from data.yaml (arrow=0, class=1, cross=2)
CLASS_NAMES = {
    0: "arrow",
    1: "class",
    2: "cross"
}

CLASS_COLORS = {
    0: "green",   # arrow
    1: "blue",    # class
    2: "red"      # cross
}


def download_dataset(force: bool = False) -> Path:
    """
    Download dataset using kagglehub.

    Args:
        force: If True, re-download even if cached

    Returns:
        Path to downloaded dataset
    """
    return Path(kagglehub.dataset_download(DATASET_ID, force_download=force))


def load_class_names_from_yaml(dataset_path: Path) -> Dict[int, str]:
    """
    Load class names from the dataset's data.yaml file.

    Args:
        dataset_path: Path to dataset root

    Returns:
        Dict mapping class_id to class_name
    """
    yaml_path = dataset_path / "UML_YOLOv8" / "data.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return {i: name for i, name in enumerate(config['names'])}


def get_yolo_label_path(image_path: Path) -> Path:
    """
    Get the label path for a YOLO format image.

    YOLO stores images in images/ and labels in labels/ subdirectories.

    Args:
        image_path: Path to image file (e.g., .../split/images/filename.jpg)

    Returns:
        Path to corresponding label file (e.g., .../split/labels/filename.txt)
    """
    return image_path.parent.parent / "labels" / image_path.with_suffix('.txt').name


def load_yolo_split(dataset_path: Path, split: str = 'train') -> List[Dict]:
    """
    Load all image-label pairs from a YOLO dataset split.

    Args:
        dataset_path: Path to dataset root
        split: 'train', 'valid', or 'test'

    Returns:
        List of dicts with 'image' and 'label' paths
    """
    images_dir = Path(dataset_path) / "UML_YOLOv8" / split / "images"

    if not images_dir.exists():
        return []

    pairs = []
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            label_path = get_yolo_label_path(img_path)
            if label_path.exists():
                pairs.append({
                    'image': img_path,
                    'label': label_path
                })

    return pairs


def parse_yolo_annotation(
    txt_path: Path,
    img_width: int,
    img_height: int,
    class_names: Dict[int, str] = None
) -> List[Dict]:
    """
    Parse YOLO format annotation file.

    YOLO format: class_id x_center y_center width height (all normalized 0-1)

    Args:
        txt_path: Path to .txt annotation file
        img_width: Image width in pixels
        img_height: Image height in pixels
        class_names: Optional dict mapping class_id to name

    Returns:
        List of annotation dicts with absolute pixel coordinates
    """
    if class_names is None:
        class_names = CLASS_NAMES

    annotations = []

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = x_center - width / 2
                y1 = y_center - height / 2

                annotations.append({
                    'class_id': class_id,
                    'class_name': class_names.get(class_id, f"Unknown_{class_id}"),
                    'x1': x1,
                    'y1': y1,
                    'x2': x1 + width,
                    'y2': y1 + height,
                    'width': width,
                    'height': height,
                    'x_center': x_center,
                    'y_center': y_center
                })

    return annotations


def parse_xml_annotation(xml_path: Path) -> List[Dict]:
    """
    Parse Faster R-CNN XML format annotation file (PASCAL VOC format).

    Args:
        xml_path: Path to .xml annotation file

    Returns:
        List of annotation dicts with bounding box info
    """
    annotations = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')

        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        annotations.append({
            'class_name': name,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'width': x2 - x1,
            'height': y2 - y1
        })

    return annotations


def load_image_with_annotations(
    image_path: Path,
    label_path: Path = None,
    annotation_format: str = 'yolo',
    class_names: Dict[int, str] = None
) -> Tuple[Image.Image, List[Dict]]:
    """
    Load an image and its annotations.

    Args:
        image_path: Path to image file
        label_path: Path to label file (if None, will be inferred based on format)
        annotation_format: 'yolo' or 'xml'
        class_names: Optional dict mapping class_id to name

    Returns:
        Tuple of (PIL Image, list of annotations)
    """
    img = Image.open(image_path)

    if annotation_format == 'yolo':
        if label_path is None:
            label_path = get_yolo_label_path(image_path)
        if label_path.exists():
            annotations = parse_yolo_annotation(
                label_path, img.size[0], img.size[1], class_names
            )
        else:
            annotations = []
    else:
        if label_path is None:
            label_path = image_path.with_suffix('.xml')
        if label_path.exists():
            annotations = parse_xml_annotation(label_path)
        else:
            annotations = []

    return img, annotations


def get_image_files(dataset_path: Path, format: str = 'yolo') -> List[Path]:
    """
    Get all image files from the dataset.

    Args:
        dataset_path: Path to dataset directory
        format: 'yolo' for UML_YOLOv8 format, 'frcnn' for Faster_R-CNN format

    Returns:
        List of image file paths
    """
    if format == 'yolo':
        image_files = []
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_path / "UML_YOLOv8" / split / "images"
            if images_dir.exists():
                image_files.extend(images_dir.glob("*.[jJ][pP][gG]"))
                image_files.extend(images_dir.glob("*.[jJ][pP][eE][gG]"))
                image_files.extend(images_dir.glob("*.[pP][nN][gG]"))
        return image_files
    else:
        frcnn_dir = dataset_path / "Faster_R-CNN" / "Faster_R-CNN"
        if frcnn_dir.exists():
            return list(frcnn_dir.glob("*.[jJ][pP][eE][gG]"))
        return []
