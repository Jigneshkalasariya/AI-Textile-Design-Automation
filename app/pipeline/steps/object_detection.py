import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from app.models.request_models import ObjectDetectionConfig
from app.core.config import settings
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

# Try importing ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics YOLOv8 not available. Contour detection fallback will be used.")

def detect_objects(img: Image.Image, config: ObjectDetectionConfig) -> Tuple[List[Image.Image], List[Dict[str, int]]]:
    """
    Detects and isolates individual motif objects within the image.
    Uses YOLOv8 if enabled and weights are available, otherwise falls back
    to OpenCV contour-based boundary detection (ideal for isolated transparent images).
    Returns:
        - List of cropped PIL images (motifs)
        - List of bounding box dicts: [{'x': x, 'y': y, 'w': w, 'h': h}]
    """
    logger.info("Running Step 3: Object Detection")
    if not config.enabled:
        logger.debug("Object detection disabled. Returning single motif (the full image).")
        return [img], [{"x": 0, "y": 0, "w": img.width, "h": img.height}]

    # YOLO / Grounding DINO implementation
    if config.model == "grounding_dino":
        logger.warning("Grounding DINO not natively installed yet. Falling back to next available method.")
    
    if YOLO_AVAILABLE and settings.ALLOW_MODEL_DOWNLOADS and (config.model == "yolo11" or config.model == "yolov8"):
        try:
            logger.debug(f"Instantiating {config.model} detector with confidence threshold: {config.confidence}")
            # Ensure correct model string
            model_name = "yolo11n.pt" if config.model == "yolo11" else "yolov8n.pt"
            model_path = settings.model_paths_path / model_name
            model = YOLO(str(model_path) if model_path.exists() else model_name)
            
            # Run inference
            results = model(img, conf=config.confidence)
            motifs = []
            bboxes = []
            
            # Extract bounding boxes
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    w = x2 - x1
                    h = y2 - y1
                    if w > 10 and h > 10:  # ignore tiny noise
                        bboxes.append({"x": x1, "y": y1, "w": w, "h": h})
                        cropped = img.crop((x1, y1, x2, y2))
                        motifs.append(cropped)
            
            if motifs:
                logger.info(f"{config.model} detected {len(motifs)} objects/motifs.")
                return motifs, bboxes
            else:
                logger.debug(f"{config.model} found 0 items. Falling back to contour analysis.")
        except Exception as e:
            logger.error(f"{config.model} inference failed: {e}. Falling back to OpenCV contour-based isolation.")

    # Fallback OpenCV contour detection (Uses Alpha channel if present, otherwise thresholding)
    logger.debug("Executing OpenCV contour detection fallback...")
    cv_img = pil_to_cv2(img)
    
    # If the image has an alpha channel, use it directly as the mask
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
        mask = cv_img[:, :, 3]
    else:
        # Fallback to thresholding grayscale representation
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motifs = []
    bboxes = []
    
    # Filter by minimum area (e.g. 100 pixels) to avoid noise
    min_area = 100
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append({"x": x, "y": y, "w": w, "h": h})
            
            # Crop the object from PIL Image
            cropped = img.crop((x, y, x + w, y + h))
            motifs.append(cropped)

    logger.info(f"OpenCV contour detection found {len(motifs)} motifs.")
    if not motifs:
        # Fallback if no contours found
        logger.debug("No distinct motifs detected. Returning whole image as single motif.")
        return [img], [{"x": 0, "y": 0, "w": img.width, "h": img.height}]

    return motifs, bboxes
