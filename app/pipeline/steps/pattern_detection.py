import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any
from app.models.request_models import PatternDetectionConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2

def detect_pattern(img: Image.Image, config: PatternDetectionConfig) -> Dict[str, Any]:
    """
    Analyzes the image structure to detect the textile repeat unit boundaries and type.
    Uses structural autocorrelation/template matching.
    Returns:
        - Dict with keys:
            - 'repeat_type': 'straight' | 'half-drop' | 'unknown'
            - 'repeat_width': width in pixels
            - 'repeat_height': height in pixels
            - 'confidence': confidence score (0.0 to 1.0)
    """
    logger.info("Running Step 4: Pattern Detection")
    width, height = img.size
    
    # Default outputs if detection fails or is disabled
    default_result = {
        "repeat_type": "straight",
        "repeat_width": width,
        "repeat_height": height,
        "confidence": 0.5
    }
    
    if not config.enabled:
        logger.debug("Pattern detection disabled. Using default dimensions.")
        return default_result

    cv_img = pil_to_cv2(img)
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    # Auto-correlation / Template matching approach:
    # 1. Take a patch from the center of the image
    patch_size = min(128, min(width, height) // 4)
    if patch_size < 16:
        logger.debug("Image too small for robust repeat unit detection. Using default values.")
        return default_result

    cx, cy = width // 2, height // 2
    x1, y1 = cx - patch_size // 2, cy - patch_size // 2
    patch = gray[y1:y1+patch_size, x1:x1+patch_size]

    # 2. Run matchTemplate on the image
    # Note: To find repeats, we look for secondary peaks (other than the central peak)
    try:
        res = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
        
        # Threshold to find local peaks
        peak_threshold = 0.65
        peaks = np.where(res >= peak_threshold)
        
        # Group close peaks using standard point cluster approximation or basic spacing
        pts = list(zip(peaks[1], peaks[0])) # (x, y) coordinates of matching points
        
        # Calculate horizontal and vertical distances between matching points
        # to estimate repeat_width and repeat_height
        x_diffs = []
        y_diffs = []
        
        # Find offset from patch center
        ref_x, ref_y = x1, y1
        
        for pt in pts:
            dx = abs(pt[0] - ref_x)
            dy = abs(pt[1] - ref_y)
            if dx > 10:
                x_diffs.append(dx)
            if dy > 10:
                y_diffs.append(dy)

        # Filter and find modal spacings
        est_w = width
        est_h = height
        repeat_type = "straight"
        
        if x_diffs:
            # Get median of differences
            median_dx = int(np.median(x_diffs))
            if 30 < median_dx < width:
                est_w = median_dx
        
        if y_diffs:
            median_dy = int(np.median(y_diffs))
            if 30 < median_dy < height:
                est_h = median_dy

        # Check for half-drop match
        # A half-drop repeat has horizontal matches that are shifted vertically by half of the repeat height.
        # Check if there are peaks at (est_w, est_h // 2)
        half_drop_detected = False
        for pt in pts:
            dx = abs(pt[0] - ref_x)
            dy = abs(pt[1] - ref_y)
            # If horizontal shift is close to est_w and vertical shift is close to est_h/2
            if abs(dx - est_w) < 15 and abs(dy - (est_h / 2)) < 15:
                half_drop_detected = True
                break
                
        if half_drop_detected:
            repeat_type = "half-drop"
            logger.info("Half-drop repeat pattern detected!")
        else:
            logger.info(f"Straight repeat pattern detected (W={est_w}, H={est_h}).")
            
        return {
            "repeat_type": repeat_type,
            "repeat_width": est_w,
            "repeat_height": est_h,
            "confidence": 0.8
        }

    except Exception as e:
        logger.error(f"Pattern detection algorithm failed: {e}. Returning default layout.")
        return default_result
