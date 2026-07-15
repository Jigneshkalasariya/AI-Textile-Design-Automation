import cv2
import numpy as np
from PIL import Image
from app.models.request_models import BackgroundRemovalConfig
from app.core.config import settings
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

# Try importing rembg
try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg is not installed. Will use OpenCV color-based fallback background removal.")

def remove_background(img: Image.Image, config: BackgroundRemovalConfig) -> Image.Image:
    """
    Removes the background of the image.
    Tries using `rembg` (U2Net) first. If disabled or it fails,
    falls back to a color-distance thresholding algorithm to create a transparent alpha channel.
    """
    logger.info("Running Step 2: Background Removal")
    if not config.enabled:
        logger.debug("Background removal disabled. Skipping.")
        return img

    if REMBG_AVAILABLE and settings.ALLOW_MODEL_DOWNLOADS:
        try:
            logger.debug(f"Attempting background removal using rembg (model={config.model_name})")
            # Convert PIL Image to bytes for rembg.remove
            img_rgba = img.convert("RGBA")
            # Apply rembg
            output_pil = rembg.remove(img_rgba, alpha_matting=config.alpha_matting)
            return output_pil
        except Exception as e:
            logger.error(f"rembg execution failed: {e}. Falling back to OpenCV thresholding.")
    
    # Fallback OpenCV implementation
    logger.debug("Executing OpenCV color distance background removal fallback...")
    cv_img = pil_to_cv2(img)
    # Convert BGR to BGRA
    if cv_img.shape[2] == 3:
        cv_img = cv2_to_bgra(cv_img)
        
    # Sample background color from the corners
    h, w = cv_img.shape[:2]
    corners = [
        cv_img[0, 0, :3],
        cv_img[0, w-1, :3],
        cv_img[h-1, 0, :3],
        cv_img[h-1, w-1, :3]
    ]
    # Use median of corners as candidate background color
    bg_color = np.median(corners, axis=0)
    
    # Compute Euclidean distance in RGB color space
    diff = cv_img[:, :, :3] - bg_color
    dist = np.linalg.norm(diff, axis=2)
    
    # Threshold distance: pixels close to background color become transparent
    threshold = 30.0
    alpha = np.where(dist < threshold, 0, 255).astype(np.uint8)
    
    # Soften edges of alpha mask (feathering / smoothing)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    
    # Set the alpha channel
    cv_img[:, :, 3] = alpha
    
    return cv2_to_pil(cv_img)

def cv2_to_bgra(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
