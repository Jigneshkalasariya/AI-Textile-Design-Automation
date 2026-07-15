import cv2
import numpy as np
from PIL import Image
from app.models.request_models import EnhanceConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

def enhance_image(img: Image.Image, config: EnhanceConfig) -> Image.Image:
    """
    Applies image enhancement including denoising (Bilateral filtering) 
    and sharpening (unsharp mask via Gaussian Blur subtraction).
    """
    logger.info("Running Step 1: Image Enhancement")
    cv_img = pil_to_cv2(img)
    
    # 1. Denoising using Bilateral Filter to preserve sharp edges
    if config.denoise_strength > 0:
        d = int(config.denoise_strength * 3)
        # Ensure d is odd and positive
        d = max(3, d if d % 2 != 0 else d + 1)
        sigma_color = config.denoise_strength * 10.0
        sigma_space = config.denoise_strength * 10.0
        
        logger.debug(f"Applying Bilateral Filter (d={d}, sc={sigma_color}, ss={sigma_space})")
        # Split color channels or apply to BGR directly
        cv_img = cv2.bilateralFilter(cv_img, d, sigma_color, sigma_space)
        
    # 2. Sharpening using Unsharp Mask
    if config.sharpen_strength > 0:
        logger.debug(f"Applying Unsharp Mask (strength={config.sharpen_strength})")
        # Gaussian blur representation
        blurred = cv2.GaussianBlur(cv_img, (5, 5), 1.0)
        # unsharp mask = original + strength * (original - blurred)
        cv_img = cv2.addWeighted(cv_img, 1.0 + config.sharpen_strength, blurred, -config.sharpen_strength, 0)
        
    # 3. RealESRGAN Placeholder
    if config.use_realesrgan:
        logger.warning("RealESRGAN was requested but is not initialized. Using procedural sharpening/bilateral filter as fallback.")

    return cv2_to_pil(cv_img)
