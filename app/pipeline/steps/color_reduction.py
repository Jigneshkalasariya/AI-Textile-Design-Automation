import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from app.models.request_models import ColorReductionConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2

def reduce_colors(img: Image.Image, config: ColorReductionConfig) -> Tuple[Image.Image, List[Tuple[int, int, int]]]:
    """
    Reduces the image to a fixed palette of size `palette_size` (quantization) using KMeans clustering.
    Returns:
        - Quantized PIL Image (in 'P' mode with internal palette)
        - List of (R, G, B) tuples representing the active palette
    """
    logger.info("Running Step 7: Color Reduction (KMeans)")
    
    cv_img = pil_to_cv2(img.convert("RGB"))
    
    # Reshape for KMeans
    pixel_data = cv_img.reshape((-1, 3))
    pixel_data = np.float32(pixel_data)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = config.palette_size
    _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Centers are returned in BGR format
    centers = np.uint8(centers)
    
    # Extract RGB palette
    palette_rgb = []
    for bgr in centers:
        palette_rgb.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
        
    flat_palette_rgb = []
    for rgb in palette_rgb:
        flat_palette_rgb.extend(rgb)
        
    # Pad to 768 elements (256 RGB colors) for PIL 'P' mode requirement
    flat_palette_rgb.extend([0] * (768 - len(flat_palette_rgb)))
    
    # Create P mode image mapping
    quantized_img = Image.new("P", img.size)
    quantized_img.putpalette(flat_palette_rgb)
    
    # The kmeans labels match exactly the palette indices
    quantized_img.putdata(labels.flatten())
    
    logger.info(f"Color reduction completed. Palette size: {len(palette_rgb)}")
    return quantized_img, palette_rgb
