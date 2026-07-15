import cv2
import numpy as np
from PIL import Image
from app.models.request_models import VectorizationConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2

def vectorize_contours(img: Image.Image, config: VectorizationConfig) -> str:
    """
    Converts raster boundaries/shapes into vector paths (SVG format).
    Uses OpenCV contour tracing and translates coordinates into SVG path strings.
    Approximates paths to simplify vertices.
    """
    logger.info("Running Step 8: Vectorization")
    
    if not config.enabled:
        logger.debug("Vectorization disabled. Returning empty SVG.")
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"></svg>'

    cv_img = pil_to_cv2(img)
    width, height = img.size
    
    # 1. Convert to grayscale and threshold
    if len(cv_img.shape) == 3:
        if cv_img.shape[2] == 4:
            # If transparent, trace the alpha channel
            gray = cv_img[:, :, 3]
        else:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img
        
    # Apply Otsu's thresholding to get binary shapes
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Translate contours to SVG path strings
    svg_paths = []
    
    for contour in contours:
        # Simplify contour using Douglas-Peucker algorithm
        if config.simplify_tolerance > 0:
            epsilon = config.simplify_tolerance * 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        else:
            approx = contour
            
        if len(approx) < 3:
            continue
            
        # Build path data
        path_data = []
        # Start command
        x0, y0 = approx[0][0]
        path_data.append(f"M {x0} {y0}")
        
        # Line commands
        for point in approx[1:]:
            x, y = point[0]
            path_data.append(f"L {x} {y}")
            
        # Close path
        path_data.append("Z")
        
        path_str = " ".join(path_data)
        # Add path tag (default color: black, fill-rule: evenodd)
        svg_paths.append(f'  <path d="{path_str}" fill="#4F46E5" fill-opacity="0.85" stroke="#312E81" stroke-width="1.5" />')

    # Wrap in SVG template
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">'
    svg_footer = "</svg>"
    
    svg_content = "\n".join([svg_header] + svg_paths + [svg_footer])
    
    logger.info(f"Vectorized image into {len(svg_paths)} SVG vector paths.")
    return svg_content
