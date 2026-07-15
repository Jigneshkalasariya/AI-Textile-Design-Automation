import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any
from app.models.request_models import TextileCADEngineConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

def remove_defects_and_noise(cv_img: np.ndarray, strength: float) -> np.ndarray:
    """
    Removes JPEG artifacts, scanning defects, dust, scratches, and background noise.
    Uses Bilateral Filter to preserve edges while smoothing, followed by morphological close/open.
    """
    if strength <= 0:
        return cv_img
    
    # Bilateral filter
    d = max(3, int(strength * 2) | 1) # Make sure it is odd
    sigma = strength * 8.0
    denoised = cv2.bilateralFilter(cv_img, d, sigma, sigma)
    
    # Morphological cleaning to remove dust and scratches
    k_size = 3 if strength < 4.0 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    # Close (to fill scratches/spots), then Open (to remove dust/noise)
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned

def correct_perspective(cv_img: np.ndarray) -> np.ndarray:
    """
    Detects skewed boundaries of the artwork using contour analysis and warps 
    the perspective back to a flat rectangular alignment.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv_img
        
    # Find largest contour by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = cv_img.shape[:2]
    img_area = h * w
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area: # Needs to cover a substantial part of the image
            break
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If it has 4 corners, we warp it
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-Left
            rect[2] = pts[np.argmax(s)] # Bottom-Right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-Right
            rect[3] = pts[np.argmax(diff)] # Bottom-Left
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(cv_img, M, (maxWidth, maxHeight))
            logger.info(f"Perspective corrected: Rectified from {pts.tolist()} to {maxWidth}x{maxHeight}")
            return warped
            
    return cv_img

def restore_colors(cv_img: np.ndarray, sat_scale: float) -> np.ndarray:
    """
    Applies CLAHE on the L channel of LAB and boosts saturation in HSV to restore faded colors.
    """
    if len(cv_img.shape) < 3:
        return cv_img
        
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Saturation scaling
    hsv = cv2.cvtColor(img_restored, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_new = np.clip(s.astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s_new, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def sharpen_image(cv_img: np.ndarray, strength: float) -> np.ndarray:
    """
    Enhances sharpness and edge crispness using Unsharp Masking.
    """
    if strength <= 0:
        return cv_img
    blurred = cv2.GaussianBlur(cv_img, (5, 5), 1.0)
    return cv2.addWeighted(cv_img, 1.0 + strength, blurred, -strength, 0)

def enhance_edges(cv_img: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blends Laplacian edge map to enforce crisp line boundaries.
    """
    if alpha <= 0:
        return cv_img
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    
    if len(cv_img.shape) == 3:
        laplacian_3ch = cv2.merge([laplacian, laplacian, laplacian])
    else:
        laplacian_3ch = laplacian
        
    return cv2.addWeighted(cv_img, 1.0, laplacian_3ch, -alpha, 0)

def upscale_resolution(cv_img: np.ndarray, target_longer_side: int = 3840) -> np.ndarray:
    """
    Upscales image using high-quality Lanczos interpolation.
    Calculates scale factor to target high resolution (minimum 4K or upscaled by 2x-4x).
    """
    h, w = cv_img.shape[:2]
    max_dim = max(h, w)
    
    # Calculate scale factor to reach target longer side (e.g. 3840 for 4K)
    scale = float(target_longer_side) / float(max_dim)
    # Clamp scale between 2.0 and 4.0
    scale = max(2.0, min(4.0, scale))
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def generate_bw_sketch(cv_img: np.ndarray) -> np.ndarray:
    """
    Generates a high-fidelity pure Black & White line sketch of the artwork.
    Produces pure black lines (0) on a pure white background (255) with no grayscale.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
    
    # 2. Smooth to reduce micro-noise and scanning artifacts
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Adaptive thresholding captures motifs and outlines
    thresh = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    
    # 4. Canny edge detection captures clear boundaries
    edges = cv2.Canny(smoothed, 30, 100)
    edges_inv = cv2.bitwise_not(edges)
    
    # 5. Combine both (bitwise AND) to get highly detailed lines and clean boundaries
    combined = cv2.bitwise_and(thresh, edges_inv)
    
    # 6. Morphological cleaning to remove dust and smooth lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Median blur to remove salt and pepper noise (isolated pixels)
    cleaned = cv2.medianBlur(combined, 3)
    # Thicken lines slightly for Textile CAD compatibility
    thickened = cv2.erode(cleaned, kernel, iterations=1)
    
    # Ensure binary output
    _, final_bw = cv2.threshold(thickened, 127, 255, cv2.THRESH_BINARY)
    return final_bw

def preprocess_image_cad(img: Image.Image, config: TextileCADEngineConfig) -> Dict[str, Image.Image]:
    """
    Step: Textile CAD Preprocessing Engine.
    Executes the automatic perspective correction, denoising, sharpening, color restoration, 
    and returns 6 independent preprocessed versions of the artwork + B&W sketch design.
    """
    logger.info("Running Step: Textile CAD Preprocessing Engine")
    
    # 0. Convert PIL image to OpenCV BGR
    cv_orig = pil_to_cv2(img)
    
    # 1. Correct Image Perspective (Common to all versions to rectify distortions)
    cv_rectified = correct_perspective(cv_orig)
    
    versions = {}
    
    # --- Version 1: Balanced restoration ---
    logger.debug("Generating CAD Version 1: Balanced restoration")
    v1_cv = remove_defects_and_noise(cv_rectified, strength=2.5)
    v1_cv = restore_colors(v1_cv, sat_scale=1.1)
    v1_cv = sharpen_image(v1_cv, strength=0.8)
    v1_cv = upscale_resolution(v1_cv)
    versions["version1"] = cv2_to_pil(v1_cv)
    
    # --- Version 2: Maximum sharpness ---
    logger.debug("Generating CAD Version 2: Maximum sharpness")
    v2_cv = remove_defects_and_noise(cv_rectified, strength=1.5)
    v2_cv = restore_colors(v2_cv, sat_scale=1.0)
    v2_cv = sharpen_image(v2_cv, strength=1.8)
    v2_cv = enhance_edges(v2_cv, alpha=0.25)
    v2_cv = upscale_resolution(v2_cv)
    versions["version2"] = cv2_to_pil(v2_cv)
    
    # --- Version 3: Print optimized ---
    logger.debug("Generating CAD Version 3: Print optimized")
    v3_cv = remove_defects_and_noise(cv_rectified, strength=3.5)
    v3_cv = restore_colors(v3_cv, sat_scale=1.25)
    v3_cv = sharpen_image(v3_cv, strength=0.5)
    v3_cv = upscale_resolution(v3_cv)
    versions["version3"] = cv2_to_pil(v3_cv)
    
    # --- Version 4: Vectorization optimized ---
    logger.debug("Generating CAD Version 4: Vectorization optimized")
    # Strong bilateral denoising to remove textures/gradients for clean tracing
    v4_cv = remove_defects_and_noise(cv_rectified, strength=5.0)
    v4_cv = restore_colors(v4_cv, sat_scale=1.2)
    v4_cv = sharpen_image(v4_cv, strength=1.5)
    v4_cv = upscale_resolution(v4_cv)
    versions["version4"] = cv2_to_pil(v4_cv)
    
    # --- Version 5: Repeat detection optimized ---
    logger.debug("Generating CAD Version 5: Repeat detection optimized")
    # Focus on structural line contrast, moderate denoise
    v5_cv = remove_defects_and_noise(cv_rectified, strength=2.0)
    # Local histogram equalization/normalization to stand out boundaries
    gray = cv2.cvtColor(v5_cv, cv2.COLOR_BGR2GRAY) if len(v5_cv.shape) == 3 else v5_cv
    normalized_gray = cv2.equalizeHist(gray)
    if len(v5_cv.shape) == 3:
        # Blend grayscale equalized channel with color channels slightly to preserve colors but pop edges
        yuv = cv2.cvtColor(v5_cv, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y_eq = cv2.equalizeHist(y)
        v5_cv = cv2.cvtColor(cv2.merge((y_eq, u, v)), cv2.COLOR_YUV2BGR)
    else:
        v5_cv = normalized_gray
    v5_cv = sharpen_image(v5_cv, strength=1.2)
    v5_cv = upscale_resolution(v5_cv)
    versions["version5"] = cv2_to_pil(v5_cv)
    
    # --- Version 6: Texcelle import optimized ---
    logger.debug("Generating CAD Version 6: Texcelle import optimized")
    # Needs flat colors and crisp edges
    v6_cv = remove_defects_and_noise(cv_rectified, strength=3.0)
    v6_cv = restore_colors(v6_cv, sat_scale=1.1)
    v6_cv = sharpen_image(v6_cv, strength=1.0)
    v6_cv = upscale_resolution(v6_cv)
    v6_pil = cv2_to_pil(v6_cv)
    
    # Flat color quantization (reduce to 32 flat color palette for engraving standard)
    quantized_v6 = v6_pil.convert("RGB").quantize(
        colors=32,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    ).convert("RGB")
    versions["version6"] = quantized_v6
    
    # --- Sketch BW: Black & White Sketch Design (MANDATORY) ---
    logger.debug("Generating CAD Sketch: Black & White Sketch Design")
    sketch_cv = generate_bw_sketch(cv_rectified)
    sketch_cv = upscale_resolution(sketch_cv)
    versions["sketch_bw"] = cv2_to_pil(sketch_cv)
    
    logger.info("Generated all 6 Textile CAD image versions + sketch successfully.")
    return versions
