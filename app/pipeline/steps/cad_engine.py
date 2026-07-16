import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any
from app.models.request_models import TextileCADEngineConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil

def analyze_image_quality(cv_img: np.ndarray) -> bool:
    """
    STEP 1: Analyses input image quality.
    Returns True if low quality (blurry, low resolution, or low contrast), requiring upscale.
    """
    h, w = cv_img.shape[:2]
    # Resolution check: If smaller than 2000 pixels on any side, consider low resolution
    if h < 2000 or w < 2000:
        logger.info(f"Quality Validation: Low resolution detected ({w}x{h}). Scaling is required.")
        return True
        
    # Sharpness check using Laplacian variance
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 100.0:
        logger.info(f"Quality Validation: Low sharpness / blur detected (Laplacian variance: {lap_var:.2f}). Enhancement is required.")
        return True
        
    # Contrast check: Standard deviation of grayscale channel
    std_dev = np.std(gray)
    if std_dev < 30.0:
        logger.info(f"Quality Validation: Low contrast detected (grayscale std dev: {std_dev:.2f}). Contrast restoration is required.")
        return True
        
    logger.info(f"Quality Validation: High quality artwork ({w}x{h}, sharpness: {lap_var:.2f}, contrast: {std_dev:.2f}). No upscaling required.")
    return False

def upscale_and_enhance_sr(cv_img: np.ndarray) -> np.ndarray:
    """
    STEP 1: Upscales the image 2x–4x and applies edge-preserving super-resolution simulation.
    Calculates scale factor dynamically to target a high-quality master resolution (~3840px on the longer side).
    """
    h, w = cv_img.shape[:2]
    max_dim = max(h, w)
    
    # Calculate scale factor to target ~3840px (4K)
    scale = 3840.0 / float(max_dim)
    # Clamp scale between 2.0 and 4.0
    scale = max(2.0, min(4.0, scale))
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info(f"Applying super-resolution enhancement: {w}x{h} -> {new_w}x{new_h} (scale factor: {scale:.2f}x)")
    
    # 1. Upscale using Lanczos4 interpolation (lossless-equivalent detail preservation)
    upscaled = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Apply bilateral filtering to suppress compression artifacts while preserving edges
    d = 5
    sigma = 15.0
    denoised = cv2.bilateralFilter(upscaled, d, sigma, sigma)
    
    # 3. Apply controlled unsharp masking to enhance micro details
    blurred = cv2.GaussianBlur(denoised, (5, 5), 1.0)
    enhanced = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
    
    return enhanced

def remove_defects_and_noise(cv_img: np.ndarray, strength: float) -> np.ndarray:
    """
    STEP 2: CLEANING
    Removes JPEG compression artifacts, scanning defects, dust, scratches, and background noise.
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
    STEP 3: GEOMETRY CORRECTION
    Detects skewed boundaries of the artwork using contour analysis and warps 
    the perspective back to a flat rectangular alignment, maintaining original proportions.
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

def restore_colors(cv_img: np.ndarray, sat_scale: float, clip_limit: float = 2.0) -> np.ndarray:
    """
    STEP 4: COLOR & DETAIL ENHANCEMENT
    Applies CLAHE on the L channel of LAB and boosts saturation in HSV to restore faded colors.
    """
    if len(cv_img.shape) < 3:
        return cv_img
        
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
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
    STEP 4: COLOR & DETAIL ENHANCEMENT
    Enhances sharpness and edge crispness using Unsharp Masking.
    """
    if strength <= 0:
        return cv_img
    blurred = cv2.GaussianBlur(cv_img, (5, 5), 1.0)
    return cv2.addWeighted(cv_img, 1.0 + strength, blurred, -strength, 0)

def enhance_edges(cv_img: np.ndarray, alpha: float) -> np.ndarray:
    """
    STEP 5: LINE ENHANCEMENT
    Blends Laplacian edge map to enforce crisp line boundaries and edge clarity.
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
    Helper to upscale image using high-quality Lanczos interpolation.
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
    and returns required preprocessed versions of the artwork + B&W sketch design.
    """
    logger.info("Running Step: Textile CAD Preprocessing Engine")
    
    # 0. Convert PIL image to OpenCV BGR
    cv_orig = pil_to_cv2(img)
    
    # STEP 1: IMAGE QUALITY VALIDATION (MANDATORY FIRST)
    is_low_quality = analyze_image_quality(cv_orig)
    if is_low_quality:
        logger.info("Quality validation: Low quality artwork. Applying upscale & super-resolution simulation...")
        cv_master = upscale_and_enhance_sr(cv_orig)
    else:
        logger.info("Quality validation: High quality artwork. Keeping original resolution.")
        cv_master = cv_orig.copy()
        
    # STEP 2: CLEANING
    logger.info("Running Step 2: Cleaning")
    # Clean noise, scan lines, dust and artifacts
    cv_cleaned = remove_defects_and_noise(cv_master, strength=2.5)
    
    # STEP 3: GEOMETRY CORRECTION
    logger.info("Running Step 3: Geometry Correction")
    # Correct perspective distortion, skew, align repeat borders
    cv_rectified = correct_perspective(cv_cleaned)
    
    versions = {}
    
    # STEP 4 & 5: COLOR, DETAIL & LINE ENHANCEMENT
    logger.info("Running Step 4 & 5: Color, Detail & Line Enhancement")
    
    # 1. MASTER ENHANCED IMAGE
    logger.debug("Generating CAD Output: master_enhanced (Balanced Restoration)")
    v1_cv = cv_rectified.copy()
    v1_cv = restore_colors(v1_cv, sat_scale=1.1, clip_limit=2.0)
    v1_cv = sharpen_image(v1_cv, strength=0.8)
    v1_cv = enhance_edges(v1_cv, alpha=0.15)
    versions["master_enhanced"] = cv2_to_pil(v1_cv)
    versions["version1"] = versions["master_enhanced"] # Backward compatibility mapping
    
    # 2. BLACK & WHITE SKETCH OUTPUT
    logger.debug("Generating CAD Output: sketch_bw")
    sketch_cv = generate_bw_sketch(cv_rectified)
    versions["sketch_bw"] = cv2_to_pil(sketch_cv)
    
    # 3. COLOR VARIANT A (SOFT ENHANCED)
    logger.debug("Generating CAD Output: color_variant_soft")
    soft_cv = cv_rectified.copy()
    soft_cv = remove_defects_and_noise(soft_cv, strength=1.5)
    soft_cv = restore_colors(soft_cv, sat_scale=1.02, clip_limit=1.2)
    soft_cv = sharpen_image(soft_cv, strength=0.5)
    versions["color_variant_soft"] = cv2_to_pil(soft_cv)
    versions["version3"] = versions["color_variant_soft"] # Backward compatibility mapping
    
    # 4. COLOR VARIANT B (VIBRANT ENHANCED)
    logger.debug("Generating CAD Output: color_variant_vibrant")
    vibrant_cv = cv_rectified.copy()
    vibrant_cv = remove_defects_and_noise(vibrant_cv, strength=2.0)
    vibrant_cv = restore_colors(vibrant_cv, sat_scale=1.25, clip_limit=3.0)
    vibrant_cv = sharpen_image(vibrant_cv, strength=1.2)
    vibrant_cv = enhance_edges(vibrant_cv, alpha=0.2)
    versions["color_variant_vibrant"] = cv2_to_pil(vibrant_cv)
    versions["version2"] = versions["color_variant_vibrant"] # Backward compatibility mapping
    
    # --- Other backward compatibility versions (Version 4, 5, 6) ---
    logger.debug("Generating CAD Version 4: Vectorization optimized")
    v4_cv = remove_defects_and_noise(cv_rectified, strength=5.0)
    v4_cv = restore_colors(v4_cv, sat_scale=1.2, clip_limit=2.0)
    v4_cv = sharpen_image(v4_cv, strength=1.5)
    versions["version4"] = cv2_to_pil(v4_cv)
    
    logger.debug("Generating CAD Version 5: Repeat detection optimized")
    v5_cv = remove_defects_and_noise(cv_rectified, strength=2.0)
    if len(v5_cv.shape) == 3:
        yuv = cv2.cvtColor(v5_cv, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y_eq = cv2.equalizeHist(y)
        v5_cv = cv2.cvtColor(cv2.merge((y_eq, u, v)), cv2.COLOR_YUV2BGR)
    else:
        v5_cv = cv2.equalizeHist(v5_cv)
    v5_cv = sharpen_image(v5_cv, strength=1.2)
    versions["version5"] = cv2_to_pil(v5_cv)
    
    logger.debug("Generating CAD Version 6: Texcelle import optimized")
    v6_cv = remove_defects_and_noise(cv_rectified, strength=3.0)
    v6_cv = restore_colors(v6_cv, sat_scale=1.1, clip_limit=2.0)
    v6_cv = sharpen_image(v6_cv, strength=1.0)
    v6_pil = cv2_to_pil(v6_cv)
    
    # Flat color quantization (reduce to 32 flat color palette for engraving standard)
    quantized_v6 = v6_pil.convert("RGB").quantize(
        colors=32,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE
    ).convert("RGB")
    versions["version6"] = quantized_v6
    
    logger.info("Generated all Textile CAD image versions + sketch successfully.")
    return versions

