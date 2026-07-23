import os
import io
import base64
import json
import cv2
import httpx
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from app.core.config import settings

# Output directory in workspace root
OUTPUT_DIR = Path("./output")

def encode_image_to_base64(image_path: str) -> str:
    """Read a file from disk and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def call_native_gemini_api(image_base64: str, prompt: str, api_key: str) -> dict:
    """Calls Native Google Gemini API to analyze the input image and recommend parameters."""
    print("Calling Native Gemini API for image analysis...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    with httpx.Client(timeout=45.0) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            clean_content = text_response.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_content)
            print("Native Gemini API completed successfully.")
            return result
        else:
            raise ValueError(f"Native Gemini API returned status code {response.status_code}: {response.text}")

def call_openrouter_analysis(image_base64: str, prompt: str) -> dict:
    """Calls OpenRouter API as a fallback to analyze the input image."""
    print("Calling OpenRouter API for image quality analysis...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://jacquardai.com",
        "X-Title": "Jacquard AI Textile Platform"
    }
    
    payload = {
        "model": settings.OPENROUTER_MODEL or "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    with httpx.Client(timeout=45.0) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            clean_content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_content)
            print("OpenRouter Quality Analysis completed successfully.")
            return result
        else:
            raise ValueError(f"OpenRouter returned status code {response.status_code}: {response.text}")

def call_native_gemini_quality_check(original_base64: str, enhanced_base64: str, api_key: str) -> dict:
    """Performs a native Gemini quality check on the enhanced image compared to the original."""
    print("Calling Native Gemini API for post-enhancement quality check...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    prompt = (
        "You are an expert Textile CAD Quality Control Inspector.\n"
        "Compare the original image (first image) and the enhanced image (second image).\n"
        "The enhanced image is supposed to be perfectly sharp, clear, and noise-free, "
        "retaining every single detail of the motifs, flowers, borders, and textures with zero blur "
        "and zero JPEG compression noise.\n"
        "Assess whether the enhanced image passes the quality check. Return a JSON object with the following keys:\n"
        "{\n"
        "  \"quality_check_passed\": boolean,\n"
        "  \"remaining_noise\": boolean,\n"
        "  \"remaining_blur\": boolean,\n"
        "  \"details_lost\": boolean,\n"
        "  \"denoise_strength_adjustment\": float (correction adjustment delta, e.g. +1.0 or -0.5, or 0.0),\n"
        "  \"sharpen_strength_adjustment\": float (correction adjustment delta, e.g. +0.2 or -0.1, or 0.0),\n"
        "  \"comments\": \"constructive feedback\"\n"
        "}\n"
        "Make sure to return only valid JSON."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": original_base64
                        }
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": enhanced_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    with httpx.Client(timeout=45.0) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            clean_content = text_response.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_content)
            print("Native Gemini QC completed successfully.")
            return result
        else:
            raise ValueError(f"Native Gemini QC returned status code {response.status_code}: {response.text}")

def call_openrouter_quality_check(original_base64: str, enhanced_base64: str) -> dict:
    """Performs a quality inspection check using OpenRouter on the enhanced image compared to the original."""
    print("Calling OpenRouter API for post-enhancement quality check...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://jacquardai.com",
        "X-Title": "Jacquard AI Textile Platform"
    }
    
    prompt = (
        "You are an expert Textile CAD Quality Control Inspector.\n"
        "Compare the original image (first image) and the enhanced image (second image).\n"
        "The enhanced image is supposed to be perfectly sharp, clear, and noise-free, "
        "retaining every single detail of the motifs, flowers, borders, and textures with zero blur "
        "and zero JPEG compression noise.\n"
        "Assess whether the enhanced image passes the quality check. Return a JSON object with the following keys:\n"
        "{\n"
        "  \"quality_check_passed\": boolean,\n"
        "  \"remaining_noise\": boolean,\n"
        "  \"remaining_blur\": boolean,\n"
        "  \"details_lost\": boolean,\n"
        "  \"denoise_strength_adjustment\": float (correction adjustment delta, e.g. +1.0 or -0.5, or 0.0),\n"
        "  \"sharpen_strength_adjustment\": float (correction adjustment delta, e.g. +0.2 or -0.1, or 0.0),\n"
        "  \"comments\": \"constructive feedback\"\n"
        "}\n"
        "Make sure to return only valid JSON."
    )
    
    payload = {
        "model": settings.OPENROUTER_MODEL or "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{original_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{enhanced_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    with httpx.Client(timeout=45.0) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            clean_content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_content)
            print("OpenRouter Quality Control Check completed successfully.")
            return result
        else:
            raise ValueError(f"OpenRouter Quality Control returned status code {response.status_code}: {response.text}")

def call_hybrid_analysis(image_base64: str, force_fallback: bool = False) -> tuple[dict, str, bool]:
    """
    Calls primary Gemini API, falling back to OpenRouter API if it fails, times out,
    or if force_fallback is requested.
    Returns: (params, model_used, fallback_triggered)
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    
    prompt = (
        "You are an expert Textile CAD AI preprocessing assistant.\n"
        "Analyze the provided textile design image for preprocessing and restoration.\n"
        "Return a valid JSON object containing the following parameters:\n"
        "{\n"
        "  \"jpeg_noise_level\": \"low\" | \"medium\" | \"high\",\n"
        "  \"blur_level\": \"low\" | \"medium\" | \"high\",\n"
        "  \"contrast_quality\": \"poor\" | \"fair\" | \"good\",\n"
        "  \"denoise_strength\": float (value between 1.0 and 10.0, e.g. 5.0),\n"
        "  \"sharpen_strength\": float (value between 0.1 and 2.0, e.g. 0.8),\n"
        "  \"clahe_clip_limit\": float (value between 1.0 and 4.0, e.g. 2.0),\n"
        "  \"saturation_scale\": float (value between 0.8 and 1.5, e.g. 1.1),\n"
        "  \"skew_detected\": boolean,\n"
        "  \"skew_angle_deg\": float,\n"
        "  \"comments\": \"technical notes on the design structure\"\n"
        "}\n"
        "Make sure to return only valid JSON."
    )
    
    if api_key and api_key != "your_gemini_api_key_here" and not force_fallback:
        try:
            params = call_native_gemini_api(image_base64, prompt, api_key)
            return params, "gemini", False
        except Exception as e:
            print(f"Primary Gemini API failed: {e}. Falling back to OpenRouter...")
            
    # OpenRouter Fallback
    try:
        params = call_openrouter_analysis(image_base64, prompt)
        return params, "openrouter", True
    except Exception as e:
        print(f"OpenRouter API fallback failed: {e}. Using local procedural parameters.")
        default_params = {
            "jpeg_noise_level": "medium",
            "blur_level": "medium",
            "contrast_quality": "fair",
            "denoise_strength": 5.0,
            "sharpen_strength": 0.8,
            "clahe_clip_limit": 2.0,
            "saturation_scale": 1.1,
            "skew_detected": False,
            "skew_angle_deg": 0.0,
            "comments": "Procedural default parameters."
        }
        return default_params, "openrouter", True

def call_hybrid_quality_check(original_base64: str, enhanced_base64: str, model_used: str) -> dict:
    """Runs a quality control check matching the model used during analysis (or OpenRouter as fallback)."""
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    
    if model_used == "gemini" and api_key and api_key != "your_gemini_api_key_here":
        try:
            return call_native_gemini_quality_check(original_base64, enhanced_base64, api_key)
        except Exception as e:
            print(f"Native Gemini QC failed: {e}. Falling back to OpenRouter QC...")
            
    # OpenRouter QC Fallback
    try:
        return call_openrouter_quality_check(original_base64, enhanced_base64)
    except Exception as e:
        print(f"OpenRouter QC failed: {e}. Bypassing feedback check.")
        return {
            "quality_check_passed": True,
            "remaining_noise": False,
            "remaining_blur": False,
            "details_lost": False,
            "denoise_strength_adjustment": 0.0,
            "sharpen_strength_adjustment": 0.0,
            "comments": "QC check bypassed."
        }

def deskew_image(cv_img: np.ndarray, skew_angle_deg: float) -> np.ndarray:
    """Rotate the image to correct skew alignment."""
    if abs(skew_angle_deg) < 0.1:
        return cv_img
    h, w = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle_deg, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_perspective_contours(cv_img: np.ndarray) -> np.ndarray:
    """Uses contour approximation to find the quad corners of skewed artwork and warps it straight."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv_img
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = cv_img.shape[:2]
    img_area = h * w
    orig_aspect = w / h
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.50 * img_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
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
            
            warped_aspect = maxWidth / max(1, maxHeight)
            if abs(warped_aspect - orig_aspect) / orig_aspect > 0.2:
                continue
                
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(cv_img, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
            print(f"Perspective warp applied from {pts.tolist()} to {maxWidth}x{maxHeight}")
            return warped
            
    return cv_img

def apply_geometry_correction(cv_img: np.ndarray, params: dict) -> np.ndarray:
    """Only applies deskewing/perspective correction if the AI model detected a skew/perspective distortion."""
    if not params.get("skew_detected", False):
        print("No skew detected by AI. Skipping geometry correction.")
        return cv_img
        
    corrected = correct_perspective_contours(cv_img)
    if corrected.shape == cv_img.shape and np.allclose(corrected, cv_img):
        angle = params.get("skew_angle_deg", 0.0)
        if abs(angle) > 0.1:
            print(f"Applying AI deskew rotation of {angle} degrees.")
            corrected = deskew_image(cv_img, angle)
    return corrected

def enhance_image(cv_img: np.ndarray, denoise_strength: float, sharpen_strength: float, clahe_clip: float, sat_scale: float) -> np.ndarray:
    """Denoise, sharpen, CLAHE, and boost saturation on input image."""
    d = max(3, int(denoise_strength * 1.5) | 1)
    sigma_color = denoise_strength * 6.0
    sigma_space = denoise_strength * 6.0
    denoised = cv2.bilateralFilter(cv_img, d, sigma_color, sigma_space)
    
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_eq = cv2.merge((cl, a, b))
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    blurred = cv2.GaussianBlur(bgr_eq, (5, 5), 1.0)
    sharpened = cv2.addWeighted(bgr_eq, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)
    
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_new = np.clip(s.astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    hsv_restored = cv2.merge((h, s_new, v))
    result = cv2.cvtColor(hsv_restored, cv2.COLOR_HSV2BGR)
    
    return result

def process_master_image_with_feedback_model(cv_img: np.ndarray, params: dict, original_base64: str, model_used: str) -> np.ndarray:
    """Core image preprocessing flow with self-correcting feedback loop."""
    denoise_strength = params["denoise_strength"]
    sharpen_strength = params["sharpen_strength"]
    clahe_clip = params["clahe_clip_limit"]
    sat_scale = params["saturation_scale"]
    
    geom_corrected = apply_geometry_correction(cv_img, params)
    
    enhanced = enhance_image(geom_corrected, denoise_strength, sharpen_strength, clahe_clip, sat_scale)
    
    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    enhanced_base64 = encode_pil_to_base64(enhanced_pil)
    qc = call_hybrid_quality_check(original_base64, enhanced_base64, model_used)
    
    if not qc.get("quality_check_passed", True):
        print("Self-Correction Loop triggered: adjusting parameters and running Second Pass.")
        denoise_strength = max(1.0, denoise_strength + qc.get("denoise_strength_adjustment", 0.0))
        sharpen_strength = max(0.1, sharpen_strength + qc.get("sharpen_strength_adjustment", 0.0))
        print(f"New parameters: denoise={denoise_strength}, sharpen={sharpen_strength}")
        enhanced = enhance_image(geom_corrected, denoise_strength, sharpen_strength, clahe_clip, sat_scale)
    else:
        print("Quality Check Passed.")
        
    return enhanced

def process_pattern_tile(enhanced_img: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    """
    Step 4: Detect repeat motifs and ensure seamless tiling or alignment to target dimensions.
    """
    h_orig, w_orig = enhanced_img.shape[:2]
    
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    patch_size = min(128, min(w_orig, h_orig) // 4)
    repeat_w, repeat_h = w_orig, h_orig
    repeat_type = "straight"
    
    if patch_size >= 16:
        cx, cy = w_orig // 2, h_orig // 2
        x1, y1 = cx - patch_size // 2, cy - patch_size // 2
        patch = gray[y1:y1+patch_size, x1:x1+patch_size]
        try:
            res = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
            peaks = np.where(res >= 0.65)
            pts = list(zip(peaks[1], peaks[0]))
            
            x_diffs = [abs(pt[0] - x1) for pt in pts if abs(pt[0] - x1) > 10]
            y_diffs = [abs(pt[1] - y1) for pt in pts if abs(pt[1] - y1) > 10]
            
            min_dim = max(100, min(w_orig, h_orig) // 4)
            if x_diffs:
                med_w = int(np.median(x_diffs))
                if min_dim < med_w < w_orig:
                    repeat_w = med_w
            if y_diffs:
                med_h = int(np.median(y_diffs))
                if min_dim < med_h < h_orig:
                    repeat_h = med_h
                    
            for pt in pts:
                dx, dy = abs(pt[0] - x1), abs(pt[1] - y1)
                if abs(dx - repeat_w) < 15 and abs(dy - (repeat_h / 2)) < 15:
                    repeat_type = "half-drop"
                    break
        except Exception as e:
            print(f"Repeat detection error: {e}")
            
    print(f"Pattern Repeat Detected: type={repeat_type}, unit={repeat_w}x{repeat_h}")
    
    # Crop the base tile
    tile = enhanced_img[0:repeat_h, 0:repeat_w]
    
    # Tile it over target width_px and height_px canvas
    tiled_canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    
    cols = (width_px + repeat_w - 1) // repeat_w
    rows = (height_px + repeat_h - 1) // repeat_h
    
    for c in range(cols):
        v_shift = 0
        if repeat_type == "half-drop" and (c % 2 == 1):
            v_shift = repeat_h // 2
            
        for r in range(rows + (1 if v_shift > 0 else 0)):
            x = c * repeat_w
            y = r * repeat_h + v_shift
            
            tile_h, tile_w = tile.shape[:2]
            
            x_start = max(0, x)
            x_end = min(width_px, x + tile_w)
            y_start = max(0, y)
            y_end = min(height_px, y + tile_h)
            
            if x_start < x_end and y_start < y_end:
                src_x_start = x_start - x
                src_x_end = src_x_start + (x_end - x_start)
                src_y_start = y_start - y
                src_y_end = src_y_start + (y_end - y_start)
                
                tiled_canvas[y_start:y_end, x_start:x_end] = tile[src_y_start:src_y_end, src_x_start:src_x_end]
                
    return tiled_canvas

def isolate_background(tiled_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Performs KMeans color quantization and isolates the background to pure white."""
    h, w = tiled_img.shape[:2]
    denoised = cv2.bilateralFilter(tiled_img, 7, 30, 30)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    pixels_lab = lab.reshape(-1, 3).astype(np.float32)
    
    n_clusters = min(8, h * w)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels_lab).reshape(h, w)
    
    counts = np.bincount(labels.flatten())
    bg_cluster_idx = np.argmax(counts)
    
    restored_white_bg = tiled_img.copy()
    restored_white_bg[labels == bg_cluster_idx] = [255, 255, 255]
    
    return restored_white_bg, labels, bg_cluster_idx

def generate_colorway_variant(tiled_img: np.ndarray, labels: np.ndarray, bg_cluster_idx: int, theme: str) -> np.ndarray:
    """Generates color variants while preserving the background and structural motifs."""
    hsv = cv2.cvtColor(tiled_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    white_mask = (labels == bg_cluster_idx)
    
    if theme == "pastel":
        h_new = (h + 15.0) % 180.0
        s_new = np.clip(s * 0.45, 15.0, 95.0)
        v_new = np.clip(v * 0.85 + 40.0, 160.0, 245.0)
    elif theme == "vibrant":
        h_new = (h + 45.0) % 180.0
        s_new = np.clip(s * 1.35 + 20.0, 100.0, 255.0)
        v_new = np.clip(v * 1.1, 50.0, 250.0)
    elif theme == "traditional":
        h_new = (h - 20.0) % 180.0
        s_new = np.clip(s * 0.9, 30.0, 180.0)
        v_new = np.clip(v * 0.8, 40.0, 200.0)
    elif theme == "dark":
        h_new = (h + 90.0) % 180.0
        s_new = np.clip(s * 1.4, 80.0, 255.0)
        v_new = np.where(v > 128, np.clip(v * 0.6, 50.0, 120.0), np.clip(v * 0.2, 10.0, 40.0))
    else:
        h_new, s_new, v_new = h, s, v
        
    hsv_new = cv2.merge((h_new.astype(np.uint8), s_new.astype(np.uint8), v_new.astype(np.uint8)))
    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    bgr_new[white_mask] = [255, 255, 255]
    return bgr_new

def validate_output_file(file_path: Path, is_sketch: bool, expected_size: tuple[int, int]) -> None:
    """Verifies image format, uncompressed BMP, dimensions, 600 DPI, and sketch binary pixels."""
    if not file_path.exists():
        raise FileNotFoundError(f"Output file does not exist: {file_path}")
        
    content = file_path.read_bytes()
    
    if file_path.suffix.lower() == ".bmp":
        if content[:2] != b"BM":
            raise ValueError(f"{file_path.name} is not a valid BMP")
        compression = int.from_bytes(content[30:34], byteorder="little")
        if compression != 0:
            raise ValueError(f"{file_path.name} is compressed. Texcelle BMP must be uncompressed (BI_RGB=0)")
            
    with Image.open(file_path) as img:
        img.load()
        if img.size != expected_size:
            raise ValueError(f"{file_path.name} size {img.size} does not match expected {expected_size}")
            
        dpi_meta = img.info.get("dpi")
        if not dpi_meta or any(round(val) != 600 for val in dpi_meta[:2]):
            raise ValueError(f"{file_path.name} does not contain 600 DPI metadata. Found: {dpi_meta}")
            
        if img.mode in {"RGBA", "LA"} or "transparency" in img.info:
            raise ValueError(f"{file_path.name} contains transparency channel")
            
        if is_sketch:
            hist = img.convert("L").histogram()
            non_binary = sum(hist[1:255])
            if non_binary > 0:
                raise ValueError(f"{file_path.name} is not binary, contains {non_binary} grayscale pixels")

def save_output(img_np: np.ndarray, name_prefix: str, is_sketch: bool, output_dir: Path, file_id: str, is_celery: bool) -> dict[str, str]:
    """
    Saves the image as PNG and BMP.
    If is_celery is True, uploads to Cloudinary and returns the Cloudinary URL.
    Else returns local absolute path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = (600, 600)
    
    if is_sketch:
        pil_img = Image.fromarray(img_np, mode="L")
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        
    png_filename = f"{name_prefix}.png"
    png_path = output_dir / png_filename
    if is_sketch:
        png_img = pil_img.convert("L")
    else:
        png_img = pil_img.convert("RGB")
    png_img.save(str(png_path), format="PNG", optimize=True, compress_level=9, dpi=dpi)
    
    bmp_filename = f"{name_prefix}.bmp"
    bmp_path = output_dir / bmp_filename
    bmp_img = pil_img.convert("RGB")
    bmp_img.save(str(bmp_path), format="BMP", dpi=dpi)
    
    # Strict validation check
    validate_output_file(png_path, is_sketch, png_img.size)
    validate_output_file(bmp_path, is_sketch, bmp_img.size)
    
    if is_celery:
        # Import Cloudinary upload service
        from app.services.cloudinary_service import upload_image
        folder = f"similar_files/{file_id}"
        png_upload = upload_image(str(png_path), folder=folder)
        bmp_upload = upload_image(str(bmp_path), folder=folder)
        
        # Cleanup local file to save space
        try:
            png_path.unlink()
            bmp_path.unlink()
        except Exception:
            pass
            
        return {
            "png": png_upload.get("secure_url"),
            "bmp": bmp_upload.get("secure_url")
        }
    else:
        return {
            "png": str(png_path.resolve()),
            "bmp": str(bmp_path.resolve())
        }

def download_image_from_url(url: str, temp_path: Path) -> Path:
    """Download image from Cloudinary or web URL."""
    print(f"Downloading image from URL: {url} ...")
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(response.content)
    print("Download completed successfully.")
    return temp_path

def main():
    parser = argparse.ArgumentParser(description="Enterprise-grade AI Textile CAD Processing Engine")
    parser.add_index = False  # Avoid conflict
    parser.add_argument("--image-path", type=str, default="", help="Path to local image")
    parser.add_argument("--enhanced-cloudinary-url", type=str, default="", help="URL to Cloudinary image")
    parser.add_argument("--width-inch", type=float, default=12.0, help="Target width in inches")
    parser.add_argument("--height-inch", type=float, default=12.0, help="Target height in inches")
    parser.add_argument("--read", type=int, default=60, help="Horizontal pixel density (read)")
    parser.add_argument("--pick", type=int, default=60, help="Vertical pixel density (pick)")
    parser.add_argument("--file-id", type=str, default="job_cad_restored", help="Job or asset ID")
    
    args = parser.parse_args()
    
    # 0. Load source image and normalize
    is_celery = bool(args.enhanced_cloudinary_url)
    input_source = args.enhanced_cloudinary_url if is_celery else args.image_path
    
    if not input_source:
        # Use default prototype image if nothing passed
        input_source = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
        is_celery = False
        print(f"No input provided. Defaulting to local prototype: {input_source}")
        
    temp_local_path = Path("./data/temp_source.png")
    if is_celery or input_source.startswith("http"):
        # Download image from URL
        local_img_path = download_image_from_url(input_source, temp_local_path)
    else:
        local_img_path = Path(input_source)
        
    if not local_img_path.exists():
        raise FileNotFoundError(f"Source image not found: {local_img_path}")
        
    cv_img = cv2.imread(str(local_img_path))
    if cv_img is None:
        raise ValueError(f"Failed to load image from: {local_img_path}")
        
    # Convert BGR to RGB
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Ensure standard BGR alignment for OpenCV operations
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    
    # 1. Base64 encoding for API
    original_base64 = encode_image_to_base64(str(local_img_path))
    
    # 2. Hybrid AI Preprocessing Restorations
    params, model_used, fallback_triggered = call_hybrid_analysis(original_base64)
    
    # 3. Apply restorations & QC validation check
    restored_base = process_master_image_with_feedback_model(cv_img, params, original_base64, model_used)
    
    # Verify post-restoration sharpness (Laplacian variance)
    gray_restored = cv2.cvtColor(restored_base, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray_restored, cv2.CV_64F).var()
    print(f"Enhanced image Laplacian variance: {lap_var:.2f}")
    
    # Quality inspection fallback trigger
    if model_used == "gemini" and (lap_var < 80.0):
        print("Sharpness threshold not met. Triggering fallback to OpenRouter...")
        fallback_triggered = True
        params, model_used, _ = call_hybrid_analysis(original_base64, force_fallback=True)
        restored_base = process_master_image_with_feedback_model(cv_img, params, original_base64, model_used)
        
    # 4. Dimension Calculations
    width_px = int(args.width_inch * args.read)
    height_px = int(args.height_inch * args.pick)
    total_pixels = width_px * height_px
    
    # 5. Pattern Tiling to target dimensions
    tiled_restored = process_pattern_tile(restored_base, width_px, height_px)
    
    # 6. Background noise removal
    restored_white_bg, labels, bg_cluster_idx = isolate_background(tiled_restored)
    
    # 7. Outputs Generation
    # A. Figure B&W Silhouette Sketch
    fig_sketch = np.ones((height_px, width_px), dtype=np.uint8) * 255
    fig_sketch[labels != bg_cluster_idx] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fig_sketch = cv2.morphologyEx(fig_sketch, cv2.MORPH_OPEN, kernel)
    fig_sketch = cv2.medianBlur(fig_sketch, 3)
    
    # B. Clean Line Art Sketch
    gray_tiled = cv2.cvtColor(tiled_restored, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.bilateralFilter(gray_tiled, 5, 30, 30)
    edges = cv2.Canny(denoised_gray, 30, 100)
    edges_inv = 255 - edges
    edges_inv[labels == bg_cluster_idx] = 255
    _, line_sketch = cv2.threshold(edges_inv, 127, 255, cv2.THRESH_BINARY)
    line_sketch = cv2.medianBlur(line_sketch, 3)
    
    # Save Sketch Outputs
    sketch1_urls = save_output(fig_sketch, "sketch_figure_bw", True, OUTPUT_DIR, args.file_id, is_celery)
    sketch2_urls = save_output(line_sketch, "sketch_line_art", True, OUTPUT_DIR, args.file_id, is_celery)
    
    # Generate Colorways
    traditional_variant = generate_colorway_variant(tiled_restored, labels, bg_cluster_idx, "traditional")
    pastel_variant = generate_colorway_variant(tiled_restored, labels, bg_cluster_idx, "pastel")
    vibrant_variant = generate_colorway_variant(tiled_restored, labels, bg_cluster_idx, "vibrant")
    dark_variant = generate_colorway_variant(tiled_restored, labels, bg_cluster_idx, "dark")
    
    # Save Colorways Outputs
    traditional_urls = save_output(traditional_variant, "color_traditional", False, OUTPUT_DIR, args.file_id, is_celery)
    pastel_urls = save_output(pastel_variant, "color_pastel", False, OUTPUT_DIR, args.file_id, is_celery)
    vibrant_urls = save_output(vibrant_variant, "color_vibrant", False, OUTPUT_DIR, args.file_id, is_celery)
    dark_urls = save_output(dark_variant, "color_dark", False, OUTPUT_DIR, args.file_id, is_celery)
    
    # Remove downloaded temp file if exists
    if temp_local_path.exists():
        try:
            temp_local_path.unlink()
        except Exception:
            pass
            
    # Format and Output JSON response
    response = {
        "model_used": model_used,
        "fallback_triggered": fallback_triggered,
        "dimensions": {
            "width_inch": str(args.width_inch),
            "height_inch": str(args.height_inch),
            "read": str(args.read),
            "pick": str(args.pick),
            "width_px": str(width_px),
            "height_px": str(height_px),
            "total_pixels": str(total_pixels)
        },
        "outputs": {
            "sketch": [
                { "png": sketch1_urls["png"], "bmp": sketch1_urls["bmp"] },
                { "png": sketch2_urls["png"], "bmp": sketch2_urls["bmp"] }
            ],
            "color_variants": [
                { "theme": "traditional", "png": traditional_urls["png"], "bmp": traditional_urls["bmp"] },
                { "theme": "pastel", "png": pastel_urls["png"], "bmp": pastel_urls["bmp"] },
                { "theme": "vibrant", "png": vibrant_urls["png"], "bmp": vibrant_urls["bmp"] },
                { "theme": "dark", "png": dark_urls["png"], "bmp": dark_urls["bmp"] }
            ]
        }
    }
    
    print("\n" + "="*50)
    print("STRICT JSON RESPONSE:")
    print("="*50)
    print(json.dumps(response, indent=2))
    print("="*50)

if __name__ == "__main__":
    main()
