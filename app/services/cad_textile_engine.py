import os
import io
import json
import cv2
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any, Union, Tuple, List
from pathlib import Path
from sklearn.cluster import KMeans

from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2, cv2_to_pil
from app.core.config import settings
from app.services.openrouter_service import openrouter_service

class TextileCADEngine:
    """
    Expert Textile CAD AI Engine for Texcelle and Industrial Textile Workflows.
    Performs Pixel-to-Pixel color quantization, dominant background isolation,
    Pencil & Fine Line Art shading effect generation, and Texcelle CAD tracing.
    """

    @staticmethod
    def enhance_image_with_params(cv_img: np.ndarray, denoise_strength: float, sharpen_strength: float, clahe_clip: float, sat_scale: float) -> np.ndarray:
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
        return cv2.cvtColor(hsv_restored, cv2.COLOR_HSV2BGR)

    @staticmethod
    def correct_perspective_contours(cv_img: np.ndarray) -> np.ndarray:
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
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
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
                return cv2.warpPerspective(cv_img, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
        return cv_img

    @staticmethod
    def generate_master_clean(cv_img: np.ndarray) -> np.ndarray:
        """
        Generates high-resolution master clean image with noise and scanning artifacts removed.
        Uses AI-guided Gemini restoration and post-QC self-correction if OpenRouter is available.
        """
        if settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            try:
                logger.info("Running AI-Guided Image Restoration on master clean")
                _, buffer = cv2.imencode(".png", cv_img)
                base64_original = base64.b64encode(buffer).decode("utf-8")
                
                prompt = (
                    "You are an expert Textile CAD AI preprocessing assistant.\n"
                    "Analyze the provided textile design image for preprocessing and restoration.\n"
                    "Return a valid JSON object containing parameters:\n"
                    "{\n"
                    "  \"skew_detected\": boolean,\n"
                    "  \"skew_angle_deg\": float,\n"
                    "  \"denoise_strength\": float (value between 1.0 and 10.0, e.g. 5.0),\n"
                    "  \"sharpen_strength\": float (value between 0.1 and 2.0, e.g. 0.8),\n"
                    "  \"clahe_clip_limit\": float (value between 1.0 and 4.0, e.g. 2.0),\n"
                    "  \"saturation_scale\": float (value between 0.8 and 1.5, e.g. 1.1)\n"
                    "}\n"
                    "Only return valid JSON."
                )
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_original}"
                                }
                            }
                        ]
                    }
                ]
                
                analysis_response = openrouter_service.call_chat_completion(messages)
                content = analysis_response["choices"][0]["message"]["content"]
                clean_content = content.replace("```json", "").replace("```", "").strip()
                params = json.loads(clean_content)
                
                geom_img = cv_img.copy()
                if params.get("skew_detected", False):
                    geom_img = TextileCADEngine.correct_perspective_contours(cv_img)
                    if geom_img.shape == cv_img.shape and np.allclose(geom_img, cv_img):
                        angle = params.get("skew_angle_deg", 0.0)
                        if abs(angle) > 0.1:
                            h_rot, w_rot = cv_img.shape[:2]
                            center = (w_rot // 2, h_rot // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            geom_img = cv2.warpAffine(cv_img, M, (w_rot, h_rot), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
                
                enhanced = TextileCADEngine.enhance_image_with_params(
                    geom_img, 
                    params.get("denoise_strength", 5.0),
                    params.get("sharpen_strength", 0.8),
                    params.get("clahe_clip_limit", 2.0),
                    params.get("saturation_scale", 1.1)
                )
                
                _, enhanced_buffer = cv2.imencode(".png", enhanced)
                base64_enhanced = base64.b64encode(enhanced_buffer).decode("utf-8")
                
                qc_prompt = (
                    "You are an expert Textile CAD Quality Control Inspector.\n"
                    "Compare the original image (first image) and the enhanced image (second image).\n"
                    "Assess whether the enhanced image passes the quality check. Return a JSON object with keys:\n"
                    "{\n"
                    "  \"quality_check_passed\": boolean,\n"
                    "  \"denoise_strength_adjustment\": float,\n"
                    "  \"sharpen_strength_adjustment\": float\n"
                    "}\n"
                    "Only return valid JSON."
                )
                
                qc_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": qc_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_original}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_enhanced}"
                                }
                            }
                        ]
                    }
                ]
                
                qc_response = openrouter_service.call_chat_completion(qc_messages)
                qc_content = qc_response["choices"][0]["message"]["content"]
                qc_clean = qc_content.replace("```json", "").replace("```", "").strip()
                qc = json.loads(qc_clean)
                
                if not qc.get("quality_check_passed", True):
                    logger.info("Self-correction triggered in engine master clean")
                    denoise_strength = max(1.0, params.get("denoise_strength", 5.0) + qc.get("denoise_strength_adjustment", 0.0))
                    sharpen_strength = max(0.1, params.get("sharpen_strength", 0.8) + qc.get("sharpen_strength_adjustment", 0.0))
                    enhanced = TextileCADEngine.enhance_image_with_params(
                        geom_img,
                        denoise_strength,
                        sharpen_strength,
                        params.get("clahe_clip_limit", 2.0),
                        params.get("saturation_scale", 1.1)
                    )
                return enhanced
            except Exception as e:
                logger.error(f"Error in AI-guided master clean: {e}. Falling back to procedural flow.")
                
        # Procedural Fallback
        denoised = cv2.bilateralFilter(cv_img, d=7, sigmaColor=25.0, sigmaSpace=25.0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab_restored = cv2.merge((cl, a, b))
        restored = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(restored, (5, 5), 1.0)
        master = cv2.addWeighted(restored, 1.35, blurred, -0.35, 0)
        return master

    @staticmethod
    def generate_pencil_sketch_art(cv_img: np.ndarray, paper_background: bool = True) -> np.ndarray:
        """
        HAND-DRAWN PENCIL & FINE LINE ART SHADING EFFECT:
        Generates detailed pencil shading, delicate line hatching, and elegant contour art
        on a clean warm paper/white canvas matching professional textile engraving standards.
        """
        h, w = cv_img.shape[:2]

        # 1. Bilateral filtering for smooth shading gradients
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        denoised_gray = cv2.bilateralFilter(gray, 7, 35, 35)

        # 2. Division Pencil Sketch (Creates soft tonal pencil shading)
        inv_gray = 255 - denoised_gray
        blur_pencil = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        pencil_division = cv2.divide(denoised_gray, 255 - blur_pencil, scale=256)

        # 3. Fine Edge Outlines (Canny & Adaptive Threshold)
        adaptive_lines = cv2.adaptiveThreshold(denoised_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)
        canny_edges = cv2.Canny(denoised_gray, 35, 110)
        canny_inv = 255 - canny_edges

        line_contours = cv2.bitwise_and(adaptive_lines, canny_inv)
        line_contours = cv2.medianBlur(line_contours, 3)

        # Combine soft pencil shading with crisp contour lines
        combined_sketch = cv2.bitwise_and(pencil_division, line_contours)

        # 4. Background Isolation (Isolate deep ground to clean paper white)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        pixels_lab = lab.reshape(-1, 3).astype(np.float32)

        n_clusters = min(8, h * w)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels_lab).reshape(h, w)
        counts = np.bincount(labels.flatten())
        bg_cluster_idx = np.argmax(counts)
        bg_color_lab = kmeans.cluster_centers_[bg_cluster_idx]

        diff = pixels_lab - bg_color_lab
        dist = np.sqrt(np.sum(diff ** 2, axis=1)).reshape(h, w)
        bg_mask = (dist <= 18.0)

        # Apply pure clean background canvas
        master_sketch = combined_sketch.copy()
        master_sketch[bg_mask] = 255

        if paper_background:
            master_sketch_bgr = cv2.cvtColor(master_sketch, cv2.COLOR_GRAY2BGR)
            paper_tint = np.zeros_like(cv_img)
            paper_tint[:, :] = [241, 247, 249] # Warm paper cream tone
            result_bgr = cv2.addWeighted(master_sketch_bgr, 0.92, paper_tint, 0.08, 0)
            return result_bgr
        else:
            return master_sketch

    @staticmethod
    def generate_quantized_motifs_white_bg(cv_img: np.ndarray, num_colors: int = 8) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        LAB Color Quantization with Pure White Background Isolation.
        """
        h, w = cv_img.shape[:2]
        denoised = cv2.bilateralFilter(cv_img, d=9, sigmaColor=50.0, sigmaSpace=50.0)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        pixels_lab = lab.reshape(-1, 3).astype(np.float32)

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels_lab).reshape(h, w)

        centers_bgr = cv2.cvtColor(
            kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3),
            cv2.COLOR_LAB2BGR
        ).reshape(-1, 3)

        quantized_bgr = centers_bgr[labels]

        counts = np.bincount(labels.flatten())
        bg_cluster_idx = np.argmax(counts)

        quantized_bgr[labels == bg_cluster_idx] = [255, 255, 255]

        return quantized_bgr, labels, bg_cluster_idx

    @staticmethod
    def generate_figure_bw_texcelle(cv_img: np.ndarray) -> np.ndarray:
        """
        INDUSTRIAL TEXCELLE CAD SKETCH:
        Extracts crisp black boundary lines (0) for ALL motif figures and interior details
        against a pure clean white background (255) using multi-scale edge fusion.
        """
        quantized_bgr, labels, bg_cluster_idx = TextileCADEngine.generate_quantized_motifs_white_bg(cv_img, num_colors=8)
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 5, 30, 30)
        
        adaptive_bw = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )
        
        edges = cv2.Canny(denoised, 30, 100)
        edges_inv = cv2.bitwise_not(edges)
        
        # Region boundaries
        label_pad = np.pad(labels, 1, mode='edge')
        diff_h = (labels != label_pad[1:-1, 2:]) | (labels != label_pad[1:-1, :-2])
        diff_v = (labels != label_pad[2:, 1:-1]) | (labels != label_pad[:-2, 1:-1])
        boundaries = (diff_h | diff_v).astype(np.uint8) * 255
        boundaries_inv = 255 - boundaries
        
        combined = cv2.bitwise_and(adaptive_bw, edges_inv)
        combined = cv2.bitwise_and(combined, boundaries_inv)
        
        cleaned = cv2.medianBlur(combined, 3)
        
        # Force background mask to white
        white_mask = (labels == bg_cluster_idx)
        cleaned[white_mask] = 255
        
        _, bw_binary = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        return bw_binary

    @staticmethod
    def generate_pixel_quantized_cad(cv_img: np.ndarray, num_colors: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        PIXEL-TO-PIXEL COLOR QUANTIZATION WITH GRID OVERLAY:
        """
        quantized_bgr, labels, _ = TextileCADEngine.generate_quantized_motifs_white_bg(cv_img, num_colors=num_colors)
        h, w = labels.shape

        grid_view = quantized_bgr.copy()
        grid_spacing = max(10, min(32, w // 40))
        for x in range(0, w, grid_spacing):
            cv2.line(grid_view, (x, 0), (x, h), (210, 210, 210), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(grid_view, (0, y), (w, y), (210, 210, 210), 1)

        return quantized_bgr, grid_view

    @staticmethod
    def generate_4_colorway_variants(master_img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates 4 distinct production colorways with pure white background directly from
        the high-fidelity master image, preserving micro-textures and gradients.
        """
        _, labels, bg_cluster_idx = TextileCADEngine.generate_quantized_motifs_white_bg(master_img, num_colors=8)
        
        hsv = cv2.cvtColor(master_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        white_mask = (labels == bg_cluster_idx)
        
        # 1. Soft Pastel Palette
        h_soft = (h + 15.0) % 180.0
        s_soft = np.clip(s * 0.45, 15.0, 95.0)
        v_soft = np.clip(v * 0.85 + 40.0, 160.0, 245.0)
        soft_hsv = cv2.merge((h_soft.astype(np.uint8), s_soft.astype(np.uint8), v_soft.astype(np.uint8)))
        soft_bgr = cv2.cvtColor(soft_hsv, cv2.COLOR_HSV2BGR)
        soft_bgr[white_mask] = [255, 255, 255]
        
        # 2. High Contrast Industrial
        h_contrast = (h * 1.2 + 90.0) % 180.0
        s_contrast = np.clip(s * 1.5, 80.0, 255.0)
        v_contrast = np.where(v > 128, np.clip(v * 1.2, 180.0, 255.0), np.clip(v * 0.5, 0.0, 90.0))
        contrast_hsv = cv2.merge((h_contrast.astype(np.uint8), s_contrast.astype(np.uint8), v_contrast.astype(np.uint8)))
        contrast_bgr = cv2.cvtColor(contrast_hsv, cv2.COLOR_HSV2BGR)
        contrast_bgr[white_mask] = [255, 255, 255]
        
        # 3. Vibrant Fashion Palette
        h_vibrant = (h + 45.0) % 180.0
        s_vibrant = np.clip(s * 1.35 + 20.0, 100.0, 255.0)
        v_vibrant = np.clip(v * 1.1, 50.0, 250.0)
        vibrant_hsv = cv2.merge((h_vibrant.astype(np.uint8), s_vibrant.astype(np.uint8), v_vibrant.astype(np.uint8)))
        vibrant_bgr = cv2.cvtColor(vibrant_hsv, cv2.COLOR_HSV2BGR)
        vibrant_bgr[white_mask] = [255, 255, 255]
        
        # 4. Monochrome Artistic
        gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
        mono_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        mono_bgr[white_mask] = [255, 255, 255]
        
        return {
            "soft": soft_bgr,
            "contrast": contrast_bgr,
            "vibrant": vibrant_bgr,
            "mono": mono_bgr
        }

    @staticmethod
    def segment_figures(cv_img: np.ndarray) -> np.ndarray:
        """
        FIGURE SEGMENTATION WITH WHITE BACKGROUND:
        """
        quantized_white_bg, labels, bg_cluster_idx = TextileCADEngine.generate_quantized_motifs_white_bg(cv_img, num_colors=8)
        num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats((labels != bg_cluster_idx).astype(np.uint8))

        label_hue = np.uint8(179 * comp_labels / max(1, num_labels))
        blank_ch = 255 * np.ones_like(label_hue)
        segmented_hsv = cv2.merge([label_hue, blank_ch, blank_ch])
        segmented_bgr = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2BGR)
        segmented_bgr[comp_labels == 0] = [255, 255, 255]

        return segmented_bgr

    @staticmethod
    def analyze_design(cv_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes design for motif detection, repeat pattern type, and complexity.
        """
        h, w = cv_img.shape[:2]
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img

        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.count_nonzero(edges) / float(h * w)
        motif_detected = bool(edge_density > 0.003)

        half_h, half_w = h // 2, w // 2
        if half_h >= 32 and half_w >= 32:
            top = gray[0:half_h, :]
            bot = gray[half_h:2*half_h, :]

            diff_grid = np.mean(np.abs(top.astype(float) - bot.astype(float)))

            bot_shifted = np.roll(bot, shift=half_w // 2, axis=1)
            diff_half_drop = np.mean(np.abs(top.astype(float) - bot_shifted.astype(float)))

            if diff_grid < 30.0 and diff_grid <= diff_half_drop:
                repeat_pattern = "grid"
            elif diff_half_drop < 35.0:
                repeat_pattern = "half-drop"
            else:
                repeat_pattern = "none"
        else:
            repeat_pattern = "none"

        if edge_density < 0.02:
            complexity = "low"
        elif edge_density < 0.07:
            complexity = "medium"
        else:
            complexity = "high"

        return {
            "motif_detected": motif_detected,
            "repeat_pattern": repeat_pattern,
            "complexity": complexity
        }

    def process_image(self, input_image: Union[str, np.ndarray, Image.Image], output_dir: str = "./output") -> Dict[str, Any]:
        """
        Main processing method for Textile CAD AI Engine.
        """
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Input image file not found: {input_image}")
            cv_img = cv2.imread(input_image)
            if cv_img is None:
                raise ValueError(f"Failed to read image from path: {input_image}")
        elif isinstance(input_image, Image.Image):
            cv_img = pil_to_cv2(input_image.convert("RGB"))
        elif isinstance(input_image, np.ndarray):
            cv_img = input_image
        else:
            raise TypeError("Unsupported image input type")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Analyze design
        analysis = self.analyze_design(cv_img)

        # 2. Master Clean Image
        master_clean = self.generate_master_clean(cv_img)
        master_path = str(out_path / "master_clean.bmp")
        cv2.imwrite(master_path, master_clean)

        # 3. Pencil & Fine Line Art Sketch (Hand-drawn Shading Effect matching reference image)
        pencil_art = self.generate_pencil_sketch_art(cv_img, paper_background=True)
        pencil_path = str(out_path / "pencil_sketch_art.png")
        cv2.imwrite(pencil_path, pencil_art)

        # 4. Pure Binary Texcelle CAD Line Sketch
        bw_texcelle = self.generate_figure_bw_texcelle(cv_img)
        bw_path = str(out_path / "bw_texcelle.bmp")
        cv2.imwrite(bw_path, bw_texcelle)

        # 5. Pixel-to-Pixel Quantized CAD & Grid View
        quantized_cad, grid_view = self.generate_pixel_quantized_cad(master_clean, num_colors=8)
        quantized_path = str(out_path / "quantized_cad.bmp")
        grid_path = str(out_path / "pixel_grid_view.bmp")
        cv2.imwrite(quantized_path, quantized_cad)
        cv2.imwrite(grid_path, grid_view)

        # 6. Color Variants (4 distinct palettes with White Background)
        color_variants = self.generate_4_colorway_variants(master_clean)
        variant_paths = {}
        for key, var_img in color_variants.items():
            v_path = str(out_path / f"color_variant_{key}.bmp")
            cv2.imwrite(v_path, var_img)
            variant_paths[key] = os.path.abspath(v_path)

        # 7. Figure Segmentation
        seg_img = self.segment_figures(cv_img)
        seg_path = str(out_path / "figure_segmentation.bmp")
        cv2.imwrite(seg_path, seg_img)

        # Return structured JSON format
        response = {
            "analysis": analysis,
            "outputs": {
                "master_clean": os.path.abspath(master_path),
                "pencil_sketch_art": os.path.abspath(pencil_path),
                "bw_texcelle": os.path.abspath(bw_path),
                "quantized_cad": os.path.abspath(quantized_path),
                "pixel_grid_view": os.path.abspath(grid_path),
                "color_variants": {
                    "soft": variant_paths["soft"],
                    "contrast": variant_paths["contrast"],
                    "vibrant": variant_paths["vibrant"],
                    "mono": variant_paths["mono"]
                }
            }
        }

        return response

cad_engine_processor = TextileCADEngine()
