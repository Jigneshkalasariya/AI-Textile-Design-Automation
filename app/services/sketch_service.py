import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.logger import logger

# Try importing torch for optional AI enhancement
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class LightweightEdgeRefiner(nn.Module):
        """
        Lightweight PyTorch CNN model to perform line/edge refinement
        and background noise suppression.
        """
        def __init__(self):
            super().__init__()
            # 3-layer CNN
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()
            
            self._initialize_weights()
            
        def _initialize_weights(self):
            with torch.no_grad():
                # Setup identity mapping for channel 0 and high-pass filters for details
                nn.init.dirac_(self.conv1.weight)
                nn.init.zeros_(self.conv1.bias)
                
                nn.init.dirac_(self.conv2.weight)
                nn.init.zeros_(self.conv2.bias)
                
                # Output layer creates a sharpening filter
                nn.init.zeros_(self.conv3.weight)
                self.conv3.weight[0, 0, 1, 1] = 5.0
                self.conv3.weight[0, 0, 0, 1] = -1.0
                self.conv3.weight[0, 0, 2, 1] = -1.0
                self.conv3.weight[0, 0, 1, 0] = -1.0
                self.conv3.weight[0, 0, 1, 2] = -1.0
                nn.init.constant_(self.conv3.bias, 0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.sigmoid(self.conv3(x))
            return x

    _ai_refiner_model: Optional[LightweightEdgeRefiner] = None
else:
    _ai_refiner_model = None


def preprocess(image: np.ndarray, max_dim: int = 1024, blur_kernel: int = 5) -> np.ndarray:
    """
    Step 1: Preprocess the input image.
    - Resize: keeps aspect ratio, limits maximum dimension to max_dim.
    - Normalize: stretches contrast.
    - Denoise: applies Gaussian blur.
    """
    logger.debug("Running Step 1: Preprocess (resize, normalize, denoise)")
    
    # 1. Resize if needed
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized image from {(w, h)} to {(new_w, new_h)}")
        
    # 2. Normalize
    # Contrast normalization: stretch to [0, 255]
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 3. Denoise (Gaussian blur)
    # Ensure kernel size is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    denoised = cv2.GaussianBlur(normalized, (blur_kernel, blur_kernel), 0)
    
    return denoised


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Step 2: Convert image to grayscale.
    """
    logger.debug("Running Step 2: Convert to grayscale")
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_edges(image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Step 3: Canny edge detection.
    Expects single-channel grayscale image.
    """
    logger.debug("Running Step 3: Detect edges (Canny)")
    return cv2.Canny(image, low_thresh, high_thresh)


def generate_sketch(image: np.ndarray, method: str = "pencil", **kwargs) -> np.ndarray:
    """
    Step 4: Generate sketch.
    - pencil: invert + divide method
    - threshold: adaptive threshold
    """
    logger.debug(f"Running Step 4: Generate sketch using method='{method}'")
    if method == "pencil":
        inverted = 255 - image
        # Gaussian blur with large kernel for sketch soft shadows
        blur_size = kwargs.get("pencil_blur_size", 21)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)
        # Division blend
        sketch = cv2.divide(image, 255 - blurred, scale=256)
        return sketch
    elif method == "threshold":
        block_size = kwargs.get("threshold_block_size", 11)
        if block_size % 2 == 0:
            block_size += 1
        c_val = kwargs.get("threshold_c", 2)
        sketch = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_val
        )
        return sketch
    else:
        raise ValueError(f"Unknown sketch method: {method}")


def enhance_lines(image: np.ndarray, operation: str = "none", kernel_size: int = 3) -> np.ndarray:
    """
    Step 5: Enhance lines using morphological operations.
    Since sketch is dark lines (0) on light background (255):
    - 'erode' will expand the dark regions (thicken lines)
    - 'dilate' will expand the bright regions (thin lines / clean noise)
    - 'none' leaves the image unchanged
    """
    logger.debug(f"Running Step 5: Enhance lines (morphology={operation}, kernel_size={kernel_size})")
    if operation == "none":
        return image
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == "erode":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "dilate":
        return cv2.dilate(image, kernel, iterations=1)
    else:
        raise ValueError(f"Unknown morphology operation: {operation}")


def optional_ai_enhancement(image: np.ndarray, enabled: bool = False) -> np.ndarray:
    """
    Step 6: Optional AI enhancement.
    Uses a lightweight PyTorch CNN edge refiner if enabled and PyTorch is available.
    """
    if not enabled:
        logger.debug("Step 6: AI enhancement disabled (skipped)")
        return image
        
    if not HAS_TORCH:
        logger.warning("Step 6: AI enhancement enabled, but torch is not installed. Skipping step.")
        return image
        
    logger.debug("Running Step 6: AI enhancement (lightweight PyTorch CNN)")
    try:
        global _ai_refiner_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _ai_refiner_model is None:
            _ai_refiner_model = LightweightEdgeRefiner().to(device)
            _ai_refiner_model.eval()
            
        # Invert sketch: model is trained / initialized on white lines on black background
        inverted = 255.0 - image.astype(np.float32)
        tensor = torch.from_numpy(inverted).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            output = _ai_refiner_model(tensor)
            
        # Convert back, rescale, invert back to black lines on white
        output_np = output.squeeze().cpu().numpy()
        enhanced = (255.0 - (output_np * 255.0)).clip(0, 255).astype(np.uint8)
        return enhanced
    except Exception as e:
        logger.error(f"Step 6: AI enhancement failed with error: {e}. Skipping.")
        return image


def add_grid(
    image: np.ndarray,
    enabled: bool = True,
    spacing: int = 50,
    color: int = 220,
    opacity: float = 0.3
) -> np.ndarray:
    """
    Step 7: Draw coordinate grid overlay and blend.
    """
    if not enabled:
        logger.debug("Step 7: Grid disabled (skipped)")
        return image
        
    logger.debug(f"Running Step 7: Add grid (spacing={spacing}, color={color}, opacity={opacity})")
    
    # Draw grid on a canvas of same shape
    grid_overlay = image.copy()
    h, w = image.shape[:2]
    
    # Draw vertical lines
    for x in range(spacing, w, spacing):
        cv2.line(grid_overlay, (x, 0), (x, h), color, 1)
        
    # Draw horizontal lines
    for y in range(spacing, h, spacing):
        cv2.line(grid_overlay, (0, y), (w, y), color, 1)
        
    # Blend grid overlay with original sketch image
    blended = cv2.addWeighted(image, 1.0 - opacity, grid_overlay, opacity, 0.0)
    return blended


def export_image(image: np.ndarray, output_path: str, format: str) -> str:
    """
    Step 8: Export image in PNG, JPG, or BMP format.
    BMP is written uncompressed/lossless.
    """
    logger.debug(f"Running Step 8: Export image as {format} to {output_path}")
    
    # Resolve file extension based on format
    fmt = format.lower().strip()
    if fmt not in ["png", "jpg", "jpeg", "bmp"]:
        raise ValueError(f"Unsupported export format: {format}")
        
    # Check output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save using cv2.imwrite
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to write image to {output_path}")
        
    logger.info(f"Successfully exported sketch image: {output_path}")
    return output_path


def process_image(
    input_path: str,
    output_dir: str,
    format: str = "bmp",
    use_ai: bool = False,
    grid: bool = True,
    **kwargs
) -> str:
    """
    Process image pipeline: converts input textile image to black & white sketch.
    
    Args:
        input_path: Path to the input JPG/PNG image.
        output_dir: Directory where the sketch will be saved.
        format: Export format - 'bmp' (default), 'png', or 'jpg'.
        use_ai: Enable optional lightweight AI line enhancement.
        grid: Add grid overlay.
        **kwargs: Additional configurations for pipeline steps:
            - max_dim: Max dimension for resize (default: 1024)
            - blur_kernel: Blur kernel size for denoising (default: 5)
            - low_canny: Canny low threshold (default: 50)
            - high_canny: Canny high threshold (default: 150)
            - sketch_method: 'pencil' (default) or 'threshold'
            - pencil_blur_size: Blur kernel for pencil sketch (default: 21)
            - threshold_block_size: Block size for adaptive threshold (default: 11)
            - threshold_c: Constant C for adaptive threshold (default: 2)
            - morphology: Line enhancement 'erode', 'dilate', or 'none' (default: 'none')
            - morphology_kernel: Kernel size for morphology (default: 3)
            - grid_spacing: Grid line interval (default: 50)
            - grid_color: Gray intensity of grid lines (default: 220)
            - grid_opacity: Opacity for grid blending (default: 0.3)
            - fuse_edges: Blend Canny edges into sketch to sharpen boundaries (default: True)
            
    Returns:
        The absolute path to the saved sketch image.
    """
    logger.info(f"Starting sketch-processing pipeline for input: {input_path}")
    import time
    start_time = time.time()
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
        
    # Read the input image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to read image from path: {input_path}")
        
    # Step 1: Preprocess
    max_dim = kwargs.get("max_dim", 1024)
    blur_kernel = kwargs.get("blur_kernel", 5)
    preprocessed = preprocess(image, max_dim=max_dim, blur_kernel=blur_kernel)
    
    # Step 2: Convert to grayscale
    gray = to_grayscale(preprocessed)
    
    # Step 3: Detect edges (Canny)
    low_canny = kwargs.get("low_canny", 50)
    high_canny = kwargs.get("high_canny", 150)
    edges = detect_edges(gray, low_thresh=low_canny, high_thresh=high_canny)
    
    # Step 4: Generate sketch
    sketch_method = kwargs.get("sketch_method", "pencil")
    sketch = generate_sketch(gray, method=sketch_method, **kwargs)
    
    # Optional boundary enhancement using Canny edges
    fuse_edges = kwargs.get("fuse_edges", True)
    if fuse_edges:
        logger.debug("Fusing Canny edges to sharpen boundaries in sketch")
        # Invert Canny edges: white lines on black background -> black lines on white background
        inverted_edges = 255 - edges
        # Combine using bitwise AND to overlay the black lines onto the sketch
        sketch = cv2.bitwise_and(sketch, inverted_edges)
        
    # Step 5: Enhance lines (morphology)
    morphology = kwargs.get("morphology", "none")
    morph_kernel = kwargs.get("morphology_kernel", 3)
    enhanced = enhance_lines(sketch, operation=morphology, kernel_size=morph_kernel)
    
    # Step 6: Optional AI enhancement
    ai_enhanced = optional_ai_enhancement(enhanced, enabled=use_ai)
    
    # Step 7: Add grid
    grid_spacing = kwargs.get("grid_spacing", 50)
    grid_color = kwargs.get("grid_color", 220)
    grid_opacity = kwargs.get("grid_opacity", 0.3)
    final_image = add_grid(
        ai_enhanced,
        enabled=grid,
        spacing=grid_spacing,
        color=grid_color,
        opacity=grid_opacity
    )
    
    # Step 8: Export image
    # Construct output filename
    input_filename = Path(input_path).stem
    out_format = format.lower().strip()
    out_filename = f"{input_filename}_sketch.{out_format}"
    output_path = os.path.join(output_dir, out_filename)
    
    saved_path = export_image(final_image, output_path, out_format)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Sketch processing pipeline completed in {elapsed_time:.3f} seconds. Output: {saved_path}")
    
    return os.path.abspath(saved_path)
