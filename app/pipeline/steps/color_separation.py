import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from app.models.request_models import ColorSeparationConfig
from app.core.logger import logger
from app.utils.image_utils import pil_to_cv2

def separate_colors(img: Image.Image, config: ColorSeparationConfig) -> Tuple[List[Image.Image], List[Tuple[int, int, int]]]:
    """
    Separates the image into K discrete color channels/layers using K-Means clustering.
    Clustering is performed in LAB color space for better perceptual color grouping.
    Returns:
        - List of PIL binary mask images (one for each color layer)
        - List of color RGB tuples (representing each layer color)
    """
    logger.info("Running Step 6: Color Separation")
    cv_img = pil_to_cv2(img)
    
    # Strip alpha if present for clustering
    if cv_img.shape[2] == 4:
        # Keep alpha to mask out background later
        alpha = cv_img[:, :, 3]
        bgr = cv_img[:, :, :3]
    else:
        alpha = None
        bgr = cv_img

    # Convert to LAB color space
    if config.color_space.upper() == "LAB":
        feature_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    else:
        feature_img = bgr

    # Flatten image pixels
    pixels = feature_img.reshape((-1, 3)).astype(np.float32)
    
    # If alpha channel is present, only cluster non-transparent pixels
    if alpha is not None:
        flat_alpha = alpha.flatten()
        fg_indices = np.where(flat_alpha > 10)[0]
        if len(fg_indices) > 0:
            cluster_pixels = pixels[fg_indices]
        else:
            cluster_pixels = pixels
    else:
        cluster_pixels = pixels

    K = config.num_colors
    
    # If number of pixels is less than K, clamp K
    if len(cluster_pixels) < K:
        K = len(cluster_pixels)
        logger.warning(f"Clustering pixels count is less than target colors. Adjusting K to {K}")

    # Set K-Means termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    # Run KMeans
    compactness, labels, centers = cv2.kmeans(
        cluster_pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Map colors back
    # Convert centers back to BGR to get RGB representation
    centers = centers.astype(np.uint8)
    if config.color_space.upper() == "LAB":
        # Create a 1xK image of centroids in LAB to convert to BGR
        centers_img = centers.reshape((1, -1, 3))
        centers_bgr = cv2.cvtColor(centers_img, cv2.COLOR_LAB2BGR).reshape((-1, 3))
    else:
        centers_bgr = centers

    # Get RGB colors
    rgb_colors = [tuple(map(int, (c[2], c[1], c[0]))) for c in centers_bgr]
    
    # Generate label mask for full image
    full_labels = np.zeros(pixels.shape[0], dtype=np.int32)
    
    if alpha is not None and len(fg_indices) > 0:
        # For background pixels, set label to -1 (transparent)
        full_labels[:] = -1
        # Predict labels for foreground using minimum distance to centers
        # (This handles mapping labels for all pixels)
        diffs = pixels[fg_indices][:, np.newaxis, :] - centers
        dists = np.linalg.norm(diffs, axis=2)
        fg_labels = np.argmin(dists, axis=1)
        full_labels[fg_indices] = fg_labels
    else:
        diffs = pixels[:, np.newaxis, :] - centers
        dists = np.linalg.norm(diffs, axis=2)
        full_labels = np.argmin(dists, axis=1)

    # Reshape label back to 2D
    label_img = full_labels.reshape(bgr.shape[:2])
    
    # Build list of binary mask images
    layers = []
    for i in range(K):
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        mask[label_img == i] = 255
        layers.append(Image.fromarray(mask))

    logger.info(f"Color separation successfully isolated {K} color layers.")
    return layers, rgb_colors
