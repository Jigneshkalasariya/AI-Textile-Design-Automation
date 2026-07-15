import cv2
import numpy as np
from PIL import Image

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert an OpenCV image (numpy array) to a PIL Image."""
    if len(cv_img.shape) == 2:  # Grayscale
        return Image.fromarray(cv_img)
    elif cv_img.shape[2] == 4:  # BGRA
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgb)
    else:  # BGR
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV image (numpy array)."""
    img_arr = np.array(pil_img)
    if len(img_arr.shape) == 2:  # Grayscale
        return img_arr
    elif img_arr.shape[2] == 4:  # RGBA
        return cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGRA)
    else:  # RGB
        return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

def create_synthetic_textile_image(width: int = 512, height: int = 512) -> Image.Image:
    """
    Generates a synthetic textile design image.
    Has a distinct background color (e.g., beige) and geometric/organic motif shapes (flowers/leaves)
    for background removal, object detection, and repeat boundary testing.
    """
    # 1. Beige Background
    img = np.full((height, width, 3), (220, 230, 245), dtype=np.uint8) # BGR beige/light pink
    
    # 2. Draw stem/vine (Green)
    cv2.ellipse(img, (width // 2, height // 2 + 50), (120, 180), 30, 0, 180, (50, 120, 30), 8)
    
    # 3. Draw leaves
    cv2.circle(img, (width // 2 - 80, height // 2 - 20), 30, (70, 150, 40), -1)
    cv2.circle(img, (width // 2 + 80, height // 2 + 20), 25, (70, 150, 40), -1)

    # 4. Draw central flower (Red & Yellow)
    center = (width // 2, height // 2 - 50)
    # Petals
    for angle in range(0, 360, 45):
        rad = np.deg2rad(angle)
        px = int(center[0] + 40 * np.cos(rad))
        py = int(center[1] + 40 * np.sin(rad))
        cv2.circle(img, (px, py), 25, (80, 80, 230), -1) # Red/pink
        cv2.circle(img, (px, py), 25, (50, 50, 180), 2) # darker edge
        
    # Flower Center
    cv2.circle(img, center, 30, (80, 220, 240), -1) # Yellow center
    cv2.circle(img, center, 30, (50, 180, 200), 2) # darker edge

    # 5. Add a simple texture pattern to background (small blue dots)
    for x in range(20, width, 80):
        for y in range(20, height, 80):
            if np.linalg.norm(np.array([x, y]) - np.array(center)) > 150:
                cv2.circle(img, (x, y), 3, (240, 120, 100), -1) # Pale blue dot

    # Return as PIL Image
    return cv2_to_pil(img)
