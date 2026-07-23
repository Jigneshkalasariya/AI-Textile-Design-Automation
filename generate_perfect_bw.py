import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Load original image
img_path = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
cv_img = cv2.imread(img_path)
h, w, _ = cv_img.shape

# 1. Bilateral filter for noise reduction
denoised = cv2.bilateralFilter(cv_img, 7, 40, 40)
gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

# --- APPROACH A: High-Precision Texcelle Vector Line Sketch (Crisp CAD Tracing) ---
# Multi-scale adaptive thresholding + Canny edge fusion + Bilateral line sharpening
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)
canny_edges = cv2.Canny(gray, 35, 110)
canny_inv = 255 - canny_edges

line_sketch = cv2.bitwise_and(adaptive_thresh, canny_inv)
line_sketch = cv2.medianBlur(line_sketch, 3)

# --- APPROACH B: Quantized Region Boundary CAD Sketch ---
# Segment image into 8 indexed colors in LAB space, then extract boundary contours between ALL color regions
lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
pixels_lab = lab.reshape(-1, 3).astype(np.float32)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
labels = kmeans.fit_predict(pixels_lab).reshape(h, w)

# Find edges between different color regions
kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
labels_float = labels.astype(np.float32)
grad_x = cv2.Sobel(labels_float, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(labels_float, cv2.CV_32F, 0, 1, ksize=3)
color_boundaries = (np.sqrt(grad_x**2 + grad_y**2) > 0.1).astype(np.uint8) * 255

color_boundaries_inv = 255 - color_boundaries
color_boundaries_inv = cv2.medianBlur(color_boundaries_inv, 3)

# --- APPROACH C: Master Industrial Texcelle Figure Sketch (Combines Filled Motif Masks + Inner Features) ---
# Background-Distance Motif Mask
bg_cluster_idx = np.argmax(np.bincount(labels.flatten()))
bg_color_lab = kmeans.cluster_centers_[bg_cluster_idx]
diff = pixels_lab - bg_color_lab
dist = np.sqrt(np.sum(diff ** 2, axis=1)).reshape(h, w)

bg_mask = (dist <= 18.0) # True for background

# Create filled figure sketch where motifs are outlined and detailed inside, and background is pure white!
master_bw = line_sketch.copy()
master_bw[bg_mask] = 255 # Force background to pure white

# Save all test candidates for artifact display
brain_dir = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81"

cv2.imwrite("./output/bw_approach_a_line_sketch.png", line_sketch)
cv2.imwrite("./output/bw_approach_b_region_boundaries.png", color_boundaries_inv)
cv2.imwrite("./output/bw_approach_c_master_texcelle.png", master_bw)

cv2.imwrite(f"{brain_dir}\\bw_approach_a_line_sketch.png", line_sketch)
cv2.imwrite(f"{brain_dir}\\bw_approach_b_region_boundaries.png", color_boundaries_inv)
cv2.imwrite(f"{brain_dir}\\bw_approach_c_master_texcelle.png", master_bw)

print("Saved Approaches A, B, and C to output and brain directory!")
