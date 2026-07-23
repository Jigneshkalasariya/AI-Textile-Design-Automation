import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Load original image
img_path = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
cv_img = cv2.imread(img_path)
h, w, c = cv_img.shape

# 1. High Quality Bilateral Denoising
denoised = cv2.bilateralFilter(cv_img, 9, 50, 50)

# 2. Color Quantization into 8 Clean Flat Colors in LAB Space
lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
pixels_lab = lab.reshape(-1, 3).astype(np.float32)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
labels = kmeans.fit_predict(pixels_lab).reshape(h, w)

centers_bgr = cv2.cvtColor(
    kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3),
    cv2.COLOR_LAB2BGR
).reshape(-1, 3)

quantized_bgr = centers_bgr[labels]

# 3. Identify Dominant Background Cluster Index (Magenta Ground)
counts = np.bincount(labels.flatten())
bg_cluster_idx = np.argmax(counts)

# --- ARTWORK 1: Master Clean Texcelle Artwork with White Background ---
# Replace background cluster with Pure White (255, 255, 255) while preserving all motif colors and details!
master_white_bg = quantized_bgr.copy()
master_white_bg[labels == bg_cluster_idx] = [255, 255, 255]

# --- ARTWORK 2: High-Precision Industrial Texcelle CAD Line Sketch ---
# Extract region boundaries between ALL quantized color clusters
labels_float = labels.astype(np.float32)
grad_x = cv2.Sobel(labels_float, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(labels_float, cv2.CV_32F, 0, 1, ksize=3)
boundaries = (np.sqrt(grad_x**2 + grad_y**2) > 0.05).astype(np.uint8) * 255

# Clean boundaries
kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
boundaries_clean = cv2.morphologyEx(boundaries, cv2.MORPH_OPEN, kernel_clean)
boundaries_clean = cv2.medianBlur(boundaries_clean, 3)

# Invert so boundaries are crisp black lines (0) on pure white background (255)
bw_sketch_lines = np.where(boundaries_clean > 127, 0, 255).astype(np.uint8)

# --- ARTWORK 3: Pixel Grid View ---
grid_view = master_white_bg.copy()
grid_spacing = 16
for x in range(0, w, grid_spacing):
    cv2.line(grid_view, (x, 0), (x, h), (210, 210, 210), 1)
for y in range(0, h, grid_spacing):
    cv2.line(grid_view, (0, y), (w, y), (210, 210, 210), 1)

# Save test outputs
brain_dir = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81"

cv2.imwrite("./output/professional_master_white_bg.png", master_white_bg)
cv2.imwrite("./output/professional_bw_cad_sketch.png", bw_sketch_lines)
cv2.imwrite("./output/professional_pixel_grid_view.png", grid_view)

cv2.imwrite(f"{brain_dir}\\professional_master_white_bg.png", master_white_bg)
cv2.imwrite(f"{brain_dir}\\professional_bw_cad_sketch.png", bw_sketch_lines)
cv2.imwrite(f"{brain_dir}\\professional_pixel_grid_view.png", grid_view)

print("Saved professional master white bg, bw cad sketch, and pixel grid view!")
