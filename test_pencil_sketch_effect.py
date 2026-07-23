import cv2
import numpy as np
from PIL import Image

# Load original Indian elephant textile artwork
img_path = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
cv_img = cv2.imread(img_path)
h, w, c = cv_img.shape

# 1. Convert to Grayscale & Bilateral Denoising
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
denoised_gray = cv2.bilateralFilter(gray, 7, 35, 35)

# --- EFFECT 1: Division Pencil Sketch with Soft Gradient Shading ---
inv_gray = 255 - denoised_gray
blur_pencil = cv2.GaussianBlur(inv_gray, (21, 21), 0)
sketch_divide = cv2.divide(denoised_gray, 255 - blur_pencil, scale=256)

# --- EFFECT 2: XDoG (Extended Difference of Gaussians) Fine Line & Hatching Art ---
g1 = cv2.GaussianBlur(denoised_gray, (3, 3), 0.8)
g2 = cv2.GaussianBlur(denoised_gray, (7, 7), 1.6)
xdog = g1.astype(float) - 0.95 * g2.astype(float)
xdog = np.clip(xdog, 0, 255).astype(np.uint8)
# Threshold and contrast adjustment
_, xdog_thresh = cv2.threshold(xdog, 240, 255, cv2.THRESH_BINARY)
xdog_sketch = cv2.bitwise_and(sketch_divide, xdog_thresh)

# --- EFFECT 3: Isolating Motifs on Clean Paper/White Background ---
# Use LAB color space background mask to clean deep magenta background to pure white/cream (#F9F8F3 or #FFFFFF)
lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
pixels_lab = lab.reshape(-1, 3).astype(np.float32)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
labels = kmeans.fit_predict(pixels_lab).reshape(h, w)
counts = np.bincount(labels.flatten())
bg_cluster_idx = np.argmax(counts)
bg_color_lab = kmeans.cluster_centers_[bg_cluster_idx]

diff = pixels_lab - bg_color_lab
dist = np.sqrt(np.sum(diff ** 2, axis=1)).reshape(h, w)
bg_mask = (dist <= 18.0) # True for background ground

# Create Master Pencil & Fine Line Art on Paper Background
master_pencil = sketch_divide.copy()
# Force background to pure white / paper tone
master_pencil[bg_mask] = 255

# Enhance contrast of pencil lines inside motifs
master_pencil_bgr = cv2.cvtColor(master_pencil, cv2.COLOR_GRAY2BGR)

# Apply subtle warm paper tint (#F9F7F1) like the reference image
paper_tint = np.zeros_like(cv_img)
paper_tint[:, :] = [241, 247, 249] # BGR for warm paper cream
blended_paper_pencil = cv2.addWeighted(master_pencil_bgr, 0.92, paper_tint, 0.08, 0)

# Save test outputs
brain_dir = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81"

cv2.imwrite("./output/pencil_sketch_division.png", sketch_divide)
cv2.imwrite("./output/pencil_sketch_master_white.png", master_pencil)
cv2.imwrite("./output/pencil_sketch_paper_tint.png", blended_paper_pencil)

cv2.imwrite(f"{brain_dir}\\pencil_sketch_division.png", sketch_divide)
cv2.imwrite(f"{brain_dir}\\pencil_sketch_master_white.png", master_pencil)
cv2.imwrite(f"{brain_dir}\\pencil_sketch_paper_tint.png", blended_paper_pencil)

print("Saved pencil sketch test outputs to output and brain directory!")
