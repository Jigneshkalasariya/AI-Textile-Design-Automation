import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Load original image
img_path = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
cv_img = cv2.imread(img_path)
h, w, c = cv_img.shape

# 1. Bilateral filter for edge-preserving smoothing
denoised = cv2.bilateralFilter(cv_img, 7, 35, 35)

# 2. LAB color space background isolation
lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
pixels_lab = lab.reshape(-1, 3).astype(np.float32)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
labels = kmeans.fit_predict(pixels_lab).reshape(h, w)

counts = np.bincount(labels.flatten())
bg_cluster_idx = np.argmax(counts)
bg_color_lab = kmeans.cluster_centers_[bg_cluster_idx]

# Delta-E distance from background color
diff = pixels_lab - bg_color_lab
dist = np.sqrt(np.sum(diff ** 2, axis=1)).reshape(h, w)

# Threshold distance to detect background vs motif figures
# Background mask (where distance <= 18.0)
bg_mask = (dist <= 18.0)
fg_mask = (dist > 18.0).astype(np.uint8) * 255

# 3. Fill Figures (Solid Morphological Filling + Contour Filling)
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
fg_filled = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
fg_filled = cv2.medianBlur(fg_filled, 5)

# Find external contours of all motifs (elephants, flowers, leaves, saddles, borders)
contours, _ = cv2.findContours(fg_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
solid_figures = np.zeros_like(fg_filled)
min_area = (h * w) * 0.00015

for cnt in contours:
    if cv2.contourArea(cnt) > min_area:
        # Fill figure completely (thickness=-1)
        cv2.drawContours(solid_figures, [cnt], -1, 255, thickness=cv2.FILLED)

# Smooth figure edges
kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
solid_figures = cv2.morphologyEx(solid_figures, cv2.MORPH_CLOSE, kernel_smooth)

# --- OUTPUT 1: Pure Binary Texcelle Filled Figure Sketch ---
# Background/Border is Pure White (255), Figures are Filled Black (0)
bw_filled_figures = np.where(solid_figures > 127, 0, 255).astype(np.uint8)

# --- OUTPUT 2: Original Color Filled Figures on Pure White Background/Border ---
# Motifs retain their vibrant original colors, while background & border are turned pure white (255, 255, 255)!
color_filled_white_bg = cv_img.copy()
color_filled_white_bg[solid_figures == 0] = [255, 255, 255] # Background/Border = White

# Save images to brain directory for display
brain_dir = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81"

cv2.imwrite("./output/bw_solid_filled_figures.png", bw_filled_figures)
cv2.imwrite("./output/color_filled_figures_white_bg.png", color_filled_white_bg)

cv2.imwrite(f"{brain_dir}\\bw_solid_filled_figures.png", bw_filled_figures)
cv2.imwrite(f"{brain_dir}\\color_filled_figures_white_bg.png", color_filled_white_bg)

print("Saved solid filled figures and color filled figures with white background/border!")
