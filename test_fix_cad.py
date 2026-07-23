import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Load user's original image
img_path = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
cv_img = cv2.imread(img_path)
h, w, c = cv_img.shape

print(f"Original image loaded: {w}x{h}")

# 1. Background Color Detection in LAB Space
lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
pixels_lab = lab.reshape(-1, 3).astype(np.float32)

# Sample grid to find most dominant background color
# Histogram / dominant color via KMeans (k=8)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
labels = kmeans.fit_predict(pixels_lab)

# Find largest cluster (dominant background)
counts = np.bincount(labels)
bg_cluster_idx = np.argmax(counts)
bg_color_lab = kmeans.cluster_centers_[bg_cluster_idx]

print(f"Background cluster {bg_cluster_idx} covers {counts[bg_cluster_idx]/(h*w)*100:.2f}% of pixels")

# Calculate Delta-E (Euclidean distance in LAB space) from background color for all pixels
diff = pixels_lab - bg_color_lab
dist = np.sqrt(np.sum(diff ** 2, axis=1)).reshape(h, w)

# Threshold distance to separate background from foreground motifs
# Distance threshold > 18.0 means distinct motif color
fg_mask = (dist > 18.0).astype(np.uint8) * 255

# Clean up noise & bridge small gaps
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
fg_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
fg_cleaned = cv2.medianBlur(fg_cleaned, 3)

# Invert for Texcelle standard (Background = 255 White, Motifs = 0 Black)
bw_texcelle = np.where(fg_cleaned > 127, 0, 255).astype(np.uint8)

cv2.imwrite("./output/test_bw_fixed.png", bw_texcelle)
print("Saved fixed BW Texcelle output to ./output/test_bw_fixed.png")

# 2. Pixel-to-Pixel Quantized Grid View (Clean 8-color CAD View)
centers_bgr = cv2.cvtColor(kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
quantized_bgr = centers_bgr[labels].reshape(h, w, 3)
cv2.imwrite("./output/test_quantized_cad.png", quantized_bgr)
print("Saved quantized CAD view to ./output/test_quantized_cad.png")

# 3. Pixel-to-Pixel Grid Overlay View
grid_img = quantized_bgr.copy()
# Draw subtle pixel grid lines every 10 pixels for Texcelle CAD view
grid_spacing = 16
for x in range(0, w, grid_spacing):
    cv2.line(grid_img, (x, 0), (x, h), (180, 180, 180), 1)
for y in range(0, h, grid_spacing):
    cv2.line(grid_img, (0, y), (w, y), (180, 180, 180), 1)

cv2.imwrite("./output/test_pixel_grid_view.png", grid_img)
print("Saved pixel grid view to ./output/test_pixel_grid_view.png")
