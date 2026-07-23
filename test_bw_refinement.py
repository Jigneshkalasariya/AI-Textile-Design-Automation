import cv2
import numpy as np
from PIL import Image

# Load original image
img_path = r"C:\Users\Jignesh K kalasariya\.gemini\antigravity-ide\brain\4235fcdd-dcd0-40ca-a31b-20fd475a7e81\media__1784617282507.jpg"
cv_img = cv2.imread(img_path)
h, w, _ = cv_img.shape

# METHOD 1: Multi-scale Edge & Feature Preserving Texcelle Line Sketch (Crisp CAD Tracing View)
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
denoised = cv2.bilateralFilter(gray, 5, 30, 30)

# Adaptive threshold for local detail (captures flower petals, elephant features, saddle details, border dots)
adaptive_bw = cv2.adaptiveThreshold(
    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 15, 4
)

# Canny edge detection for crisp outline reinforcement
edges = cv2.Canny(denoised, 30, 100)
edges_inv = cv2.bitwise_not(edges)

# Combine adaptive details with Canny edges
combined_lines = cv2.bitwise_and(adaptive_bw, edges_inv)

# Morphological clean up to remove isolated 1-2px noise dots
kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cleaned_lines = cv2.medianBlur(combined_lines, 3)

# Strict binary thresholding
_, final_line_sketch = cv2.threshold(cleaned_lines, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("./output/test_bw_method1_lines.png", final_line_sketch)

# METHOD 2: Quantized Figure Segmentation Contour Sketch (Clean Industrial Component Borders)
# Perform LAB color quantization to 8 clean channels, then extract boundary lines between ALL color regions!
lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
pixels_lab = lab.reshape(-1, 3).astype(np.float32)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42, n_init=5)
labels = kmeans.fit_predict(pixels_lab).reshape(h, w)

# Find boundaries between different quantized color regions
# Gradient of labels / laplacian
from scipy.ndimage import laplace
# Find pixels where neighbor has a different label
label_pad = np.pad(labels, 1, mode='edge')
diff_h = (labels != label_pad[1:-1, 2:]) | (labels != label_pad[1:-1, :-2])
diff_v = (labels != label_pad[2:, 1:-1]) | (labels != label_pad[:-2, 1:-1])
boundaries = (diff_h | diff_v).astype(np.uint8) * 255

# Invert boundaries: Black lines (0) on White background (255)
boundaries_inv = 255 - boundaries
# Dilate lines slightly to make crisp CAD lines
kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
thick_boundaries = cv2.erode(boundaries_inv, kernel_line, iterations=1)

cv2.imwrite("./output/test_bw_method2_boundaries.png", thick_boundaries)

# METHOD 3: Hybrid Figure-Preserving Texcelle Sketch (Combines Motif Boundaries + Structural Interior Lines)
# Clean background + detailed interior features
hybrid = cv2.bitwise_and(final_line_sketch, thick_boundaries)
cv2.imwrite("./output/test_bw_method3_hybrid.png", hybrid)

print("Saved Method 1, Method 2, and Method 3 test images to ./output/")
