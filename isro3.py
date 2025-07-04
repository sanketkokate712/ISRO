import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# --- Load DTM ---
with rasterio.open("us_nga_egm96_15.tif") as dtm_src:
    dtm = dtm_src.read(1)

# --- Compute slope from elevation ---
dy, dx = np.gradient(dtm)
slope = np.sqrt(dx**2 + dy**2)

# Normalize slope for visualization
slope_normalized = ((slope - slope.min()) / (slope.max() - slope.min()) * 255).astype(np.uint8)

# --- Load TMC grayscale image ---
tmc_img = cv2.imread("moon_surface.jpg", cv2.IMREAD_GRAYSCALE)

# Edge detection using Laplacian
edges = cv2.Laplacian(tmc_img, cv2.CV_64F)
edges_abs = np.abs(edges)
edges_norm = ((edges_abs - edges_abs.min()) / (edges_abs.max() - edges_abs.min()) * 255).astype(np.uint8)

# --- Resize edge image to match DTM shape ---
edges_resized = cv2.resize(edges_norm, (dtm.shape[1], dtm.shape[0]), interpolation=cv2.INTER_LINEAR)

# --- Thresholds ---
slope_thresh = slope > 5            # Slope threshold (lowered)
edges_thresh = edges_resized > 15   # Texture threshold (lowered)

# --- Combine masks to detect landslide zones ---
landslide_mask = slope_thresh & edges_thresh

# --- Debug Visualization ---
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(dtm, cmap='terrain')
plt.title("DTM - Raw Elevation")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(slope_normalized, cmap='inferno')
plt.title("Slope Map")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(slope_thresh, cmap='gray')
plt.title("Slope Threshold (>5)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(tmc_img, cmap='gray')
plt.title("TMC Image")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(edges_resized, cmap='gray')
plt.title("Texture (Laplacian Edges, Resized)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(landslide_mask, cmap='hot')
plt.title("🚨 Detected Landslide Zones")
plt.axis("off")

plt.tight_layout()
plt.show()
