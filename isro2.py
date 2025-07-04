import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import os

# --- Load TMC grayscale and convert to color ---
tmc_img = cv2.imread("moon_surface.jpg", cv2.IMREAD_GRAYSCALE)
tmc_img_color = cv2.cvtColor(tmc_img, cv2.COLOR_GRAY2RGB)

# --- Load DTM and compute slope ---
with rasterio.open("us_nga_egm96_15.tif") as dtm_src:
    dtm = dtm_src.read(1)

dy, dx = np.gradient(dtm)
slope = np.sqrt(dx**2 + dy**2)
slope_thresh = slope > 5

# --- Edge Detection on TMC image ---
edges = cv2.Laplacian(tmc_img, cv2.CV_64F)
edges_abs = np.abs(edges)
edges_norm = ((edges_abs - edges_abs.min()) / (edges_abs.max() - edges_abs.min()) * 255).astype(np.uint8)
edges_resized = cv2.resize(edges_norm, (dtm.shape[1], dtm.shape[0]), interpolation=cv2.INTER_LINEAR)
edges_thresh = edges_resized > 15

# --- Landslide detection mask (from slope + texture) ---
landslide_mask = slope_thresh & edges_thresh

# --- Resize landslide mask to match TMC image ---
resized_mask = cv2.resize(landslide_mask.astype(np.uint8), (tmc_img.shape[1], tmc_img.shape[0]), interpolation=cv2.INTER_NEAREST)
resized_mask = resized_mask.astype(bool)

# --- Overlay red zones for landslides ---
overlay = tmc_img_color.copy()
overlay[resized_mask] = [255, 0, 0]  # Red overlay for landslides

# --- Load or Generate Boulder CSV ---
csv_path = "boulder_detection_output.csv"
if not os.path.exists(csv_path):
    print("⚠️ CSV not found! Generating boulder detections...")

    # Detect blobs in TMC image
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(tmc_img)

    # Extract boulder info
    boulder_data = []
    for k in keypoints:
        x, y = int(k.pt[0]), int(k.pt[1])
        size = round(k.size, 2)
        elev = round(dtm[y, x], 2) if 0 <= y < dtm.shape[0] and 0 <= x < dtm.shape[1] else -1
        boulder_data.append((x, y, size, elev))

    df = pd.DataFrame(boulder_data, columns=["X", "Y", "Diameter_Px", "Elevation"])
    df.to_csv(csv_path, index=False)
    print("✅ CSV generated and saved.")
else:
    df = pd.read_csv(csv_path)
    print("✅ Boulder CSV loaded successfully.")

# --- Overlay green circles for detected boulders ---
for _, row in df.iterrows():
    x, y = int(row["X"]), int(row["Y"])
    radius = max(1, int(row["Diameter_Px"] / 2))
    if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
        cv2.circle(overlay, (x, y), radius, (0, 255, 0), 1)

# --- Final Output Plot ---
plt.figure(figsize=(12, 6))
plt.imshow(overlay)
plt.title("🌕 Boulders (Green) + Landslides (Red) Overlay Map")
plt.axis("off")
plt.tight_layout()
plt.show()
