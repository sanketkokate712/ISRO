import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load TMC again just in case
tmc_img = cv2.imread("moon_surface.jpg", 0)

# --- Set up blob detector parameters ---
try:
    params = cv2.SimpleBlobDetector_Params()
except AttributeError:
    # For older OpenCV versions
    params = cv2.SimpleBlobDetector_Params

params.filterByArea = True
params.minArea = 20
params.maxArea = 500

params.filterByCircularity = True
params.minCircularity = 0.6

params.filterByConvexity = False
params.filterByInertia = False

# Create detector
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(tmc_img)

# --- Draw keypoints ---
im_with_keypoints = cv2.drawKeypoints(
    tmc_img, keypoints, np.array([]),
    (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# --- Plot results ---
plt.figure(figsize=(10, 6))
plt.imshow(im_with_keypoints)
plt.title(f"Detected Boulders: {len(keypoints)} found")
plt.axis("off")
plt.show()

# --- Store stats from DTM ---
import rasterio
with rasterio.open("us_nga_egm96_15.tif") as dtm_src:
    dtm = dtm_src.read(1)

boulder_data = []
for k in keypoints:
    x, y = int(k.pt[0]), int(k.pt[1])
    size = round(k.size, 2)
    elev = round(dtm[y, x], 2) if 0 <= y < dtm.shape[0] and 0 <= x < dtm.shape[1] else -1
    boulder_data.append((x, y, size, elev))

# --- Save to CSV ---
import pandas as pd
df = pd.DataFrame(boulder_data, columns=["X", "Y", "Diameter_Px", "Elevation"])
df.to_csv("boulder_detection_output.csv", index=False)

print("✅ Boulder detection done and results saved to CSV!")
df.head()