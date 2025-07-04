import cv2
import rasterio
import matplotlib.pyplot as plt

# Load TMC grayscale image
TMC_PATH = "moon_surface.jpg"
tmc_img = cv2.imread(TMC_PATH, cv2.IMREAD_GRAYSCALE)

# Load DTM (elevation) image
dtm_path = "us_nga_egm96_15.tif"
with rasterio.open(dtm_path) as dtm_src:
    dtm = dtm_src.read(1)

# Show both images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(tmc_img, cmap='gray')
plt.title("TMC Image (Simulated Boulders)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dtm, cmap='terrain')
plt.title("DTM Image (Simulated Elevation)")
plt.axis("off")

plt.tight_layout()
plt.show()
