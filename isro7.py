import streamlit as st
import cv2
import numpy as np
import rasterio
import pandas as pd
import cv2
from PIL import Image

st.set_page_config(page_title="Lunar Boulder + Landslide Detector", layout="wide")
st.title("🌖 Lunar Boulder & Landslide Risk Detection App")

st.markdown("""
Upload a **grayscale lunar surface image** and a **DTM `.tif` file**.
This app will detect:
- 🟢 **Boulders**
- 🔴 **Landslide risk areas**
and generate useful plots and data tables.
""")

# File uploads
tmc_file = st.file_uploader("Upload Lunar Surface Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
dtm_file = st.file_uploader("Upload DTM Elevation File (.tif)", type=["tif", "tiff"])

# Slope threshold
slope_thresh = st.slider("⚠️ Slope Threshold for High Landslide Risk", 0.1, 2.0, 0.7, 0.05)

if tmc_file and dtm_file:
    # Load lunar image
    tmc_img = Image.open(tmc_file).convert("L")
    tmc_array = np.array(tmc_img)

    # Load DTM
    with rasterio.open(dtm_file) as dtm_src:
        dtm = dtm_src.read(1)
        # Detect boulders
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.6
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(tmc_array)

        # Compute slope
        dy, dx = np.gradient(dtm.astype(float))
        slope = np.sqrt(dx ** 2 + dy ** 2)

        # Normalize slope for visualization
        slope_norm = (slope - np.min(slope)) / (np.max(slope) - np.min(slope))

        # --- Visualization 1: Boulders in Green ---
        boulder_overlay = cv2.cvtColor(tmc_array, cv2.COLOR_GRAY2BGR)
        boulder_overlay = cv2.drawKeypoints(
            tmc_array, keypoints, np.array([]), (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        st.image(boulder_overlay, caption=f"🟢 Boulders Detected: {len(keypoints)}", channels="BGR",
                 use_column_width=True)

        # --- Visualization 2: Landslide Risk Overlay in RED ---
        landslide_risk = (slope > slope_thresh).astype(np.uint8) * 255

        # Resize risk map to match surface image if needed
        if landslide_risk.shape != tmc_array.shape:
            landslide_risk_resized = cv2.resize(landslide_risk, (tmc_array.shape[1], tmc_array.shape[0]))
        else:
            landslide_risk_resized = landslide_risk

        # Convert surface image to BGR
        tmc_bgr = cv2.cvtColor(tmc_array, cv2.COLOR_GRAY2BGR)

        # Create a red mask where landslide risk > threshold
        red_mask = np.zeros_like(tmc_bgr)
        red_mask[..., 2] = landslide_risk_resized  # Red channel

        # Blend: original + red where risk
        landslide_overlay = cv2.addWeighted(tmc_bgr, 1.0, red_mask, 0.5, 0)

        st.image(landslide_overlay, caption="🔴 Landslide Risk Areas", channels="BGR", use_column_width=True)

        # --- Data Table ---
        boulder_data = []
        for k in keypoints:
            x, y = int(k.pt[0]), int(k.pt[1])
            size = round(k.size, 2)
            elev = round(dtm[y, x], 2) if 0 <= y < dtm.shape[0] and 0 <= x < dtm.shape[1] else -1
            local_slope = round(slope[y, x], 3) if 0 <= y < slope.shape[0] and 0 <= x < slope.shape[1] else 0
            risk = "🔴 High" if local_slope > slope_thresh else "🟢 Low"
            boulder_data.append((x, y, size, elev, local_slope, risk))

        df = pd.DataFrame(boulder_data, columns=["X", "Y", "Diameter_Px", "Elevation", "Slope", "Landslide_Risk"])
        st.subheader("📊 Boulder + Landslide Risk Table")
        st.dataframe(df)

        # --- Histogram: Boulder Diameter ---
        st.subheader("📈 Boulder Diameter Distribution")
        fig, ax = plt.subplots()
        df["Diameter_Px"].plot(kind="hist", bins=20, color="lightgreen", ax=ax)
        ax.set_title("Boulder Diameter Histogram")
        ax.set_xlabel("Diameter (px)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # --- Heatmap: Slope ---
        st.subheader("🌄 Normalized Slope Heatmap")
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(slope_norm, cmap="inferno")
        fig2.colorbar(im, ax=ax2)
        ax2.set_title("Slope Heatmap (Normalized)")
        st.pyplot(fig2)

        # --- CSV Download ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Boulder + Risk Data as CSV",
            data=csv,
            file_name="boulder_landslide_data.csv",
            mime="text/csv"
        )


