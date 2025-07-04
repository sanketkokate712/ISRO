import os

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from shapely.geometry import Point
from skimage import measure
from skimage.feature import graycomatrix, graycoprops


def _load_dtm(path):
    """Load digital terrain model"""
    try:
        with rasterio.open(path) as src:
            return src.read(1), src.profile
    except Exception as e:
        print(f"Error loading DTM: {e}")
        return None


def _load_image(path):
    """Load moon surface image"""
    try:
        if path.endswith('.tif'):
            with rasterio.open(path) as src:
                return src.read(1), src.profile
        else:  # For .jpg/.png
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError
            return img, {'transform': [1, 0, 0, 0, 1, 0]}  # Fake geotransform
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


class LunarFeatureDetector:
    def __init__(self, image_path, dtm_path, sun_elevation=15):
        self.landslide_mask = None
        self.boulders_gdf = None
        self.image_path = image_path
        self.dtm_path = dtm_path
        self.sun_elevation = np.radians(sun_elevation)

        # Load datasets
        self.image = _load_image(image_path)
        self.dtm = _load_dtm(dtm_path)

        # Verify data loading
        if self.image is None or self.dtm is None:
            raise ValueError("Failed to load input files. Please check file paths and formats.")

        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        """Coregister and normalize datasets"""
        # Resample image to match DTM resolution
        self.image_resampled = cv2.resize(
            self.image[0],
            self.dtm[0].shape[::-1],
            interpolation=cv2.INTER_CUBIC
        )

        # Calculate illumination correction
        self._calculate_illumination()

        # Normalize image
        self.image_normalized = cv2.normalize(
            self.image_resampled, None, 0, 255, cv2.NORM_MINMAX
        )

    def _calculate_illumination(self):
        """Calculate solar illumination model"""
        x_grad, y_grad = np.gradient(self.dtm[0])

slope = np.arctan(np.sqrt(x_grad ** 2 + y_grad ** 2))
        aspect = np.arctan2(-y_grad, -x_grad)

        self.illum = np.sin(self.sun_elevation) * np.cos(slope) + \
                     np.cos(self.sun_elevation) * np.sin(slope) * np.cos(aspect - np.pi)

    def _calculate_slope(self):
        """Calculate slope in degrees"""
        x_grad, y_grad = np.gradient(self.dtm[0])
        return np.degrees(np.arctan(np.sqrt(x_grad ** 2 + y_grad ** 2)))  # Fixed missing parenthesis

    def _calculate_curvature(self):
        """Calculate profile curvature"""
        dx, dy = np.gradient(self.dtm[0])
        dxx, _ = np.gradient(dx)
        _, dyy = np.gradient(dy)
        return dxx + dyy

    def _calculate_roughness(self):
        """Calculate terrain roughness"""
        return generic_filter(self.dtm[0], np.std, size=3)

    def detect_landslides(self):
        """Detect landslides using terrain and texture features"""
        try:
            # Calculate features
            slope = self._calculate_slope()
            curv = self._calculate_curvature()
            rough = self._calculate_roughness()

            # Texture analysis
            glcm = graycomatrix(
                self.image_normalized.astype(np.uint8),
                [5], [0], 256, symmetric=True
            )
            entropy = graycoprops(glcm, 'entropy')[0, 0]

            # Create feature stack
            np.dstack((
                slope,
                curv,
                rough,
                np.full_like(slope, entropy),
                self.image_normalized / 255.0
            ))

            # Simple thresholding (replace with ML model in production)
            self.landslide_mask = np.where(
                (slope > 25) &
                (curv < -0.05) &
                (self.image_normalized > 150),
                1, 0
            )

            return self.landslide_mask

        except Exception as e:
            print(f"Landslide detection failed: {e}")
            return None

    def detect_boulders(self):
        """Detect boulders using shadow and size characteristics"""
        try:
            # Edge detection
            edges = cv2.Canny(self.image_normalized, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            boulder_data = []
            min_boulder_size = 5  # pixels

            for cnt in contours:
                if len(cnt) >= 5:  # Need at least 5 points for ellipse fitting
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (d1, d2), angle = ellipse

                    # Filter by size
                    if d1 >= min_boulder_size and d2 >= min_boulder_size:
                        # Estimate height from shadow (simplified)
                        height = min(d1, d2) * 0.5 * np.tan(self.sun_elevation)

                        boulder_data.append({
                            'geometry': Point(x, y),
                            'diameter': (d1 + d2) / 2,
                            'height': height,
                            'angle': angle
                        })

            self.boulders_gdf = gpd.GeoDataFrame(boulder_data)
            return self.boulders_gdf

        except Exception as e:
            print(f"Boulder detection failed: {e}")
            return None

    def generate_outputs(self, output_dir="results"):
        """Generate all detection outputs"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Create annotated map
            plt.figure(figsize=(12, 8))
            plt.imshow(self.image_normalized, cmap='gray')

            # Add landslides
            if hasattr(self, 'landslide_mask'):
                contours = measure.find_contours(self.landslide_mask, 0.5)
                for contour in contours:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')

            # Add boulders
            if hasattr(self, 'boulders_gdf') and not self.boulders_gdf.empty:
                plt.scatter(
                    self.boulders_gdf.geometry.x,
                    self.boulders_gdf.geometry.y,
                    s=self.boulders_gdf['diameter'],
                    c='blue',
                    alpha=0.5
                )

            plt.title('Lunar Feature Detection')
            plt.savefig(os.path.join(output_dir, 'detection_results.png'))
            plt.close()

            # Save statistics
            stats = {}
            if hasattr(self, 'landslide_mask'):
                stats['landslide_area_pixels'] = np.sum(self.landslide_mask)
            if hasattr(self, 'boulders_gdf') and not self.boulders_gdf.empty:
                stats['boulder_count'] = len(self.boulders_gdf)
                stats['avg_boulder_diameter'] = self.boulders_gdf['diameter'].mean()

            with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
                for k, v in stats.items():
                    f.write(f"{k}: {v}\n")

            # Save vector data
            if hasattr(self, 'boulders_gdf') and not self.boulders_gdf.empty:
                self.boulders_gdf.to_file(
                    os.path.join(output_dir, 'boulders.geojson'),
                    driver='GeoJSON'
                )

            print(f"Success! Results saved to {output_dir} directory")
            return True

        except Exception as e:
            print(f"Output generation failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize with your files
    detector = LunarFeatureDetector(
        image_path="moon_surface.jpg",
        dtm_path="us_nga_egm96_15.tif",
        sun_elevation=15
    )

    # Run detections
    landslides = detector.detect_landslides()
    boulders = detector.detect_boulders()

    # Generate outputs
    detector.generate_outputs()