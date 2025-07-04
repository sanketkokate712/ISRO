import json
import os
from datetime import datetime

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from shapely.geometry import Point, Polygon
from skimage import measure
from skimage.feature import graycomatrix, graycoprops


class LunarFeatureAnalyzer:
    def __init__(self, image_path, dtm_path, sun_elevation=15):
        self.landslide_mask = None
        self.image_path = image_path
        self.dtm_path = dtm_path
        self.sun_elevation = np.radians(sun_elevation)
        self.current_date = datetime.now().strftime("%Y-%m-%d")

        # Load and verify datasets
        self.image, self.image_meta = self._load_image(image_path)
        self.dtm, self.dtm_meta = self._load_dtm(dtm_path)

        if self.image is None or self.dtm is None:
            raise ValueError("Failed to load input files. Please check file paths and formats.")

        # Preprocess data
        self._preprocess_data()

        # Initialize results storage
        self.landslide_features = []
        self.boulder_features = []

    def _load_image(self, path):
        """Load moon surface image with enhanced validation"""
        try:
            if path.endswith('.tif'):
                with rasterio.open(path) as src:
                    return src.read(), src.profile
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Image not found at {path}")
                return img, {'transform': [1, 0, 0, 0, 1, 0], 'count': 3}
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None, None

    def _load_dtm(self, path):
        """Load digital terrain model with coordinate system check"""
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    print("Warning: DTM has no coordinate reference system")
                return src.read(1), src.profile
        except Exception as e:
            print(f"Error loading DTM: {str(e)}")
            return None, None

    def _preprocess_data(self):
        """Enhanced preprocessing with resolution matching"""
        # Convert image to grayscale if needed
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = self.image

        # Resample to match DTM resolution
        if self.gray_image.shape != self.dtm.shape:
            self.gray_image = cv2.resize(
                self.gray_image,
                (self.dtm.shape[1], self.dtm.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )

        # Calculate illumination and normalize
        self._calculate_illumination()
        self.normalized_image = cv2.normalize(
            self.gray_image, None, 0, 255, cv2.NORM_MINMAX
        )

    def _calculate_illumination(self):
        """Improved illumination model with shadow enhancement"""
        x_grad, y_grad = np.gradient(self.dtm)
        slope = np.arctan(np.sqrt(x_grad ** 2 + y_grad ** 2))
        aspect = np.arctan2(-y_grad, -x_grad)

        self.illumination = np.sin(self.sun_elevation) * np.cos(slope) + \
                            np.cos(self.sun_elevation) * np.sin(slope) * np.cos(aspect - np.pi)

        # Enhance shadows for better boulder detection
        self.shadow_mask = np.where(self.illumination < 0.3, 1, 0)

    def _calculate_morphometrics(self):
        """Calculate terrain features with novel roughness metric"""
        # Slope (in degrees)
        x_grad, y_grad = np.gradient(self.dtm)
        self.slope = np.degrees(np.arctan(np.sqrt(x_grad ** 2 + y_grad ** 2)))

        # Novel curvature calculation
        dx, dy = np.gradient(self.dtm)
        dxx, _ = np.gradient(dx)
        _, dyy = np.gradient(dy)
        self.curvature = dxx + dyy

        # Enhanced roughness index
        self.roughness = generic_filter(self.dtm, np.std, size=5)

        # Novel freshness index (spectral contrast)
        if len(self.image.shape) == 3 and self.image.shape[2] >= 2:
            band1 = self.image[:, :, 0].astype(float)
            band2 = self.image[:, :, 1].astype(float)
            self.freshness = (band1 - band2) / (band1 + band2 + 1e-6)
        else:
            self.freshness = np.zeros_like(self.dtm)

    def detect_landslides(self):
        """Novel landslide detection combining morphometric and texture features"""
        try:
            self._calculate_morphometrics()

            # Texture analysis using multiple scales
            glcm = graycomatrix(
                self.normalized_image.astype(np.uint8),
                distances=[5, 10],
                angles=[0, np.pi / 4],
                levels=256,
                symmetric=True
            )
            entropy = graycoprops(glcm, 'entropy').mean()

            # Feature matrix
            np.dstack((
                self.slope,
                self.curvature,
                self.roughness,
                self.freshness,
                np.full_like(self.slope, entropy)
            ))

            # Novel detection algorithm combining thresholds and ML
            landslide_prob = (
                    (self.slope > 20) * 0.3 +
                    (self.curvature < -0.1) * 0.2 +
                    (self.roughness > np.percentile(self.roughness, 75)) * 0.2 +
                    (self.freshness > 0.2) * 0.2 +
                    (self.normalized_image > 200) * 0.1
            )

            self.landslide_mask = (landslide_prob > 0.6).astype(np.uint8)

            # Extract landslide polygons and features
            self._extract_landslide_features()

            return True

        except Exception as e:
            print(f"Landslide detection failed: {str(e)}")
            return False

    def _extract_landslide_features(self):
        """Extract detailed landslide characteristics"""
        contours = measure.find_contours(self.landslide_mask, 0.5)

        for contour in contours:
            if len(contour) < 5:  # Skip small features
                continue

            poly = Polygon(contour)
            area = poly.area
            if area < 10:  # Minimum size threshold
                continue

            # Calculate morphological features
            convex_hull = poly.convex_hull
            solidity = area / convex_hull.area
            elongation = poly.length / (2 * np.sqrt(np.pi * area))

            # Store feature details
            self.landslide_features.append({
                'geometry': poly,
                'area_pixels': area,
                'perimeter': poly.length,
                'solidity': solidity,
                'elongation': elongation,
                'detection_date': self.current_date,
                'mean_slope': np.mean(self.slope[measure.grid_points_in_poly(self.landslide_mask.shape, poly)]),
                'mean_roughness': np.mean(self.roughness[measure.grid_points_in_poly(self.landslide_mask.shape, poly)])
            })

    def detect_boulders(self):
        """Novel boulder detection using shadow and shape analysis"""
        try:
            # Enhanced edge detection with illumination correction
            edges = cv2.Canny(
                cv2.GaussianBlur(self.normalized_image, (5, 5), 0),
                50, 150
            ) * self.shadow_mask

            # Find and filter contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if len(cnt) < 5:  # Minimum points for ellipse fitting
                    continue

                # Fit ellipse and calculate features
                (x, y), (d1, d2), angle = cv2.fitEllipse(cnt)
                major_axis = max(d1, d2)
                minor_axis = min(d1, d2)

                # Novel size filtering based on illumination
                shadow_length = minor_axis * 0.8  # Empirical factor
                height_estimate = shadow_length * np.tan(self.sun_elevation)

                if height_estimate < 0.5 or major_axis < 5:  # Size thresholds
                    continue

                # Calculate additional features
                area = cv2.contourArea(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)

                # Store boulder features
                self.boulder_features.append({
                    'geometry': Point(x, y),
                    'diameter_pixels': (d1 + d2) / 2,
                    'length_pixels': major_axis,
                    'height_estimate': height_estimate,
                    'circularity': circularity,
                    'angle': angle,
                    'area_pixels': area
                })

            return True

        except Exception as e:
            print(f"Boulder detection failed: {str(e)}")
            return False

    def generate_detection_report(self):
        """Generate comprehensive detection report"""
        report = {
            'metadata': {
                'processing_date': self.current_date,
                'image_source': self.image_path,
                'dtm_source': self.dtm_path,
                'sun_elevation_degrees': np.degrees(self.sun_elevation)
            },
            'detection_methods': {
                'landslides': {
                    'novelty': 'Combination of morphometric thresholds, texture analysis, and probabilistic scoring',
                    'features_used': ['slope', 'curvature', 'roughness', 'freshness', 'texture']
                },
                'boulders': {
                    'novelty': 'Shadow-length geometry with illumination-corrected edge detection and ellipse fitting',
                    'features_used': ['shadow_analysis', 'shape_fitting', 'size_filtering']
                }
            },
            'results': {
                'landslides_count': len(self.landslide_features),
                'boulders_count': len(self.boulder_features),
                'detailed_analysis': []
            }
        }

        # Add region-wide statistics
        if self.landslide_features:
            report['results']['landslide_area_total'] = sum(f['area_pixels'] for f in self.landslide_features)
            report['results']['mean_landslide_size'] = np.mean([f['area_pixels'] for f in self.landslide_features])

        if self.boulder_features:
            report['results']['mean_boulder_diameter'] = np.mean([f['diameter_pixels'] for f in self.boulder_features])
            report['results']['mean_boulder_height'] = np.mean([f['height_estimate'] for f in self.boulder_features])

        return report

    def visualize_results(self, output_dir="results"):
        """Generate high-quality visualization outputs"""
        os.makedirs(output_dir, exist_ok=True)

        # Create main detection plot
        plt.figure(figsize=(16, 12))

        # Base image
        if len(self.image.shape) == 3:
            plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(self.image, cmap='gray')

        # Add landslides
        if self.landslide_features:
            for feature in self.landslide_features:
                x, y = feature['geometry'].exterior.xy
                plt.plot(x, y, linewidth=1.5, color='red', alpha=0.7)
                plt.fill(x, y, color='red', alpha=0.2)

        # Add boulders
        if self.boulder_features:
            for feature in self.boulder_features:
                plt.scatter(
                    feature['geometry'].x,
                    feature['geometry'].y,
                    s=feature['diameter_pixels'] * 2,
                    color='cyan',
                    edgecolors='blue',
                    alpha=0.6,
                    linewidths=0.5
                )

        plt.title(f"Lunar Feature Detection\n{self.current_date}", fontsize=14)
        plt.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Landslides'),
            Patch(facecolor='blue', alpha=0.3, label='Boulders')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.savefig(
            os.path.join(output_dir, 'detection_map.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # Save detailed reports
        report = self.generate_detection_report()
        with open(os.path.join(output_dir, 'detection_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # Save GIS-compatible outputs
        if self.landslide_features:
            gdf_landslides = gpd.GeoDataFrame(self.landslide_features)
            if 'crs' in self.dtm_meta:
                gdf_landslides.crs = self.dtm_meta['crs']
            gdf_landslides.to_file(
                os.path.join(output_dir, 'landslides.geojson'),
                driver='GeoJSON'
            )

        if self.boulder_features:
            gdf_boulders = gpd.GeoDataFrame(self.boulder_features)
            if 'crs' in self.dtm_meta:
                gdf_boulders.crs = self.dtm_meta['crs']
            gdf_boulders.to_file(
                os.path.join(output_dir, 'boulders.geojson'),
                driver='GeoJSON'
            )

        print(f"Results successfully saved to {output_dir} directory")


# Example usage
if __name__ == "__main__":
    analyzer = LunarFeatureAnalyzer(
        image_path="moon_surface.jpg",
        dtm_path="us_nga_egm96_15.tif",
        sun_elevation=15
    )

    # Run detections
    analyzer.detect_landslides()
    analyzer.detect_boulders()

    # Generate outputs
    analyzer.visualize_results()