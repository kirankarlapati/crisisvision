"""
Unit and integration tests for CrisisVision building-level extraction.

Tests verify:
  - Correct centroid calculation on synthetic shapes
  - Correct bounding-box calculation (inclusive coordinates)
  - Correct pixel area calculation
  - Correct damage-class majority voting
  - Exclusion of background pixels (class 0) from damage voting
  - Handling of empty masks
  - Filtering of small components below min_area threshold
  - Handling of multiple separated components
  - Building ID generation and sequential numbering (B001, B002, ...)
  - JSON schema validation & constraint enforcement
  - Coordinates remaining within image boundaries
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.building_extraction import (
    extract_buildings,
    validate_building_data,
    save_annotated_buildings_image,
    BuildingValidationError,
    DAMAGE_LABELS,
)


class TestBuildingExtractionGeometry:
    """Tests geometric properties: centroids, bounding boxes, areas, and IDs."""

    def test_single_rectangular_building(self):
        """A known 20x30 rectangle at rows [100..119], cols [200..229]."""
        height, width = 500, 600
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        # Draw rectangle: row 100..119 (height 20), col 200..229 (width 30)
        b_map[100:120, 200:230] = 1
        d_map[100:120, 200:230] = 1  # class 1: no_damage

        result = extract_buildings(b_map, d_map, analysis_id="test_geo_1", min_area=25)

        assert result["total_buildings"] == 1
        assert len(result["buildings"]) == 1

        b = result["buildings"][0]
        assert b["building_id"] == "B001"
        assert b["area_pixels"] == 20 * 30  # 600 px

        # Centroid should be exactly center:
        # y = (100 + 119) / 2 = 109.5
        # x = (200 + 229) / 2 = 214.5
        assert pytest.approx(b["centroid"]["y"], 0.1) == 109.5
        assert pytest.approx(b["centroid"]["x"], 0.1) == 214.5

        # Bounding box [x_min, y_min, x_max, y_max]
        bbox = b["bounding_box"]
        assert bbox["x_min"] == 200
        assert bbox["y_min"] == 100
        assert bbox["x_max"] == 229
        assert bbox["y_max"] == 119

        # Coordinate system
        assert result["coordinate_system"] == "image_pixel"
        assert result["image_width"] == width
        assert result["image_height"] == height

    def test_multiple_buildings_spatial_ordering(self):
        """Multiple buildings should be ordered top-to-bottom and receive sequential IDs."""
        height, width = 400, 400
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        # Building 1: top left
        b_map[20:50, 20:50] = 1
        d_map[20:50, 20:50] = 1

        # Building 2: top right
        b_map[20:50, 200:230] = 1
        d_map[20:50, 200:230] = 2

        # Building 3: bottom
        b_map[250:280, 100:130] = 1
        d_map[250:280, 100:130] = 3

        result = extract_buildings(b_map, d_map, analysis_id="test_multi")

        assert result["total_buildings"] == 3
        ids = [b["building_id"] for b in result["buildings"]]
        assert ids == ["B001", "B002", "B003"]

        # First two should have smaller y than third
        assert result["buildings"][0]["centroid"]["y"] < result["buildings"][2]["centroid"]["y"]
        assert result["buildings"][1]["centroid"]["y"] < result["buildings"][2]["centroid"]["y"]


class TestDamageMajorityVoting:
    """Tests damage assignment, background exclusion, and confidence calculation."""

    def test_majority_voting_major_damage(self):
        """Building has 70% major damage (class 3) and 30% minor damage (class 2)."""
        height, width = 200, 200
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        # 10x10 patch = 100 px
        b_map[50:60, 50:60] = 1
        # 70 pixels class 3
        d_map[50:57, 50:60] = 3  # 7 * 10 = 70 pixels
        # 30 pixels class 2
        d_map[57:60, 50:60] = 2  # 3 * 10 = 30 pixels

        result = extract_buildings(b_map, d_map, min_area=25)
        b = result["buildings"][0]

        assert b["damage_class_id"] == 3
        assert b["damage_label"] == "major_damage"
        assert pytest.approx(b["confidence"], 0.01) == 0.70

        dist = b["damage_distribution"]
        assert pytest.approx(dist["major_damage"], 0.01) == 0.70
        assert pytest.approx(dist["minor_damage"], 0.01) == 0.30
        assert dist["no_damage"] == 0.0
        assert dist["destroyed"] == 0.0

    def test_background_pixels_excluded_from_voting(self):
        """Footprint contains 50% background (0) and 50% destroyed (4).

        Background pixels must NOT be voted on; destroyed should dominate 100% of valid vote.
        """
        height, width = 200, 200
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        b_map[50:60, 50:60] = 1  # 100 pixels
        # 50 pixels are class 0 (background)
        d_map[50:55, 50:60] = 0
        # 50 pixels are class 4 (destroyed)
        d_map[55:60, 50:60] = 4

        result = extract_buildings(b_map, d_map, min_area=25)
        b = result["buildings"][0]

        assert b["damage_class_id"] == 4
        assert b["damage_label"] == "destroyed"
        # Among valid pixels, 100% are destroyed
        assert pytest.approx(b["confidence"], 0.01) == 1.00
        assert pytest.approx(b["damage_distribution"]["destroyed"], 0.01) == 1.00

    def test_purely_background_footprint_fallback(self):
        """If all pixels within footprint happen to be class 0, fallback gracefully."""
        height, width = 100, 100
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        b_map[20:40, 20:40] = 1
        d_map[20:40, 20:40] = 0  # all 0

        result = extract_buildings(b_map, d_map, min_area=25)
        assert result["total_buildings"] == 1
        b = result["buildings"][0]
        assert b["damage_class_id"] == 1
        assert b["damage_label"] == "no_damage"
        assert b["confidence"] == 0.0


class TestFilteringAndEdgeCases:
    """Tests empty masks, noise speck filtering, boundary conditions."""

    def test_empty_mask_handling(self):
        """Completely black building mask returns empty building list without error."""
        b_map = np.zeros((300, 300), dtype=np.uint8)
        d_map = np.zeros((300, 300), dtype=np.uint8)

        result = extract_buildings(b_map, d_map, analysis_id="empty_test")
        assert result["total_buildings"] == 0
        assert result["buildings"] == []
        assert result["damage_summary"] == {
            "no_damage": 0,
            "minor_damage": 0,
            "major_damage": 0,
            "destroyed": 0,
        }

    def test_noise_filtering_min_area(self):
        """Small speckles below min_area must be excluded."""
        height, width = 200, 200
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        # Speckle 1: 3x3 = 9 pixels (should be filtered out)
        b_map[10:13, 10:13] = 1
        d_map[10:13, 10:13] = 1

        # Speckle 2: 1x4 = 4 pixels (should be filtered out)
        b_map[50:51, 50:54] = 1
        d_map[50:51, 50:54] = 2

        # Valid building: 10x10 = 100 pixels (should be kept)
        b_map[100:110, 100:110] = 1
        d_map[100:110, 100:110] = 1

        result = extract_buildings(b_map, d_map, min_area=25)
        assert result["total_buildings"] == 1
        assert result["buildings"][0]["area_pixels"] == 100

    def test_coordinates_clamped_to_boundaries(self):
        """Buildings right at the image edge have valid clamped coordinates."""
        height, width = 100, 100
        b_map = np.zeros((height, width), dtype=np.uint8)
        d_map = np.zeros((height, width), dtype=np.uint8)

        # Building at bottom-right corner [80..99, 80..99]
        b_map[80:100, 80:100] = 1
        d_map[80:100, 80:100] = 1

        result = extract_buildings(b_map, d_map, min_area=25)
        b = result["buildings"][0]
        bbox = b["bounding_box"]

        assert 0 <= bbox["x_min"] <= bbox["x_max"] < width
        assert 0 <= bbox["y_min"] <= bbox["y_max"] < height
        assert 0 <= b["centroid"]["x"] < width
        assert 0 <= b["centroid"]["y"] < height
        assert bbox["x_max"] == 99
        assert bbox["y_max"] == 99


class TestSchemaValidation:
    """Tests schema validation on valid and intentionally corrupted structures."""

    def test_valid_schema_passes(self):
        b_map = np.zeros((100, 100), dtype=np.uint8)
        d_map = np.zeros((100, 100), dtype=np.uint8)
        b_map[20:40, 20:40] = 1
        d_map[20:40, 20:40] = 1

        result = extract_buildings(b_map, d_map)
        # Should not raise
        validate_building_data(result)

    def test_missing_root_key_raises(self):
        data = {
            "coordinate_system": "image_pixel",
            "image_width": 100,
            "image_height": 100,
            "buildings": [],
        }
        with pytest.raises(BuildingValidationError, match="analysis_id"):
            validate_building_data(data)

    def test_invalid_coordinate_system_raises(self):
        data = {
            "analysis_id": "test",
            "coordinate_system": "geographic_wgs84",  # invalid
            "image_width": 100,
            "image_height": 100,
            "buildings": [],
        }
        with pytest.raises(BuildingValidationError, match="coordinate_system"):
            validate_building_data(data)

    def test_out_of_bounds_centroid_raises(self):
        data = {
            "analysis_id": "test",
            "coordinate_system": "image_pixel",
            "image_width": 100,
            "image_height": 100,
            "buildings": [{
                "building_id": "B001",
                "centroid": {"x": 150.0, "y": 50.0},  # x exceeds width
                "bounding_box": {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20},
                "area_pixels": 100,
                "damage_class_id": 1,
                "damage_label": "no_damage",
                "confidence": 1.0,
                "damage_distribution": {
                    "no_damage": 1.0, "minor_damage": 0.0, "major_damage": 0.0, "destroyed": 0.0
                },
            }],
        }
        with pytest.raises(BuildingValidationError, match="centroid"):
            validate_building_data(data)


class TestAnnotationImageGeneration:
    """Tests generation of visual bounding box and centroid annotated images."""

    def test_save_annotated_image(self, tmp_path):
        from PIL import Image

        # Create dummy post image
        img_path = tmp_path / "post_test.png"
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(img_path)

        buildings_data = {
            "analysis_id": "test",
            "coordinate_system": "image_pixel",
            "image_width": 200,
            "image_height": 200,
            "total_buildings": 1,
            "damage_summary": {"no_damage": 1, "minor_damage": 0, "major_damage": 0, "destroyed": 0},
            "buildings": [{
                "building_id": "B001",
                "centroid": {"x": 50.0, "y": 50.0},
                "bounding_box": {"x_min": 40, "y_min": 40, "x_max": 60, "y_max": 60},
                "area_pixels": 400,
                "damage_class_id": 1,
                "damage_label": "no_damage",
                "confidence": 1.0,
                "damage_distribution": {
                    "no_damage": 1.0, "minor_damage": 0.0, "major_damage": 0.0, "destroyed": 0.0
                },
            }],
        }

        out_path = str(tmp_path / "annotated.png")
        save_annotated_buildings_image(str(img_path), buildings_data, out_path)

        assert Path(out_path).exists()
        out_img = Image.open(out_path)
        assert out_img.size == (200, 200)

