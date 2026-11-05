"""
Unit and integration tests for CrisisVision pixel-based damage hotspot mapping.

Tests verify:
  - Severity mapping for all xBD damage classes (0, 1, 2, 3, 4)
  - Handling of empty building list
  - Handling of scenes with no-damage-only buildings
  - Spatial aggregation of a single destroyed building
  - Clustering of multiple damaged buildings close together (single hotspot)
  - Spatial separation of multiple damaged buildings far apart (two hotspots)
  - Correct hotspot centroid and bounding box calculations
  - Correct damage-class breakdown counts and associated building IDs
  - Priority score calculation and category assignment
  - Clamping and handling of image boundaries
  - Validation of risk mapping JSON schema
  - Continuous heatmap dimension matching
  - Configurable kernel spread (sigma_pixels) and activation thresholds
  - Image overlay generation
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.risk_mapping import (
    calculate_building_severity,
    generate_damage_heatmap,
    extract_hotspots,
    calculate_priority_score,
    assign_priority_category,
    validate_risk_data,
    save_risk_map_image,
    RiskMappingValidationError,
    DAMAGE_SEVERITY_WEIGHTS,
)


class TestSeverityMapping:
    """Tests damage class to normalized severity score conversion."""

    def test_severity_values(self):
        assert calculate_building_severity({"damage_class_id": 0}) == 0.0
        assert calculate_building_severity({"damage_class_id": 1}) == 0.0
        assert pytest.approx(calculate_building_severity({"damage_class_id": 2}), 0.01) == 1.0 / 3.0
        assert pytest.approx(calculate_building_severity({"damage_class_id": 3}), 0.01) == 2.0 / 3.0
        assert calculate_building_severity({"damage_class_id": 4}) == 1.0

    def test_severity_from_labels(self):
        assert calculate_building_severity({"damage_label": "no_damage"}) == 0.0
        assert pytest.approx(calculate_building_severity({"damage_label": "minor_damage"}), 0.01) == 1.0 / 3.0
        assert pytest.approx(calculate_building_severity({"damage_label": "major_damage"}), 0.01) == 2.0 / 3.0
        assert calculate_building_severity({"damage_label": "destroyed"}) == 1.0


class TestDamageHeatmapGeneration:
    """Tests distance-weighted Gaussian spatial aggregation."""

    def test_empty_buildings_heatmap_is_zeros(self):
        data = {
            "analysis_id": "test_empty",
            "image_width": 200,
            "image_height": 200,
            "buildings": [],
        }
        hm = generate_damage_heatmap(data, sigma_pixels=30.0)
        assert hm.shape == (200, 200)
        assert np.all(hm == 0.0)

    def test_no_damage_buildings_do_not_increase_heatmap(self):
        """No-damage buildings must have severity 0 and contribute 0 to the heatmap."""
        data = {
            "analysis_id": "test_no_damage",
            "image_width": 200,
            "image_height": 200,
            "buildings": [
                {
                    "building_id": "B001",
                    "centroid": {"x": 100.0, "y": 100.0},
                    "damage_class_id": 1,
                    "damage_label": "no_damage",
                },
                {
                    "building_id": "B002",
                    "centroid": {"x": 120.0, "y": 120.0},
                    "damage_class_id": 1,
                    "damage_label": "no_damage",
                },
            ],
        }
        hm = generate_damage_heatmap(data, sigma_pixels=30.0)
        assert hm.max() == 0.0

    def test_single_destroyed_building_peak_at_centroid(self):
        """Destroyed building (severity 1.0) must produce a peak of 1.0 at its exact centroid."""
        cx, cy = 150.0, 100.0
        data = {
            "analysis_id": "test_single_destroyed",
            "image_width": 300,
            "image_height": 300,
            "buildings": [
                {
                    "building_id": "B001",
                    "centroid": {"x": cx, "y": cy},
                    "damage_class_id": 4,
                    "damage_label": "destroyed",
                }
            ],
        }
        sigma = 40.0
        hm = generate_damage_heatmap(data, sigma_pixels=sigma)

        # Peak at centroid should be 1.0
        assert pytest.approx(hm[int(cy), int(cx)], 0.01) == 1.0

        # At distance sigma, value should be 1.0 * exp(-0.5) ≈ 0.6065
        assert pytest.approx(hm[int(cy), int(cx + sigma)], 0.02) == np.exp(-0.5)

    def test_dimensions_match_original_image(self):
        w, h = 800, 600
        data = {
            "analysis_id": "test_dims",
            "image_width": w,
            "image_height": h,
            "buildings": [],
        }
        hm = generate_damage_heatmap(data, sigma_pixels=50.0)
        assert hm.shape == (h, w)


class TestHotspotClusteringAndExtraction:
    """Tests identifying hotspot regions, clustering nearby buildings, and separation."""

    def test_empty_buildings_returns_zero_hotspots(self):
        data = {
            "analysis_id": "test_empty_hotspots",
            "image_width": 500,
            "image_height": 500,
            "buildings": [],
        }
        risk = extract_hotspots(data, sigma_pixels=50.0, threshold=0.5)
        assert risk["summary"]["total_hotspots"] == 0
        assert risk["hotspots"] == []
        validate_risk_data(risk)

    def test_no_damage_buildings_returns_zero_hotspots(self):
        data = {
            "analysis_id": "test_no_damage_hotspots",
            "image_width": 500,
            "image_height": 500,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": 250.0, "y": 250.0}, "damage_class_id": 1, "damage_label": "no_damage"}
            ],
        }
        risk = extract_hotspots(data, sigma_pixels=50.0, threshold=0.5)
        assert risk["summary"]["total_hotspots"] == 0
        validate_risk_data(risk)

    def test_single_destroyed_building_forms_one_hotspot(self):
        cx, cy = 300.0, 300.0
        data = {
            "analysis_id": "test_single_hotspot",
            "image_width": 600,
            "image_height": 600,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": cx, "y": cy}, "damage_class_id": 4, "damage_label": "destroyed"}
            ],
        }
        risk = extract_hotspots(data, sigma_pixels=40.0, threshold=0.5, min_area_pixels=10)
        assert risk["summary"]["total_hotspots"] == 1
        h = risk["hotspots"][0]
        assert h["hotspot_id"] == "H001"
        assert pytest.approx(h["centroid"]["x"], 2.0) == cx
        assert pytest.approx(h["centroid"]["y"], 2.0) == cy
        assert h["building_count"] == 1
        assert h["destroyed_count"] == 1
        assert h["building_ids"] == ["B001"]
        validate_risk_data(risk)

    def test_multiple_damaged_buildings_close_merge_into_one_hotspot(self):
        """Two destroyed buildings within 25 px (< sigma=50) should merge into 1 cluster."""
        data = {
            "analysis_id": "test_cluster_merge",
            "image_width": 600,
            "image_height": 600,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": 200.0, "y": 200.0}, "damage_class_id": 4, "damage_label": "destroyed"},
                {"building_id": "B002", "centroid": {"x": 225.0, "y": 215.0}, "damage_class_id": 3, "damage_label": "major_damage"},
            ],
        }
        risk = extract_hotspots(data, sigma_pixels=50.0, threshold=0.5, min_area_pixels=10)
        assert risk["summary"]["total_hotspots"] == 1
        h = risk["hotspots"][0]
        assert h["building_count"] == 2
        assert h["destroyed_count"] == 1
        assert h["major_damage_count"] == 1
        assert "B001" in h["building_ids"]
        assert "B002" in h["building_ids"]

    def test_multiple_damaged_buildings_far_apart_create_separate_hotspots(self):
        """Two destroyed buildings separated by 500 px (>> sigma=40) must form 2 distinct hotspots."""
        data = {
            "analysis_id": "test_separate_clusters",
            "image_width": 1000,
            "image_height": 1000,
            "buildings": [
                # Cluster A at (200, 200)
                {"building_id": "B001", "centroid": {"x": 200.0, "y": 200.0}, "damage_class_id": 4, "damage_label": "destroyed"},
                # Cluster B at (800, 800)
                {"building_id": "B002", "centroid": {"x": 800.0, "y": 800.0}, "damage_class_id": 4, "damage_label": "destroyed"},
            ],
        }
        risk = extract_hotspots(data, sigma_pixels=40.0, threshold=0.5, min_area_pixels=10)
        assert risk["summary"]["total_hotspots"] == 2
        ids = [h["hotspot_id"] for h in risk["hotspots"]]
        assert ids == ["H001", "H002"]

        # Hotspots must be spatially separated
        assert risk["hotspots"][0]["centroid"]["x"] < 500
        assert risk["hotspots"][1]["centroid"]["x"] > 500


class TestPriorityScoringAndCategories:
    """Tests priority score calculation and category classification."""

    def test_critical_priority_dense_destroyed(self):
        """Multiple destroyed buildings should achieve critical priority."""
        associated = [
            {"damage_class_id": 4, "damage_label": "destroyed"},
            {"damage_class_id": 4, "damage_label": "destroyed"},
            {"damage_class_id": 4, "damage_label": "destroyed"},
            {"damage_class_id": 3, "damage_label": "major_damage"},
        ]
        score, cat = calculate_priority_score(associated, peak_intensity=2.5)
        assert score >= 0.75
        assert cat == "critical_priority"

    def test_moderate_priority_minor_damage(self):
        """Minor damage should yield moderate or lower priority."""
        associated = [
            {"damage_class_id": 2, "damage_label": "minor_damage"},
        ]
        score, cat = calculate_priority_score(associated, peak_intensity=0.6)
        assert score < 0.50
        assert cat in ["moderate_priority", "low_priority"]

    def test_category_assignment_boundaries(self):
        assert assign_priority_category(0.85) == "critical_priority"
        assert assign_priority_category(0.60) == "high_priority"
        assert assign_priority_category(0.35) == "moderate_priority"
        assert assign_priority_category(0.15) == "low_priority"


class TestBoundaryHandlingAndConfigurability:
    """Tests edge boundaries and configurable parameters."""

    def test_buildings_on_image_boundaries(self):
        """Buildings right at (0, 0) and (width-1, height-1) must have valid clamped coords."""
        w, h = 400, 400
        data = {
            "analysis_id": "test_edges",
            "image_width": w,
            "image_height": h,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": 2.0, "y": 2.0}, "damage_class_id": 4, "damage_label": "destroyed"},
                {"building_id": "B002", "centroid": {"x": 398.0, "y": 398.0}, "damage_class_id": 4, "damage_label": "destroyed"},
            ],
        }
        risk = extract_hotspots(data, sigma_pixels=30.0, threshold=0.5, min_area_pixels=10)
        assert risk["summary"]["total_hotspots"] == 2
        for hotspot in risk["hotspots"]:
            bbox = hotspot["bounding_box"]
            assert 0 <= bbox["x_min"] <= bbox["x_max"] < w
            assert 0 <= bbox["y_min"] <= bbox["y_max"] < h

    def test_configurable_threshold_filters(self):
        """Higher threshold suppresses weaker hotspots."""
        data = {
            "analysis_id": "test_threshold",
            "image_width": 400,
            "image_height": 400,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": 200.0, "y": 200.0}, "damage_class_id": 3, "damage_label": "major_damage"}
            ],
        }
        # Major damage has severity 2/3 ≈ 0.667.
        # Threshold 0.5 -> should be detected
        risk_low_th = extract_hotspots(data, sigma_pixels=30.0, threshold=0.5, min_area_pixels=5)
        assert risk_low_th["summary"]["total_hotspots"] == 1

        # Threshold 0.8 -> peak 0.667 is below 0.8 -> should NOT be detected
        risk_high_th = extract_hotspots(data, sigma_pixels=30.0, threshold=0.8, min_area_pixels=5)
        assert risk_high_th["summary"]["total_hotspots"] == 0


class TestSchemaValidation:
    """Tests schema validation on generated risk data."""

    def test_schema_validation_passes(self):
        data = {
            "analysis_id": "test_valid",
            "image_width": 500,
            "image_height": 500,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": 200.0, "y": 200.0}, "damage_class_id": 4, "damage_label": "destroyed"}
            ],
        }
        risk = extract_hotspots(data)
        validate_risk_data(risk)

    def test_corrupted_schema_raises(self):
        corrupted = {
            "analysis_id": "test",
            "coordinate_system": "geographic_gps",  # invalid
            "image_width": 500,
            "image_height": 500,
            "risk_mapping_method": "gaussian_damage_severity",
            "parameters": {},
            "summary": {},
            "hotspots": [],
            "limitations": [],
        }
        with pytest.raises(RiskMappingValidationError, match="coordinate_system"):
            validate_risk_data(corrupted)


class TestRiskMapImageGeneration:
    """Tests rendering of composite risk map overlay images."""

    def test_save_risk_map_overlay(self, tmp_path):
        base_path = tmp_path / "post_test.png"
        Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save(base_path)

        data = {
            "analysis_id": "test_overlay",
            "image_width": 300,
            "image_height": 300,
            "buildings": [
                {"building_id": "B001", "centroid": {"x": 150.0, "y": 150.0}, "damage_class_id": 4, "damage_label": "destroyed"}
            ],
        }
        hm = generate_damage_heatmap(data, sigma_pixels=40.0)
        risk = extract_hotspots(data, sigma_pixels=40.0, threshold=0.5, min_area_pixels=10)

        out_path = str(tmp_path / "risk_map.png")
        save_risk_map_image(str(base_path), hm, risk, out_path)

        assert Path(out_path).exists()
        out_img = Image.open(out_path)
        assert out_img.size == (300, 300)

