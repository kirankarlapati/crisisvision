"""
CrisisVision Building-Level Extraction Service.

Converts pixel-level building masks and damage maps into discrete building
instances with image pixel coordinates, bounding boxes, areas, majority-voted
damage classes, and uncertainty distributions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

# xBD Damage Class ID to standardized snake_case label mapping
DAMAGE_LABELS: Dict[int, str] = {
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}

# Human-readable display titles
DAMAGE_DISPLAY_NAMES: Dict[int, str] = {
    0: "Background",
    1: "No damage",
    2: "Minor damage",
    3: "Major damage",
    4: "Destroyed",
}

# Color palette (RGB) corresponding to damage classes
DAMAGE_HEX_COLORS: Dict[str, str] = {
    "no_damage": "#00c800",       # Green
    "minor_damage": "#ffdc00",    # Yellow
    "major_damage": "#ff8c00",    # Orange
    "destroyed": "#dc0000",       # Red
}

DAMAGE_RGB_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 200, 0),
    2: (255, 220, 0),
    3: (255, 140, 0),
    4: (220, 0, 0),
}


class BuildingValidationError(ValueError):
    """Raised when extracted building data violates JSON schema constraints."""


def extract_buildings(
    building_map: np.ndarray,
    damage_map: np.ndarray,
    analysis_id: str = "analysis_001",
    min_area: int = 25,
    max_area: Optional[int] = None,
    connectivity: int = 2,
) -> Dict[str, Any]:
    """Extract individual building instances from semantic segmentation masks.

    Applies connected-component analysis (8-connectivity by default) on the binary
    building mask, filters out small false positives (noise), computes image pixel
    centroids and bounding boxes, and aggregates per-building damage classifications
    via majority voting over valid non-background pixels.

    Args:
        building_map: 2D uint8 or bool array (H, W) where non-zero indicates building footprint.
        damage_map: 2D uint8 array (H, W) with damage class IDs 0..4.
        analysis_id: Identifier for the current analysis run.
        min_area: Minimum pixel area to consider as a valid building (filters speckle noise).
                  Default 25 px (~6.25 m² at 0.5m GSD).
        max_area: Optional maximum pixel area (filters large merged artifacts if specified).
        connectivity: Connectivity for connected component labeling (1 for 4-way, 2 for 8-way).

    Returns:
        Structured dictionary matching the required CrisisVision building schema:
        {
            "analysis_id": str,
            "coordinate_system": "image_pixel",
            "image_width": int,
            "image_height": int,
            "total_buildings": int,
            "damage_summary": Dict[str, int],
            "buildings": List[Dict[str, Any]],
        }
    """
    if building_map.ndim != 2 or damage_map.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got building_map ndim={building_map.ndim}, "
            f"damage_map ndim={damage_map.ndim}"
        )

    if building_map.shape != damage_map.shape:
        raise ValueError(
            f"Shape mismatch: building_map {building_map.shape} vs damage_map {damage_map.shape}"
        )

    height, width = building_map.shape

    # Handle completely empty building mask gracefully
    if not np.any(building_map > 0):
        logger.info("Building mask is completely empty. Returning 0 buildings.")
        empty_result: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "coordinate_system": "image_pixel",
            "image_width": int(width),
            "image_height": int(height),
            "total_buildings": 0,
            "damage_summary": {
                "no_damage": 0,
                "minor_damage": 0,
                "major_damage": 0,
                "destroyed": 0,
            },
            "buildings": [],
        }
        validate_building_data(empty_result)
        return empty_result

    # Label connected components
    labeled_mask, num_features = label(
        building_map > 0,
        connectivity=connectivity,
        return_num=True,
    )

    regions = regionprops(labeled_mask)

    # Filter regions by area and sort spatially (top-to-bottom, left-to-right)
    # Sorting ensures consistent and intuitive building ID numbering across runs
    valid_regions = [
        r for r in regions
        if r.area >= min_area and (max_area is None or r.area <= max_area)
    ]
    valid_regions.sort(key=lambda r: (r.centroid[0], r.centroid[1]))

    buildings: List[Dict[str, Any]] = []

    for idx, prop in enumerate(valid_regions, start=1):
        building_id = f"B{idx:03d}"

        # Centroid: skimage returns (row, col) -> (y, x)
        cy, cx = prop.centroid
        x_coord = round(float(cx), 2)
        y_coord = round(float(cy), 2)

        # Clamp centroid within image bounds
        x_coord = max(0.0, min(float(width - 1), x_coord))
        y_coord = max(0.0, min(float(height - 1), y_coord))

        # Bounding box: skimage returns (min_row, min_col, max_row, max_col)
        min_r, min_c, max_r, max_c = prop.bbox
        x_min = int(min_c)
        y_min = int(min_r)
        # Convert half-open [min, max) to inclusive [min, max_inclusive]
        x_max = max(x_min, int(max_c - 1))
        y_max = max(y_min, int(max_r - 1))

        # Clamp bounding box within image boundaries
        x_min = max(0, min(width - 1, x_min))
        y_min = max(0, min(height - 1, y_min))
        x_max = max(x_min, min(width - 1, x_max))
        y_max = max(y_min, min(height - 1, y_max))

        area_pixels = int(prop.area)

        # Extract damage pixels within building footprint
        comp_mask = (labeled_mask == prop.label)
        comp_damages = damage_map[comp_mask]

        # Exclude background pixels (class 0) from the building's damage vote
        valid_damages = comp_damages[comp_damages > 0]

        if len(valid_damages) > 0:
            counts = {c: int(np.sum(valid_damages == c)) for c in [1, 2, 3, 4]}
            # Majority voting: dominant class with highest count; break ties by higher severity
            dominant_class = max([1, 2, 3, 4], key=lambda c: (counts[c], c))
            total_valid = len(valid_damages)
            damage_distribution = {
                DAMAGE_LABELS[c]: round(float(counts[c] / total_valid), 4)
                for c in [1, 2, 3, 4]
            }
            # Confidence represents the spatial agreement (purity) of the dominant class
            confidence = round(float(counts[dominant_class] / total_valid), 4)
        else:
            # All pixels in footprint were 0 (background in damage head)
            # Default to no_damage with zero confidence
            dominant_class = 1
            damage_distribution = {
                "no_damage": 1.0,
                "minor_damage": 0.0,
                "major_damage": 0.0,
                "destroyed": 0.0,
            }
            confidence = 0.0

        building_entry: Dict[str, Any] = {
            "building_id": building_id,
            "centroid": {
                "x": x_coord,
                "y": y_coord,
            },
            "bounding_box": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
            "area_pixels": area_pixels,
            "damage_class_id": int(dominant_class),
            "damage_label": DAMAGE_LABELS[dominant_class],
            "confidence": confidence,
            "damage_distribution": damage_distribution,
        }
        buildings.append(building_entry)

    damage_summary = {
        "no_damage": sum(1 for b in buildings if b["damage_label"] == "no_damage"),
        "minor_damage": sum(1 for b in buildings if b["damage_label"] == "minor_damage"),
        "major_damage": sum(1 for b in buildings if b["damage_label"] == "major_damage"),
        "destroyed": sum(1 for b in buildings if b["damage_label"] == "destroyed"),
    }

    result = {
        "analysis_id": str(analysis_id),
        "coordinate_system": "image_pixel",
        "image_width": int(width),
        "image_height": int(height),
        "total_buildings": len(buildings),
        "damage_summary": damage_summary,
        "buildings": buildings,
    }

    validate_building_data(result)
    logger.info(
        "Extracted %d buildings from %dx%d image. Summary: %s",
        len(buildings),
        width,
        height,
        damage_summary,
    )
    return result


def validate_building_data(data: Dict[str, Any]) -> None:
    """Validate building extraction data against the crisis vision schema.

    Raises:
        BuildingValidationError: If any required key is missing or invalid.
    """
    required_root_keys = [
        "analysis_id",
        "coordinate_system",
        "image_width",
        "image_height",
        "buildings",
    ]
    for key in required_root_keys:
        if key not in data:
            raise BuildingValidationError(f"Missing required root key: '{key}'")

    if data["coordinate_system"] != "image_pixel":
        raise BuildingValidationError(
            f"Invalid coordinate_system: '{data['coordinate_system']}'. Expected 'image_pixel'."
        )

    width = data["image_width"]
    height = data["image_height"]

    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise BuildingValidationError(f"Invalid dimensions: width={width}, height={height}")

    if not isinstance(data["buildings"], list):
        raise BuildingValidationError("Key 'buildings' must be a list.")

    for b in data["buildings"]:
        bid = b.get("building_id")
        if not isinstance(bid, str) or not bid:
            raise BuildingValidationError(f"Invalid building_id: {bid}")

        centroid = b.get("centroid")
        if not isinstance(centroid, dict) or "x" not in centroid or "y" not in centroid:
            raise BuildingValidationError(f"Building {bid} missing valid centroid")

        cx = centroid["x"]
        cy = centroid["y"]
        if not (0 <= cx < width) or not (0 <= cy < height):
            raise BuildingValidationError(
                f"Building {bid} centroid ({cx}, {cy}) outside image [0, {width}) x [0, {height})"
            )

        bbox = b.get("bounding_box")
        if not isinstance(bbox, dict):
            raise BuildingValidationError(f"Building {bid} missing bounding_box dict")

        for k in ["x_min", "y_min", "x_max", "y_max"]:
            if k not in bbox:
                raise BuildingValidationError(f"Building {bid} bounding_box missing '{k}'")

        if not (0 <= bbox["x_min"] <= bbox["x_max"] < width):
            raise BuildingValidationError(
                f"Building {bid} x bounding box [{bbox['x_min']}, {bbox['x_max']}] invalid for width {width}"
            )
        if not (0 <= bbox["y_min"] <= bbox["y_max"] < height):
            raise BuildingValidationError(
                f"Building {bid} y bounding box [{bbox['y_min']}, {bbox['y_max']}] invalid for height {height}"
            )

        if not isinstance(b.get("area_pixels"), int) or b["area_pixels"] <= 0:
            raise BuildingValidationError(f"Building {bid} invalid area_pixels: {b.get('area_pixels')}")

        if b.get("damage_class_id") not in {1, 2, 3, 4}:
            raise BuildingValidationError(f"Building {bid} invalid damage_class_id: {b.get('damage_class_id')}")

        if b.get("damage_label") not in DAMAGE_LABELS.values():
            raise BuildingValidationError(f"Building {bid} invalid damage_label: {b.get('damage_label')}")

        dist = b.get("damage_distribution")
        if not isinstance(dist, dict):
            raise BuildingValidationError(f"Building {bid} missing damage_distribution")
        for label_name in DAMAGE_LABELS.values():
            if label_name not in dist:
                raise BuildingValidationError(
                    f"Building {bid} damage_distribution missing '{label_name}'"
                )


def save_annotated_buildings_image(
    base_image_path: str,
    buildings_data: Dict[str, Any],
    output_path: str,
    draw_bboxes: bool = True,
    draw_centroids: bool = True,
    draw_labels: bool = True,
) -> str:
    """Render an annotated satellite image with building bounding boxes, centroids, and IDs.

    Args:
        base_image_path: Path to post-disaster image or heatmap overlay.
        buildings_data: Structured dictionary returned by extract_buildings().
        output_path: Destination path for the saved image.
        draw_bboxes: Whether to draw bounding box rectangles.
        draw_centroids: Whether to draw centroid markers.
        draw_labels: Whether to annotate building IDs.

    Returns:
        output_path string.
    """
    base_img = Image.open(base_image_path).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for b in buildings_data.get("buildings", []):
        cid = b["damage_class_id"]
        color = DAMAGE_RGB_COLORS.get(cid, (255, 255, 255))
        stroke_color = color + (230,)
        fill_box = color + (40,)

        bbox = b["bounding_box"]
        x0, y0, x1, y1 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        if draw_bboxes:
            # Semi-transparent rectangle fill with solid border
            draw.rectangle([x0, y0, x1, y1], outline=stroke_color, width=2, fill=fill_box)

        cx, cy = b["centroid"]["x"], b["centroid"]["y"]
        if draw_centroids:
            r = 3
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))

        if draw_labels:
            bid = b["building_id"]
            # Position text slightly above or inside bbox
            tx = max(0, x0)
            ty = max(0, y0 - 12)
            draw.rectangle([tx, ty, tx + len(bid) * 7 + 4, ty + 11], fill=(15, 23, 42, 220))
            draw.text((tx + 2, ty), bid, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(base_img, overlay).convert("RGB")
    combined.save(output_path)
    logger.info("Saved annotated buildings image to %s", output_path)
    return output_path

