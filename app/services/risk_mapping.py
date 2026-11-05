"""
CrisisVision Pixel-Based Damage Hotspot Mapping Service.

Aggregates individual building damage severities into a continuous spatial
hotspot intensity map using distance-weighted Gaussian spatial aggregation,
identifies priority damage clusters through connected-component analysis, and
generates standardized, explainable risk assessment JSON.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

# Severity mapping based on verified xBD damage classes:
# Class 0: Background (excluded)
# Class 1: No damage -> Severity 0 -> 0.0 / 3 = 0.0
# Class 2: Minor damage -> Severity 1 -> 1.0 / 3 ≈ 0.3333
# Class 3: Major damage -> Severity 2 -> 2.0 / 3 ≈ 0.6667
# Class 4: Destroyed -> Severity 3 -> 3.0 / 3 = 1.0
DAMAGE_SEVERITY_WEIGHTS: Dict[int, float] = {
    0: 0.0,
    1: 0.0,
    2: 1.0 / 3.0,
    3: 2.0 / 3.0,
    4: 1.0,
}

LABEL_TO_CLASS_ID: Dict[str, int] = {
    "background": 0,
    "no_damage": 1,
    "minor_damage": 2,
    "major_damage": 3,
    "destroyed": 4,
}

# Provisional priority classification thresholds for disaster inspection triage.
# Note: These are provisional heuristic guidelines to assist first responders,
# NOT validated structural evacuation thresholds.
DEFAULT_PRIORITY_THRESHOLDS: Dict[str, float] = {
    "critical_priority": 0.75,
    "high_priority": 0.50,
    "moderate_priority": 0.25,
    "low_priority": 0.0,
}

PRIORITY_DISPLAY_NAMES: Dict[str, str] = {
    "critical_priority": "Critical Priority",
    "high_priority": "High Priority",
    "moderate_priority": "Moderate Priority",
    "low_priority": "Low Priority",
}

PRIORITY_COLORS: Dict[str, str] = {
    "critical_priority": "#dc2626",   # Deep Red
    "high_priority": "#ea580c",       # Orange Red
    "moderate_priority": "#eab308",   # Amber / Yellow
    "low_priority": "#3b82f6",        # Blue
}

STANDARD_LIMITATIONS: List[str] = [
    "Coordinates are image pixels, not geographic coordinates",
    "Damage predictions may be incorrect",
    "No population or infrastructure data is available",
    "This analysis does not determine evacuation safety",
]


class RiskMappingValidationError(ValueError):
    """Raised when generated risk mapping data violates the schema."""


def calculate_building_severity(building: Dict[str, Any]) -> float:
    """Calculate the normalized damage severity score S_i for a building.

    The score is normalized into the range [0.0, 1.0]:
      - No damage: 0.0
      - Minor damage: 1/3 (~0.3333)
      - Major damage: 2/3 (~0.6667)
      - Destroyed: 1.0

    Args:
        building: Building entry dictionary from building extraction JSON.

    Returns:
        Float in [0.0, 1.0].
    """
    if "damage_class_id" in building:
        cid = int(building["damage_class_id"])
        return DAMAGE_SEVERITY_WEIGHTS.get(cid, 0.0)

    label_str = building.get("damage_label", "").lower()
    cid = LABEL_TO_CLASS_ID.get(label_str, 1)
    return DAMAGE_SEVERITY_WEIGHTS.get(cid, 0.0)


def generate_damage_heatmap(
    buildings_data: Dict[str, Any],
    sigma_pixels: float = 50.0,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Generate continuous distance-weighted Gaussian damage severity heatmap.

    Formula:
        H(x, y) = sum_i S_i * exp(-d_i^2 / (2 * sigma^2))
    where S_i is the normalized severity of building i and d_i is Euclidean
    pixel distance to location (x, y).

    To ensure high computational efficiency on large images, contributions are
    accumulated locally within a 3*sigma window for each building rather than
    evaluating the full image grid. No-damage buildings (severity = 0) are
    skipped, ensuring they do not increase hotspot scores.

    Args:
        buildings_data: Dictionary containing 'buildings', 'image_width', 'image_height'.
        sigma_pixels: Spatial spread parameter in image pixels (default 50.0 px).
        image_shape: Optional (height, width) override.

    Returns:
        2D float32 array (H, W) with continuous aggregated damage intensity values.
    """
    if image_shape is not None:
        height, width = image_shape
    else:
        width = int(buildings_data.get("image_width", 1024))
        height = int(buildings_data.get("image_height", 1024))

    heatmap = np.zeros((height, width), dtype=np.float32)
    buildings = buildings_data.get("buildings", [])

    if not buildings or sigma_pixels <= 0:
        return heatmap

    radius = int(np.ceil(3.0 * sigma_pixels))
    two_sigma_sq = 2.0 * (sigma_pixels ** 2)

    for b in buildings:
        severity = calculate_building_severity(b)
        if severity <= 0.0:
            # No-damage buildings contribute 0 severity
            continue

        centroid = b.get("centroid", {})
        cx = float(centroid.get("x", 0.0))
        cy = float(centroid.get("y", 0.0))

        # Local window bounding coordinates clamped to image borders
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(width, int(np.ceil(cx + radius + 1)))
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(height, int(np.ceil(cy + radius + 1)))

        if x1 <= x0 or y1 <= y0:
            continue

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        kernel_patch = severity * np.exp(-dist_sq / two_sigma_sq)

        heatmap[y0:y1, x0:x1] += kernel_patch.astype(np.float32)

    return heatmap


def calculate_priority_score(
    associated_buildings: List[Dict[str, Any]],
    peak_intensity: float,
) -> Tuple[float, str]:
    """Calculate an explainable priority score and assign a category.

    Formula combines:
      - Average building damage severity (weight 0.40)
      - Damaged building concentration / density (weight 0.35)
      - Peak spatial kernel intensity (weight 0.25)

    Returns:
        Tuple of (priority_score in [0.0, 1.0], category string).
    """
    if not associated_buildings:
        score = min(1.0, float(peak_intensity) / 2.0) if peak_intensity > 0 else 0.0
        score = round(float(score), 4)
        cat = assign_priority_category(score)
        return score, cat

    severities = [calculate_building_severity(b) for b in associated_buildings]
    avg_severity = float(np.mean(severities)) if severities else 0.0
    max_sev = float(np.max(severities)) if severities else 0.0

    n_destroyed = sum(1 for b in associated_buildings if b.get("damage_label") == "destroyed")
    n_major = sum(1 for b in associated_buildings if b.get("damage_label") == "major_damage")
    n_minor = sum(1 for b in associated_buildings if b.get("damage_label") == "minor_damage")

    # Weighted concentration index
    weighted_damage = n_destroyed * 1.0 + n_major * 0.75 + n_minor * 0.25
    concentration_factor = min(1.0, weighted_damage / 4.0)

    # Intensity factor
    intensity_factor = min(1.0, float(peak_intensity) / 2.5)

    composite_score = (
        0.40 * avg_severity +
        0.35 * concentration_factor +
        0.25 * intensity_factor
    )
    priority_score = round(float(np.clip(composite_score, 0.0, 1.0)), 4)
    priority_category = assign_priority_category(priority_score)

    return priority_score, priority_category


def assign_priority_category(score: float) -> str:
    """Map numeric priority score to discrete category label."""
    if score >= DEFAULT_PRIORITY_THRESHOLDS["critical_priority"]:
        return "critical_priority"
    if score >= DEFAULT_PRIORITY_THRESHOLDS["high_priority"]:
        return "high_priority"
    if score >= DEFAULT_PRIORITY_THRESHOLDS["moderate_priority"]:
        return "moderate_priority"
    return "low_priority"


def extract_hotspots(
    buildings_data: Dict[str, Any],
    sigma_pixels: float = 50.0,
    threshold: float = 0.5,
    min_area_pixels: int = 50,
) -> Dict[str, Any]:
    """Identify and rank damage hotspot clusters from building extraction output.

    Pipeline:
      1. Compute continuous spatial damage heatmap using Gaussian kernel aggregation.
      2. Threshold at `threshold` to generate binary hotspot candidate regions.
      3. Segment contiguous regions using connected component analysis (8-connectivity).
      4. Filter out regions with pixel area < `min_area_pixels`.
      5. Associate damaged buildings with each hotspot region (unambiguous centroid-in-mask,
         with nearest-hotspot assignment for peripheral damaged buildings within sigma).
      6. Compute centroid, bounding box, area, damage counts, and priority scores.
      7. Return standardized, validated risk mapping JSON.

    Args:
        buildings_data: Validated building extraction JSON dictionary.
        sigma_pixels: Gaussian kernel spread in image pixels.
        threshold: Heatmap intensity threshold for defining hotspot boundaries.
        min_area_pixels: Minimum contiguous pixel area required for a hotspot.

    Returns:
        Structured risk mapping JSON dictionary.
    """
    analysis_id = str(buildings_data.get("analysis_id", "incident_unknown"))
    width = int(buildings_data.get("image_width", 1024))
    height = int(buildings_data.get("image_height", 1024))
    buildings = buildings_data.get("buildings", [])

    # Step 1: Generate damage heatmap
    heatmap = generate_damage_heatmap(
        buildings_data,
        sigma_pixels=sigma_pixels,
        image_shape=(height, width),
    )

    # Step 2 & 3: Binary thresholding & connected component labeling
    binary_mask = heatmap >= threshold

    if not np.any(binary_mask):
        logger.info("No regions exceeded threshold %.2f. 0 hotspots identified.", threshold)
        empty_result: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "coordinate_system": "image_pixel",
            "image_width": width,
            "image_height": height,
            "risk_mapping_method": "gaussian_damage_severity",
            "parameters": {
                "sigma_pixels": float(sigma_pixels),
                "threshold": float(threshold),
                "min_area_pixels": int(min_area_pixels),
            },
            "summary": {
                "total_hotspots": 0,
                "critical_priority_hotspots": 0,
                "high_priority_hotspots": 0,
                "moderate_priority_hotspots": 0,
                "low_priority_hotspots": 0,
                "minor_damage_buildings": 0,
                "major_damage_buildings": 0,
                "destroyed_buildings": 0,
            },
            "hotspots": [],
            "limitations": list(STANDARD_LIMITATIONS),
        }
        validate_risk_data(empty_result)
        return empty_result

    labeled_mask, num_features = label(binary_mask, connectivity=2, return_num=True)
    regions = regionprops(labeled_mask, intensity_image=heatmap)

    valid_regions = [r for r in regions if r.area >= min_area_pixels]
    # Sort regions spatially top-to-bottom, left-to-right
    valid_regions.sort(key=lambda r: (r.centroid[0], r.centroid[1]))

    # Pre-filter damaged buildings
    damaged_buildings = [
        b for b in buildings
        if calculate_building_severity(b) > 0.0
    ]

    # Map each damaged building to at most one hotspot
    assigned_building_ids = set()
    hotspot_building_map: Dict[int, List[Dict[str, Any]]] = {
        r.label: [] for r in valid_regions
    }

    # Pass 1: Direct centroid-in-mask membership
    for b in damaged_buildings:
        bid = b.get("building_id")
        centroid = b.get("centroid", {})
        cx = int(round(float(centroid.get("x", 0.0))))
        cy = int(round(float(centroid.get("y", 0.0))))

        if 0 <= cx < width and 0 <= cy < height:
            comp_label = int(labeled_mask[cy, cx])
            if comp_label in hotspot_building_map:
                hotspot_building_map[comp_label].append(b)
                assigned_building_ids.add(bid)

    # Pass 2: Associate unassigned damaged buildings within sigma radius of the nearest hotspot centroid
    for b in damaged_buildings:
        bid = b.get("building_id")
        if bid in assigned_building_ids:
            continue

        centroid = b.get("centroid", {})
        cx = float(centroid.get("x", 0.0))
        cy = float(centroid.get("y", 0.0))

        # Find closest hotspot centroid
        best_dist = float("inf")
        best_label = None

        for r in valid_regions:
            h_cy, h_cx = r.centroid
            dist = np.hypot(cx - h_cx, cy - h_cy)
            if dist < best_dist and dist <= (sigma_pixels * 1.5):
                best_dist = dist
                best_label = r.label

        if best_label is not None:
            hotspot_building_map[best_label].append(b)
            assigned_building_ids.add(bid)

    hotspots: List[Dict[str, Any]] = []

    for idx, prop in enumerate(valid_regions, start=1):
        hotspot_id = f"H{idx:03d}"

        # Centroid: skimage returns (row, col) -> (y, x)
        cy, cx = prop.centroid
        x_coord = max(0.0, min(float(width - 1), round(float(cx), 2)))
        y_coord = max(0.0, min(float(height - 1), round(float(cy), 2)))

        # Bounding box
        min_r, min_c, max_r, max_c = prop.bbox
        x_min = max(0, min(width - 1, int(min_c)))
        y_min = max(0, min(height - 1, int(min_r)))
        x_max = max(x_min, min(width - 1, int(max_c - 1)))
        y_max = max(y_min, min(height - 1, int(max_r - 1)))

        associated = hotspot_building_map.get(prop.label, [])
        associated.sort(key=lambda b: b.get("building_id", ""))

        n_buildings = len(associated)
        n_minor = sum(1 for b in associated if b.get("damage_label") == "minor_damage")
        n_major = sum(1 for b in associated if b.get("damage_label") == "major_damage")
        n_destroyed = sum(1 for b in associated if b.get("damage_label") == "destroyed")

        b_ids = [b.get("building_id") for b in associated if "building_id" in b]

        severities = [calculate_building_severity(b) for b in associated]
        avg_sev = round(float(np.mean(severities)), 4) if severities else 0.0
        max_sev = round(float(np.max(severities)), 4) if severities else 0.0

        peak_intensity = (
            float(prop.intensity_max) if hasattr(prop, "intensity_max")
            else float(getattr(prop, "max_intensity", 1.0))
        )

        p_score, p_category = calculate_priority_score(associated, peak_intensity)

        hotspot_entry: Dict[str, Any] = {
            "hotspot_id": hotspot_id,
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
            "pixel_area": int(prop.area),
            "building_count": int(n_buildings),
            "minor_damage_count": int(n_minor),
            "major_damage_count": int(n_major),
            "destroyed_count": int(n_destroyed),
            "average_severity": avg_sev,
            "max_severity": max_sev,
            "priority_score": p_score,
            "priority_category": p_category,
            "building_ids": b_ids,
        }
        hotspots.append(hotspot_entry)

    summary = {
        "total_hotspots": len(hotspots),
        "critical_priority_hotspots": sum(1 for h in hotspots if h["priority_category"] == "critical_priority"),
        "high_priority_hotspots": sum(1 for h in hotspots if h["priority_category"] == "high_priority"),
        "moderate_priority_hotspots": sum(1 for h in hotspots if h["priority_category"] == "moderate_priority"),
        "low_priority_hotspots": sum(1 for h in hotspots if h["priority_category"] == "low_priority"),
        "minor_damage_buildings": sum(h["minor_damage_count"] for h in hotspots),
        "major_damage_buildings": sum(h["major_damage_count"] for h in hotspots),
        "destroyed_buildings": sum(h["destroyed_count"] for h in hotspots),
    }

    result = {
        "analysis_id": analysis_id,
        "coordinate_system": "image_pixel",
        "image_width": int(width),
        "image_height": int(height),
        "risk_mapping_method": "gaussian_damage_severity",
        "parameters": {
            "sigma_pixels": float(sigma_pixels),
            "threshold": float(threshold),
            "min_area_pixels": int(min_area_pixels),
        },
        "summary": summary,
        "hotspots": hotspots,
        "limitations": list(STANDARD_LIMITATIONS),
    }

    validate_risk_data(result)
    logger.info("Hotspot extraction complete: %d hotspots generated.", len(hotspots))
    return result


def validate_risk_data(data: Dict[str, Any]) -> None:
    """Validate risk mapping data against schema constraints."""
    required_root_keys = [
        "analysis_id",
        "coordinate_system",
        "image_width",
        "image_height",
        "risk_mapping_method",
        "parameters",
        "summary",
        "hotspots",
        "limitations",
    ]
    for k in required_root_keys:
        if k not in data:
            raise RiskMappingValidationError(f"Missing root key: '{k}'")

    if data["coordinate_system"] != "image_pixel":
        raise RiskMappingValidationError(
            f"Invalid coordinate_system '{data['coordinate_system']}'. Expected 'image_pixel'."
        )

    width = data["image_width"]
    height = data["image_height"]
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise RiskMappingValidationError(f"Invalid dimensions: width={width}, height={height}")

    if not isinstance(data["hotspots"], list):
        raise RiskMappingValidationError("Key 'hotspots' must be a list.")

    for h in data["hotspots"]:
        hid = h.get("hotspot_id")
        if not isinstance(hid, str) or not hid:
            raise RiskMappingValidationError(f"Invalid hotspot_id: {hid}")

        centroid = h.get("centroid")
        if not isinstance(centroid, dict) or "x" not in centroid or "y" not in centroid:
            raise RiskMappingValidationError(f"Hotspot {hid} missing valid centroid")

        cx = centroid["x"]
        cy = centroid["y"]
        if not (0 <= cx < width) or not (0 <= cy < height):
            raise RiskMappingValidationError(
                f"Hotspot {hid} centroid ({cx}, {cy}) outside image [0, {width}) x [0, {height})"
            )

        bbox = h.get("bounding_box")
        if not isinstance(bbox, dict):
            raise RiskMappingValidationError(f"Hotspot {hid} missing bounding_box dict")

        for k in ["x_min", "y_min", "x_max", "y_max"]:
            if k not in bbox:
                raise RiskMappingValidationError(f"Hotspot {hid} bounding_box missing '{k}'")

        if not (0 <= bbox["x_min"] <= bbox["x_max"] < width):
            raise RiskMappingValidationError(
                f"Hotspot {hid} x bbox [{bbox['x_min']}, {bbox['x_max']}] invalid for width {width}"
            )
        if not (0 <= bbox["y_min"] <= bbox["y_max"] < height):
            raise RiskMappingValidationError(
                f"Hotspot {hid} y bbox [{bbox['y_min']}, {bbox['y_max']}] invalid for height {height}"
            )

        cat = h.get("priority_category")
        if cat not in DEFAULT_PRIORITY_THRESHOLDS:
            raise RiskMappingValidationError(f"Hotspot {hid} invalid priority_category: {cat}")

        score = h.get("priority_score")
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            raise RiskMappingValidationError(f"Hotspot {hid} invalid priority_score: {score}")

        if not isinstance(h.get("building_ids"), list):
            raise RiskMappingValidationError(f"Hotspot {hid} building_ids must be a list")


def save_risk_map_image(
    base_image_path: str,
    heatmap: np.ndarray,
    risk_data: Dict[str, Any],
    output_path: str,
    threshold: float = 0.5,
) -> str:
    """Render a transparent color-mapped hotspot risk overlay on top of base image.

    Color ramp:
      - Below 0.5 * threshold: completely transparent (base image unaffected)
      - Transition (0.5 * threshold .. threshold): faint amber glow
      - Moderate (threshold .. 1.2): bright yellow-orange
      - High (1.2 .. 2.0): deep orange-red
      - Critical (>= 2.0): intense crimson

    Also annotates bounding box contours and Hotspot badges for each identified hotspot.

    Args:
        base_image_path: Path to post-disaster satellite image.
        heatmap: 2D float32 continuous intensity array matching base image resolution.
        risk_data: Validated risk mapping JSON.
        output_path: Destination path for saved overlay image.
        threshold: Hotspot activation threshold.

    Returns:
        output_path string.
    """
    base = Image.open(base_image_path).convert("RGBA")
    width, height = base.size

    # Resize heatmap if shapes differ
    if heatmap.shape != (height, width):
        heat_pil = Image.fromarray(heatmap, mode="F").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        heatmap = np.asarray(heat_pil, dtype=np.float32)

    # Normalize intensity for visualization ramp
    max_val = max(1.5, float(heatmap.max()))
    norm = np.clip(heatmap / max_val, 0.0, 1.0)

    # Vectorized RGBA color generation: Smooth Yellow -> Orange -> Crimson ramp
    r = np.clip(255 * (norm * 1.6), 0, 255).astype(np.uint8)
    g = np.clip(255 * (1.1 - np.abs(norm - 0.45) * 2.1), 0, 255).astype(np.uint8)
    b = np.clip(255 * (0.15 * norm), 0, 255).astype(np.uint8)

    # Alpha: transparent below 0.4 * threshold, gently ramping up to ~0.65 opacity
    alpha_ramp = np.clip((heatmap - (0.4 * threshold)) / (max_val - 0.4 * threshold + 1e-6), 0.0, 1.0)
    alpha = (alpha_ramp * 165).astype(np.uint8)

    rgba_overlay = np.stack([r, g, b, alpha], axis=-1)
    heat_overlay_img = Image.fromarray(rgba_overlay, mode="RGBA")

    # Composite heatmap over base image
    composite = Image.alpha_composite(base, heat_overlay_img)
    draw = ImageDraw.Draw(composite)

    # Annotate hotspot regions
    for h in risk_data.get("hotspots", []):
        cat = h.get("priority_category", "moderate_priority")
        stroke_color = (234, 88, 12, 240) if cat in ["high_priority", "critical_priority"] else (234, 179, 8, 220)
        fill_badge = (15, 23, 42, 230)

        bbox = h["bounding_box"]
        x0, y0, x1, y1 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        # Hotspot bounding box contour
        draw.rectangle([x0, y0, x1, y1], outline=stroke_color, width=2)

        # Centroid crosshair
        cx, cy = h["centroid"]["x"], h["centroid"]["y"]
        r_c = 4
        draw.ellipse([cx - r_c, cy - r_c, cx + r_c, cy + r_c], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))

        # Hotspot badge label
        label_text = f"{h['hotspot_id']} [{h['priority_category'].replace('_', ' ').title()}]"
        tx = max(0, x0)
        ty = max(0, y0 - 16)
        text_w = len(label_text) * 7 + 8
        draw.rectangle([tx, ty, tx + text_w, ty + 14], fill=fill_badge)
        draw.text((tx + 4, ty + 1), label_text, fill=(255, 255, 255, 255))

    out_rgb = composite.convert("RGB")
    out_rgb.save(output_path)
    logger.info("Saved risk map overlay image to %s", output_path)
    return output_path
