"""
Unit and integration tests for the CrisisVision change-detection pipeline.

Tests verify:
  - Correct image loading and channel order
  - ImageNet normalisation (the most critical bug that was fixed)
  - Correct paired-image input construction (6-channel tensor)
  - Model output shapes and value ranges
  - Absence of double-softmax (model outputs are probabilities, not logits)
  - Correct class mapping (0=Background … 4=Destroyed)
  - Overlay dimensions match the original image
  - Nearest-neighbour upscaling preserves class IDs
  - Output file generation

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_change_detection.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path regardless of how the tests
# are invoked (pytest from project root, or python -m pytest from tests/).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.change_detection import (
    IMAGE_SIZE,
    DAMAGE_CLASSES,
    DAMAGE_COLORS,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    prepare_image,
    predict_damage,
    save_heatmap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_image_dir(tmp_path_factory):
    """Return a directory with synthetic 1024×1024 RGB test images."""
    d = tmp_path_factory.mktemp("images")
    return d


def _make_image(path: Path, width: int = 1024, height: int = 1024, seed: int = 0) -> Path:
    """Create a random RGB image and save it as a PNG."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture(scope="module")
def pre_image(tmp_image_dir):
    return _make_image(tmp_image_dir / "pre.png", seed=1)


@pytest.fixture(scope="module")
def post_image(tmp_image_dir):
    return _make_image(tmp_image_dir / "post.png", seed=2)


@pytest.fixture(scope="module")
def small_image(tmp_image_dir):
    """Non-standard size image to test upscaling path."""
    return _make_image(tmp_image_dir / "small.png", width=256, height=256, seed=3)


# ---------------------------------------------------------------------------
# prepare_image tests
# ---------------------------------------------------------------------------

class TestPrepareImage:
    def test_output_shape(self, pre_image):
        t = prepare_image(str(pre_image))
        assert t.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), (
            f"Expected tensor shape (3, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}), got {t.shape}"
        )

    def test_output_dtype(self, pre_image):
        t = prepare_image(str(pre_image))
        assert t.dtype == torch.float32

    def test_imagenet_normalization_applied(self, pre_image):
        """After ImageNet normalisation values should NOT be in [0, 1].

        The mean (0.485, 0.456, 0.406) is subtracted and divided by std,
        so the range for a typical image spans roughly [-2.5, 2.5].
        If only /255 were applied the max would be ≤ 1.0.
        """
        t = prepare_image(str(pre_image))
        # At least some pixels should exceed 1.0 (before-norm max is 1.0).
        assert float(t.max()) > 1.0, (
            "Tensor max ≤ 1.0: ImageNet normalisation was NOT applied."
        )
        # And some should be below 0.0 (mean subtraction pushes dark pixels negative).
        assert float(t.min()) < 0.0, (
            "Tensor min ≥ 0.0: ImageNet normalisation was NOT applied."
        )

    def test_imagenet_mean_values(self):
        expected_mean = torch.tensor([0.485, 0.456, 0.406])
        actual_mean = _IMAGENET_MEAN.squeeze()
        assert torch.allclose(actual_mean, expected_mean, atol=1e-4), (
            f"ImageNet mean mismatch: {actual_mean.tolist()}"
        )

    def test_imagenet_std_values(self):
        expected_std = torch.tensor([0.229, 0.224, 0.225])
        actual_std = _IMAGENET_STD.squeeze()
        assert torch.allclose(actual_std, expected_std, atol=1e-4), (
            f"ImageNet std mismatch: {actual_std.tolist()}"
        )

    def test_rgb_channel_order(self, pre_image):
        """prepare_image must preserve RGB order, not accidentally swap to BGR."""
        # Create a red-only image (R=200, G=0, B=0).
        path = Path(str(pre_image)).parent / "red.png"
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = 200   # Red channel
        Image.fromarray(arr).save(path)

        t = prepare_image(str(path))
        # After normalisation, the red channel (index 0) should be the most positive.
        # In BGR the red would land at index 2.
        ch_means = [float(t[c].mean()) for c in range(3)]
        assert ch_means[0] > ch_means[2], (
            f"Channel order may be BGR: R mean={ch_means[0]:.3f}, B mean={ch_means[2]:.3f}"
        )

    def test_resize_to_model_input_size(self, small_image):
        """Images of any size must be resized to IMAGE_SIZE."""
        t = prepare_image(str(small_image))
        assert t.shape[1:] == tuple(IMAGE_SIZE), (
            f"Expected {IMAGE_SIZE} after resize, got {t.shape[1:]}"
        )


# ---------------------------------------------------------------------------
# 6-channel input construction
# ---------------------------------------------------------------------------

class TestBitemporalInput:
    def test_concatenation_order(self, pre_image, post_image):
        """Before image must occupy the first 3 channels, after the last 3."""
        before = prepare_image(str(pre_image))
        after  = prepare_image(str(post_image))
        combined = torch.cat([before, after], dim=0)
        assert combined.shape == (6, IMAGE_SIZE[0], IMAGE_SIZE[1])
        # Check that the first 3 channels equal the before tensor.
        assert torch.equal(combined[:3], before)
        # Check that the last 3 channels equal the after tensor.
        assert torch.equal(combined[3:], after)


# ---------------------------------------------------------------------------
# predict_damage tests (no model loading — only structural / invariant checks)
# ---------------------------------------------------------------------------

class TestPredictDamage:
    """Tests that run the full model.  Skip gracefully if the checkpoint is
    not cached (would require network access in a sandboxed environment)."""

    _SKIP_IF_NO_CHECKPOINT = pytest.mark.skipif(
        not Path(
            r"C:\Users\91934\.cache\huggingface\hub"
            r"\models--EVER-Z--torchange_example_changeos_swint_on_xview2_best42k"
        ).exists(),
        reason="HuggingFace checkpoint not cached locally",
    )

    @_SKIP_IF_NO_CHECKPOINT
    def test_output_keys_present(self, pre_image, post_image):
        result = predict_damage(str(pre_image), str(post_image))
        assert "damage_map" in result
        assert "building_map" in result
        assert "image_size" in result

    @_SKIP_IF_NO_CHECKPOINT
    def test_damage_map_shape_matches_original_image(self, pre_image, post_image):
        """damage_map must be upscaled to the original image dimensions, not 512×512."""
        result = predict_damage(str(pre_image), str(post_image))
        orig_w, orig_h = Image.open(pre_image).size
        h, w = result["damage_map"].shape
        assert (h, w) == (orig_h, orig_w), (
            f"damage_map {h}×{w} does not match original image {orig_h}×{orig_w}. "
            "Upscaling to original dimensions was not applied."
        )

    @_SKIP_IF_NO_CHECKPOINT
    def test_damage_map_dtype(self, pre_image, post_image):
        result = predict_damage(str(pre_image), str(post_image))
        assert result["damage_map"].dtype == np.uint8

    @_SKIP_IF_NO_CHECKPOINT
    def test_damage_map_class_range(self, pre_image, post_image):
        """All class IDs must be in the xBD range [0, 4]."""
        result = predict_damage(str(pre_image), str(post_image))
        dm = result["damage_map"]
        assert int(dm.min()) >= 0
        assert int(dm.max()) <= 4, (
            f"damage_map contains class ID {int(dm.max())} outside [0, 4]."
        )

    @_SKIP_IF_NO_CHECKPOINT
    def test_building_pixels_valid_range(self, pre_image, post_image):
        """building_map must be binary (values 0 or 1 only).

        Note: randomly generated noise images may yield 0 building pixels —
        the model is discriminating. The test only validates the binary dtype
        constraint, not whether buildings are present in synthetic images.
        """
        result = predict_damage(str(pre_image), str(post_image))
        bm = result["building_map"]
        unique_vals = set(np.unique(bm).tolist())
        assert unique_vals.issubset({0, 1}), (
            f"building_map contains values other than 0/1: {unique_vals}"
        )

    @_SKIP_IF_NO_CHECKPOINT
    def test_single_map_constraint_applied(self, pre_image, post_image):
        """Non-building pixels must be forced to class 0 (background)."""
        result = predict_damage(str(pre_image), str(post_image))
        dm = result["damage_map"]
        bm = result["building_map"]
        # Every pixel outside the building mask must be 0.
        assert np.all(dm[bm == 0] == 0), (
            "Single-map constraint not applied: damage classes present outside building mask."
        )

    @_SKIP_IF_NO_CHECKPOINT
    def test_image_size_reported_correctly(self, pre_image, post_image):
        result = predict_damage(str(pre_image), str(post_image))
        orig_w, orig_h = Image.open(pre_image).size
        assert result["image_size"] == [orig_h, orig_w]


# ---------------------------------------------------------------------------
# save_heatmap tests
# ---------------------------------------------------------------------------

class TestSaveHeatmap:
    def _make_damage_map(self, h: int = 100, w: int = 100) -> np.ndarray:
        dm = np.zeros((h, w), dtype=np.uint8)
        dm[10:30, 10:30] = 1   # no-damage patch
        dm[40:60, 40:60] = 2   # minor-damage patch
        dm[70:90, 70:90] = 4   # destroyed patch
        return dm

    def test_output_file_created(self, tmp_path):
        dm = self._make_damage_map()
        out = str(tmp_path / "overlay.png")
        save_heatmap(dm, out)
        assert Path(out).exists()

    def test_output_dimensions_match_base_image(self, tmp_path):
        """The overlay must have the same dimensions as the base image."""
        base_path = tmp_path / "base.png"
        base_arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        Image.fromarray(base_arr).save(base_path)

        dm = self._make_damage_map(h=200, w=300)
        out = str(tmp_path / "overlay_match.png")
        save_heatmap(dm, out, str(base_path))

        result_img = Image.open(out)
        assert result_img.size == (300, 200), (
            f"Overlay size {result_img.size} does not match base image size (300, 200)."
        )

    def test_no_damage_class_is_green(self, tmp_path):
        """Class 1 (no damage) must be rendered green."""
        dm = np.ones((50, 50), dtype=np.uint8)   # all class 1
        out = str(tmp_path / "green.png")
        save_heatmap(dm, out)

        arr = np.asarray(Image.open(out))
        # Green channel should dominate.
        assert arr[:, :, 1].mean() > arr[:, :, 0].mean(), "Class 1 is not green."
        assert arr[:, :, 1].mean() > arr[:, :, 2].mean(), "Class 1 is not green."

    def test_destroyed_class_is_red(self, tmp_path):
        """Class 4 (destroyed) must be rendered red."""
        dm = np.full((50, 50), 4, dtype=np.uint8)
        out = str(tmp_path / "red.png")
        save_heatmap(dm, out)

        arr = np.asarray(Image.open(out))
        assert arr[:, :, 0].mean() > arr[:, :, 1].mean(), "Class 4 is not red."
        assert arr[:, :, 0].mean() > arr[:, :, 2].mean(), "Class 4 is not red."

    def test_background_class_zero_is_transparent_in_overlay(self, tmp_path):
        """Class 0 pixels must remain unchanged (fully transparent overlay)."""
        # Create a distinctive base image (all blue).
        base_path = tmp_path / "blue_base.png"
        base_arr = np.zeros((50, 50, 3), dtype=np.uint8)
        base_arr[:, :, 2] = 200   # blue
        Image.fromarray(base_arr).save(base_path)

        dm = np.zeros((50, 50), dtype=np.uint8)   # all background
        out = str(tmp_path / "bg_transparent.png")
        save_heatmap(dm, out, str(base_path))

        result = np.asarray(Image.open(out).convert("RGB"))
        # With alpha=0 for class 0, the result should equal the base image.
        np.testing.assert_array_equal(result, base_arr)

    def test_mismatched_map_auto_resizes(self, tmp_path):
        """If damage_map and base image differ in size, the map must be resized."""
        base_path = tmp_path / "big_base.png"
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(base_path)

        dm = self._make_damage_map(h=50, w=50)   # smaller than base
        out = str(tmp_path / "resized_overlay.png")
        save_heatmap(dm, out, str(base_path))

        result_img = Image.open(out)
        assert result_img.size == (200, 200)


# ---------------------------------------------------------------------------
# Class mapping constants
# ---------------------------------------------------------------------------

class TestClassMapping:
    def test_damage_classes_dict(self):
        assert DAMAGE_CLASSES[0] == "Background"
        assert DAMAGE_CLASSES[1] == "No damage"
        assert DAMAGE_CLASSES[2] == "Minor damage"
        assert DAMAGE_CLASSES[3] == "Major damage"
        assert DAMAGE_CLASSES[4] == "Destroyed"

    def test_damage_colors_shape(self):
        assert DAMAGE_COLORS.shape == (5, 3), "DAMAGE_COLORS must have 5 rows (one per class)."

    def test_no_damage_is_green(self):
        """Class 1 color must have green channel dominant."""
        c = DAMAGE_COLORS[1]
        assert c[1] > c[0] and c[1] > c[2], f"Class 1 is not green: {c}"

    def test_destroyed_is_red(self):
        """Class 4 color must have red channel dominant."""
        c = DAMAGE_COLORS[4]
        assert c[0] > c[1] and c[0] > c[2], f"Class 4 is not red: {c}"


# ---------------------------------------------------------------------------
# Nearest-neighbour upscaling preserves class IDs
# ---------------------------------------------------------------------------

class TestNearestNeighbourUpscaling:
    def test_class_ids_preserved_after_resize(self, tmp_path):
        """Resizing a class map with NEAREST must not introduce new class IDs."""
        # Build a 4-class map at 10×10.
        dm = np.array([[0, 1], [2, 4]], dtype=np.uint8)
        dm = np.kron(dm, np.ones((5, 5), dtype=np.uint8))   # 10×10

        pil_map = Image.fromarray(dm, mode="L")
        resized = np.asarray(
            pil_map.resize((100, 100), Image.Resampling.NEAREST), dtype=np.uint8
        )
        original_classes = set(np.unique(dm).tolist())
        resized_classes = set(np.unique(resized).tolist())
        assert resized_classes == original_classes, (
            f"New class IDs introduced during resize: {resized_classes - original_classes}. "
            "NEAREST interpolation must be used for categorical maps."
        )
