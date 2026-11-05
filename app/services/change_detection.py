
import logging
import torch
import numpy as np
from PIL import Image
from torchange.models.changeos import ChangeOS


logger = logging.getLogger(__name__)

MODEL_ID = "EVER-Z/torchange_example_changeos_swint_on_xview2_best42k"

# The model was trained with 512×512 tiles at stride 256 (xView2 tiles are from 1024×1024).
# Input must be 512×512 for the head's fixed upsample_scale=4 to produce 512×512 outputs.
IMAGE_SIZE = (512, 512)

# ImageNet normalisation used during training (see model README "Training recipe").
# The model backbone (Swin-T) is ImageNet-1k initialised and the training augmentation
# pipeline applies standard ImageNet mean/std after converting pixels to [0, 1].
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

_model = None

# xBD / xView2 damage class mapping
DAMAGE_CLASSES = {
    0: "Background",
    1: "No damage",
    2: "Minor damage",
    3: "Major damage",
    4: "Destroyed",
}

# Visualization colors for each damage class (RGB)
# Background is kept transparent in the overlay (alpha=0 for class 0).
DAMAGE_COLORS = np.array([
    [0,   0,   0  ],   # 0: Background  — transparent (overridden per pixel in save_heatmap)
    [0,   200, 0  ],   # 1: No damage   — green
    [255, 220, 0  ],   # 2: Minor       — yellow
    [255, 140, 0  ],   # 3: Major       — orange
    [220, 0,   0  ],   # 4: Destroyed   — red
], dtype=np.uint8)


def load_model():
    """Load the pretrained ChangeOS (Swin-T) model once and cache it.

    The checkpoint is downloaded from HuggingFace Hub on first call and cached
    locally.  Subsequent calls return the cached instance immediately.
    """
    global _model

    if _model is None:
        logger.info("Loading ChangeOS model from %s …", MODEL_ID)
        _model = ChangeOS.from_pretrained(MODEL_ID).eval()
        _model.to("cpu")
        logger.info("Model loaded successfully.")

    return _model


def prepare_image(image_path: str) -> torch.Tensor:
    """Load a single RGB image, resize to 512×512, and apply ImageNet normalisation.

    The model was trained with the standard torchvision ImageNet mean/std applied
    *after* scaling pixel values to [0, 1].  Skipping this step (i.e. only dividing
    by 255) produces activations far outside the distribution seen at training time
    and causes most buildings to be classified as background.

    Args:
        image_path: Path to any PIL-readable image file.

    Returns:
        Float32 tensor of shape (3, 512, 512) normalised to ImageNet statistics.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Scale to [0, 1], then apply ImageNet mean/std.
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)   # (3, H, W)
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD  # ImageNet normalisation
    return tensor


def predict_damage(pre_image_path: str, post_image_path: str) -> dict:
    """Run ChangeOS inference on a before/after image pair.

    Pipeline:
    1. Load and normalise both images to 512×512.
    2. Concatenate into a 6-channel tensor [pre_RGB | post_RGB].
    3. Run the model in eval/no-grad mode.
    4. Extract building mask from t1_semantic_prediction (sigmoid, already in [0,1]).
    5. Extract damage class from change_prediction (softmax probabilities, already in [0,1]).
       NOTE: The model head already applies softmax — do NOT apply softmax again.
    6. Apply the single-map constraint: set non-building pixels to class 0 (background).
    7. Upscale the prediction maps to the original post-disaster image dimensions.

    Args:
        pre_image_path:  Path to the pre-disaster image.
        post_image_path: Path to the post-disaster image.

    Returns:
        dict with keys:
          - damage_map (np.uint8, H_orig×W_orig): per-pixel damage class 0–4.
          - building_map (np.uint8, H_orig×W_orig): binary building footprint.
          - image_size (list[int]): [H_orig, W_orig] of the POST image.
    """
    model = load_model()

    # Determine original dimensions of the post-disaster image for final upscaling.
    with Image.open(post_image_path) as _img:
        orig_w, orig_h = _img.size   # PIL gives (width, height)

    before = prepare_image(pre_image_path)   # (3, 512, 512)
    after  = prepare_image(post_image_path)  # (3, 512, 512)

    # ChangeOS bitemporal convention: [pre_R,pre_G,pre_B, post_R,post_G,post_B] → (1, 6, 512, 512)
    images = torch.cat([before, after], dim=0).unsqueeze(0)

    logger.debug("Running ChangeOS inference on CPU …")
    with torch.inference_mode():
        output = model(images)

    # --- change_prediction: (B, 5, H, W) — softmax already applied by the model head ---
    # Do NOT call .softmax() again; the model returns probabilities, not logits.
    change_probs = output["change_prediction"][0]          # (5, 512, 512) probabilities
    damage_map = change_probs.argmax(dim=0).cpu().numpy().astype(np.uint8)  # (512, 512)

    # --- t1_semantic_prediction: (B, 1, H, W) — sigmoid already applied ---
    # Threshold at 0.5 to get a binary building mask.
    building_prob = output["t1_semantic_prediction"][0, 0].cpu().numpy()  # (512, 512)
    building_map = (building_prob > 0.5).astype(np.uint8)                 # (512, 512) binary

    # Apply single-map constraint (from the model README):
    # pixels not identified as buildings receive class 0 (background).
    damage_map[building_map == 0] = 0

    # Upscale both maps from 512×512 to the original image resolution.
    # Use nearest-neighbour interpolation to avoid blending discrete class IDs.
    if (orig_h, orig_w) != IMAGE_SIZE:
        damage_pil  = Image.fromarray(damage_map,  mode="L").resize(
            (orig_w, orig_h), Image.Resampling.NEAREST
        )
        building_pil = Image.fromarray(building_map, mode="L").resize(
            (orig_w, orig_h), Image.Resampling.NEAREST
        )
        damage_map   = np.asarray(damage_pil,   dtype=np.uint8)
        building_map = np.asarray(building_pil, dtype=np.uint8)

    logger.info(
        "Inference done. Building pixels: %d. Damage class distribution: %s",
        int(building_map.sum()),
        {DAMAGE_CLASSES[c]: int(n) for c, n in zip(*np.unique(damage_map, return_counts=True))},
    )

    return {
        "damage_map":   damage_map,
        "building_map": building_map,
        "image_size":   [orig_h, orig_w],
    }


def save_heatmap(damage_map: np.ndarray, output_path: str, base_image_path: str = None) -> str:
    """Render and save a spatially accurate damage overlay on the post-disaster image.

    The overlay is composited at the resolution of the base image; the damage_map
    is *not* resized here because predict_damage() already upscales it to the
    original image dimensions.

    Color coding (matches DAMAGE_COLORS):
      - Transparent : background / non-building pixels
      - Green       : no damage
      - Yellow      : minor damage
      - Orange      : major damage
      - Red         : destroyed

    Args:
        damage_map:      2-D uint8 array of class IDs (0–4), H×W.
        output_path:     Destination file path for the PNG overlay image.
        base_image_path: If provided, the damage map is composited over this image.

    Returns:
        output_path
    """
    class_map = np.asarray(damage_map, dtype=np.uint8)

    if base_image_path:
        base = Image.open(base_image_path).convert("RGB")
        base_w, base_h = base.size

        # The damage_map should already match the original image dimensions after
        # predict_damage() upscales it.  If for any reason the sizes differ (e.g.
        # the caller provides an independent damage_map), resize with NEAREST to
        # avoid introducing blurred class-ID artifacts.
        if (class_map.shape[0] != base_h) or (class_map.shape[1] != base_w):
            class_map = np.asarray(
                Image.fromarray(class_map, mode="L").resize(
                    (base_w, base_h), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )

        base_array = np.asarray(base, dtype=np.float32)
        colored    = DAMAGE_COLORS[class_map]            # (H, W, 3)

        # Composite: blend only building pixels (class > 0) at alpha=0.55.
        building_mask = class_map > 0
        result = base_array.copy()
        alpha = 0.55
        result[building_mask] = (
            (1.0 - alpha) * base_array[building_mask]
            + alpha * colored[building_mask].astype(np.float32)
        )

        out_array = np.clip(result, 0, 255).astype(np.uint8)
    else:
        # No base image — just colour-code the damage map directly.
        out_array = DAMAGE_COLORS[class_map]

    Image.fromarray(out_array).save(output_path)
    logger.info("Heatmap saved to %s", output_path)
    return output_path


def detect_cloud_mask(image_path: str) -> np.ndarray:
    """Estimate potential cloud pixels in an RGB image using a simple HSV heuristic.

    This is a basic heuristic (bright, low-saturation pixels) and is NOT used in
    the main inference pipeline.  It is retained as a utility for diagnostics.

    Args:
        image_path: Path to any PIL-readable image file.

    Returns:
        Boolean array of shape IMAGE_SIZE where True indicates likely cloud pixels.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    saturation = hsv[:, :, 1]
    brightness  = hsv[:, :, 2]

    # Bright pixels with low saturation are likely clouds or bright bare ground.
    cloud_mask = (brightness > 180) & (saturation < 80)

    return cloud_mask
