
import torch
import numpy as np
from PIL import Image
from torchange.models.changeos import ChangeOS


MODEL_ID = "EVER-Z/torchange_example_changeos_swint_on_xview2_best42k"
IMAGE_SIZE = (512, 512)

_model = None


def load_model():
    """Load the pretrained model once and reuse it."""
    global _model

    if _model is None:
        _model = ChangeOS.from_pretrained(MODEL_ID).eval()
        _model.to("cpu")

    return _model


def prepare_image(image_path):
    """Load an RGB satellite image and prepare its tensor."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    return tensor


def predict_damage(pre_image_path, post_image_path):
    """Predict building damage from pre- and post-disaster images."""
    model = load_model()

    before = prepare_image(pre_image_path)
    after = prepare_image(post_image_path)

    # ChangeOS expects the two RGB images as six channels.
    images = torch.cat([before, after], dim=0).unsqueeze(0)

    with torch.inference_mode():
        output = model(images)

    probabilities = output["change_prediction"][0].softmax(dim=0)
    damage_map = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)

    building_map = (
        output["t1_semantic_prediction"][0, 0] > 0.5
    ).cpu().numpy()

    # Restrict damage classes to pixels identified as buildings.
    damage_map[~building_map] = 0

    return {
        "damage_map": damage_map,
        "building_map": building_map.astype(np.uint8),
        "image_size": list(damage_map.shape),
    }



def save_heatmap(damage_map, output_path, base_image_path=None):
    """Save a presentation-ready damage overlay on the post-disaster image."""
    import numpy as np
    from PIL import Image

    # Damage class colors
    colors = np.array([
        [35, 35, 35],       # Class 0: Background
        [255, 220, 0],      # Class 1: No damage
        [255, 140, 0],      # Class 2: Minor damage
        [255, 50, 50],      # Class 3: Major damage
        [180, 0, 255],      # Class 4: Destroyed
    ], dtype=np.uint8)

    class_map = np.asarray(damage_map, dtype=np.uint8)
    colored = colors[class_map]

    if base_image_path:
        base = Image.open(base_image_path).convert("RGB")
        base = base.resize(
            (class_map.shape[1], class_map.shape[0]),
            Image.Resampling.BILINEAR,
        )
        base_array = np.asarray(base, dtype=np.float32)

        # Apply the overlay only to predicted damage/building pixels.
        mask = (class_map > 0) & (class_map < len(colors))

        result = base_array.copy()
        alpha = 0.50

        result[mask] = (
            (1 - alpha) * base_array[mask]
            + alpha * colored[mask]
        )

        colored = np.clip(result, 0, 255).astype(np.uint8)

    Image.fromarray(colored).save(output_path)
    return output_path


def detect_cloud_mask(image_path):
    """Estimate potential cloud pixels in an RGB image."""
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # Initial heuristic: bright pixels with low saturation.
    cloud_mask = (brightness > 180) & (saturation < 80)

    return cloud_mask
