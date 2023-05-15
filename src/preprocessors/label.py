import numpy as np
from tensorflow.keras.utils import to_categorical
import json
from PIL import ImageColor


def label_to_categorical(labels, class_colors_mapping_rgb):
    """
    label: (probably) np.array the one from test file """
    num_classes = len(class_colors_mapping_rgb.keys())

    masks = []
    for label in labels:
        mask = np.zeros(label.shape, dtype=np.uint8)
        for idx, color in enumerate(class_colors_mapping_rgb.values()):
            mask[np.all(label == color, axis=-1)] = idx + 1
        mask = mask[:, :, 0]
        masks.append(mask)

    masks = np.asarray(masks, dtype=np.uint8)

    cat = to_categorical(masks, num_classes=num_classes)
    return cat


def get_colors_mapping(class_list_path: str):
    with open(class_list_path) as f:
        class_colors_mapping_hex = json.load(f)
    class_colors_mapping_rgb = {}
    for color, label in class_colors_mapping_hex.items():
        class_colors_mapping_rgb[label] = ImageColor.getcolor(color, "RGB")
    return class_colors_mapping_rgb


def masks_to_image(predicted_masks, num_classes, colormap):
    r = np.zeros_like(predicted_masks).astype(np.uint8)
    g = np.zeros_like(predicted_masks).astype(np.uint8)
    b = np.zeros_like(predicted_masks).astype(np.uint8)
    for l in range(0, num_classes):
        idx = predicted_masks == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


CLASS_LIST_PATH = "./data/class_list.json"

class_colors_mapping_rgb = get_colors_mapping(CLASS_LIST_PATH)
num_classes = len(class_colors_mapping_rgb.keys())

colormap = {idx + 1: color for idx, color in enumerate(class_colors_mapping_rgb.values())}
colormap[0] = (255, 255, 255)
