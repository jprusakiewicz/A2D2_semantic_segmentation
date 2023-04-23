import numpy as np
from tensorflow.keras.utils import to_categorical


def label_to_categorical(label, class_colors_mapping_rgb):
    """
    label: (probably) np.array the one from test file """
    num_classes = len(class_colors_mapping_rgb.keys())

    mask = np.zeros(label.shape, dtype=np.uint8)
    for idx, color in enumerate(class_colors_mapping_rgb.values()):
        mask[np.all(label == color, axis=-1)] = idx+1
    mask = mask[:, :, 0]
    masks = np.expand_dims(mask, axis=0)
    masks = np.expand_dims(masks, axis=3)

    cat = to_categorical(masks, num_classes=num_classes)
    cat.shape
    return cat
