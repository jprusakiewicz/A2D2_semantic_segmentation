import numpy as np
import json
from PIL import ImageColor
import sys

sys.path.append("src")
from data.read_data import SAMPLE_IMAGE, SAMPLE_LABEL
from models.unet.test_unet import unet_model
from preprocessors.label import label_to_categorical


def train_model():
    with open("./data/class_list.json") as f:
        class_colors_mapping_hex = json.load(f)

    class_colors_mapping_rgb = {}
    for color, label in class_colors_mapping_hex.items():
        class_colors_mapping_rgb[label] = ImageColor.getcolor(color, "RGB")

    num_classes = len(class_colors_mapping_rgb.keys())

    my_model = unet_model(input_shape=(151, 240, 3), num_classes=num_classes)

    my_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

    cat = label_to_categorical(SAMPLE_LABEL, class_colors_mapping_rgb)
    my_model.fit(x=np.expand_dims(SAMPLE_IMAGE, axis=0), y=cat, epochs=15, steps_per_epoch=1, verbose=2)
    return my_model


def save_model(model, path: str = "./"):
    model.save(path)


if __name__ == '__main__':
    MODEL_SAVE_PATH = "./trained_models/keras_test_unet"
    model = train_model()
    save_model(model, MODEL_SAVE_PATH)
