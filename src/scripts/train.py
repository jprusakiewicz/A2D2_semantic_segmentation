import numpy as np
import sys

sys.path.append("src")
from data.read_data import SAMPLE_IMAGE, SAMPLE_LABEL
from models.unet.test_unet import unet_model
from preprocessors.label import label_to_categorical, class_colors_mapping_rgb, num_classes, colormap


def train_model(x, y, num_classes: int):
    my_model = unet_model(input_shape=(151, 240, 3), num_classes=num_classes)
    my_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
    my_model.fit(x=x, y=y, epochs=500, steps_per_epoch=1, verbose=2)

    return my_model


def save_model(model, path: str = "./"):
    model.save(path)


if __name__ == '__main__':
    MODEL_SAVE_PATH = "./trained_models/keras_test_unet"

    cat = label_to_categorical(SAMPLE_LABEL, class_colors_mapping_rgb)
    model = train_model(np.expand_dims(SAMPLE_IMAGE, axis=0), cat, num_classes)
    save_model(model, MODEL_SAVE_PATH)
