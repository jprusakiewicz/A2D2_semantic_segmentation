import sys

import numpy as np

from omegaconf import OmegaConf

sys.path.append("src")
from data.read_data import read_images
from models.unet.test_unet import unet_model
from preprocessors.label import label_to_categorical, class_colors_mapping_rgb, num_classes, colormap


def train_model(x, y, num_classes: int, config):
    my_model = unet_model(input_shape=(151, 240, 3), num_classes=num_classes)
    my_model.compile(**config.compile_params)
    my_model.fit(x=x, y=y, **config.fit_params)
    return my_model


def save_model(model, path: str = "./"):
    model.save(path)


def run_training(config):
    sample_labels = read_images(config.training.sample_label_path)
    sample_images = read_images(config.training.sample_image_path)
    cat = label_to_categorical(sample_labels, class_colors_mapping_rgb)
    model = train_model(x=sample_images, y=cat,
                        num_classes=num_classes, config=config.training.model)
    save_model(model, config.training.model_save_path)


if __name__ == '__main__':
    config = OmegaConf.load('config/test_config.yaml')
    run_training(config)
