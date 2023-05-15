import os
import tarfile
from os.path import isfile

from PIL import Image
import numpy as np


def read_data(tar_file: str):
    """
    GENERAQTED BY CHATGPT LOL
    Reads all images from a single scene in the A2D2 dataset.

    Parameters:
    scene_dir (str): Path to the directory containing the tar file for the scene.

    Returns:
    A list of PIL Image objects, each corresponding to an image in the scene.
    """

    # Open the tar file and read the images
    images = []
    with tarfile.open(tar_file) as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith('.png'):
                f = tf.extractfile(member)
                image = np.array(Image.open(f))
                images.append(image)
    return images


def read_image(image_path: str):
    return np.array(Image.open(image_path).resize((240, 151)))


def read_images(images_directory_path: str):
    return np.asarray([np.array(Image.open(os.path.join(images_directory_path, image)).resize((240, 151)))
            for image in os.listdir(images_directory_path)
            if isfile(os.path.join(images_directory_path, image))])
