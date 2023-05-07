import os
import tarfile
from PIL import Image
import numpy


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


SAMPLE_IMAGE = numpy.array(
    Image.open("./data/sample_data/20180807145028_camera_frontcenter_000006882.png").resize((240, 151)))
SAMPLE_LABEL = numpy.array(
    Image.open("./data/sample_data/20180807145028_label_frontcenter_000006882.png").resize((240, 151)))
