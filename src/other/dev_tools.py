import cv2
import IPython
import numpy as np
from PIL import Image


def imshow(img):
    _, ret = cv2.imencode('.jpg', img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def save_image(image: np.ndarray, path: str = "image.jpg"):
    PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
    PIL_image.save(path)
