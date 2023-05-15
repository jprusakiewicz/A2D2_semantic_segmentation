import sys
import numpy as np
from huggingface_hub import from_pretrained_keras

sys.path.append("src")

from data.read_data import read_image
from preprocessors.label import num_classes, colormap, masks_to_image
from other.dev_tools import save_image

model = from_pretrained_keras("/Users/kuba/PJATK/magisterka/a2d2_semantic_segmentation/trained_models/keras_test_unet",
                              local_files_only=True)
image = read_image("/Users/kuba/PJATK/magisterka/a2d2-preview/camera_lidar_semantic/20181204_135952/camera/cam_front_center/20181204135952_camera_frontcenter_000002806.png")
predicted = model.predict(np.expand_dims(image, axis=0))
predictions = np.squeeze(predicted)
predictions = np.argmax(predictions, axis=2)

predicted_image = masks_to_image(predictions, num_classes, colormap)
save_image(predicted_image)
