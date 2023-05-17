"""
A2D2
└── camera_lidar_semantic
    ├── 20180807_145028
    │         ├── camera
    │         │         └── cam_front_center
    │         │                       ├──20181204170238_camera_frontcenter_000000330.json
    │         │                       ├── 20181204170238_camera_frontcenter_000000330.png
    │         │                       ├── 20181204170238_camera_frontcenter_000000582.json
    │         │                       └── <more files>
    │         ├── label
    │         │        └── cam_front_center
    │         │                       ├──20181204170238_label_frontcenter_000000330.json
    │         │                       ├── 20181204170238_label_frontcenter_000000582.json
    │         │                       └── <more files>
    │         │
    │         └── lidar
    │             └── cam_front_center
    │                           └── <more files>
    ├── 20180810_142822
    │         ├── camera
    │         │        ├── cam_front_center
    │         │        ├── cam_front_left
    │         │        └── cam_rear_center
    │         ├── label
    │         │       ├── cam_front_center
    │         │       ├── cam_front_left
    │         │       └── cam_rear_center
    │         └── lidar
    │                 ├── cam_front_center
    │                 ├── cam_front_left
    │                 └── cam_rear_center
    └──<more directories>
"""
import os

import numpy as np
from tensorflow.keras.utils import Sequence


class A2D2DataLoader(Sequence):

    def __init__(self, dir_path: str = "../A2D2", batch_size: int = 16):
        self.dir_path = dir_path
        self.batch_size = batch_size
        # Add any other initialization steps here

        self.data_paths = self.lookup_data()
        self.num_samples = len(self.data_paths)

        # Load and preprocess your data
        # self.data = self.load_data()
        # Total number of samples

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index) -> tuple:
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        # Extract the batch of data based on the start and end indices
        batch_data_paths = self.data_paths[start_index:end_index]

        batch_data = self.load_data(batch_data_paths)

        # Preprocess the batch_data as needed
        processed_data = self.preprocess_data(batch_data)

        # Return the batch of processed data
        return processed_data

    def lookup_data(self) -> list[dict]:
        all = []
        for x in [d for d in os.listdir(os.path.join(self.dir_path, "camera_lidar_semantic"))
                  if os.path.isdir(os.path.join(self.dir_path, "camera_lidar_semantic", d))]:
            images = list(sorted([os.path.join(self.dir_path, "camera_lidar_semantic", x, "camera/cam_front_center", image) for image in os.listdir(os.path.join(self.dir_path, "camera_lidar_semantic", x, "camera/cam_front_center")) if image.endswith(".png")]))
            labels = list(sorted([os.path.join(self.dir_path, "camera_lidar_semantic", x, "camera/cam_front_center", label) for label in os.listdir(os.path.join(self.dir_path, "camera_lidar_semantic", x, "label/cam_front_center"))]))
            assert len(images) == len(labels)

            for image, label in zip(images, labels):
                all.append({"image": image, "label": label})

        return all

    def load_data(self, paths: list[dict]) -> list[dict]:
        # paths: each dict consist of "image" and "label" keys
        # Load and preprocess your data here
        # Return the loaded data in a suitable format
        pass

    def preprocess_data(self, data: list[dict]) -> list[dict]:
        # paths: each dict consist of "image" and "label" keys
        # Implement your data preprocessing steps here
        # Return the preprocessed data
        pass
