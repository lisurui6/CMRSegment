import os
from pathlib import Path
from CMRSegment.config import DatasetConfig, DataConfig
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from typing import List, Tuple
from CMRSegment.common.data_table import DataTable


def construct_training_validation_dataset(
    data_config: DataConfig, feature_size: int, n_slices: int
) -> Tuple["Torch2DSegmentationDataset", "Torch2DSegmentationDataset"]:
    datasets = [
        DatasetConfig.from_conf(name, mode=data_config.data_mode, mount_prefix=data_config.mount_prefix)
        for name in data_config.dataset_names
    ]
    image_paths = []
    label_paths = []
    for dataset in datasets:
        paths = sorted(os.listdir(str(dataset.dir)))
        for path in paths:
            for phase in ["ED", "ES"]:
                path = dataset.dir.joinpath(path)
                image_path = path.joinpath(dataset.image_label_format.image.format(phase=phase))
                label_path = path.joinpath(dataset.image_label_format.label.format(phase=phase))
                if image_path.exists() and label_path.exists():
                    image_paths.append(path.joinpath(dataset.image_label_format.image.format(phase=phase)))
                    label_paths.append(path.joinpath(dataset.image_label_format.label.format(phase=phase)))
    c = list(zip(image_paths, label_paths))
    random.shuffle(c)
    image_paths, label_paths = zip(*c)

    train_image_paths = image_paths[:int((1 - data_config.validation_split) * len(image_paths))]
    val_image_paths = image_paths[int((1 - data_config.validation_split) * len(image_paths)):]

    train_label_paths = label_paths[:int((1 - data_config.validation_split) * len(label_paths))]
    val_label_paths = label_paths[int((1 - data_config.validation_split) * len(label_paths)):]

    train_set = Torch2DSegmentationDataset(
        train_image_paths, train_label_paths, feature_size=feature_size, n_slices=n_slices
    )
    val_set = Torch2DSegmentationDataset(val_image_paths, val_label_paths, feature_size=feature_size, n_slices=n_slices)
    return train_set, val_set


class TorchDataset(Dataset):
    def __init__(self, image_paths: List[Path], label_paths: List[Path]):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def sequential_loader(self, batch_size: int, n_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, sampler=SequentialSampler(self), num_workers=n_workers, pin_memory=pin_memory
        )

    def random_loader(self, batch_size: int, n_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, sampler=RandomSampler(self), num_workers=n_workers, pin_memory=pin_memory
        )

    def export(self, output_path: Path):
        """Save paths to csv and config"""
        data_table = DataTable(columns=["image_paths", "label_paths"], data=zip(self.image_paths, self.label_paths))
        data_table.to_csv(output_path)
        return output_path

    def augment(self):
        pass


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


class Torch2DSegmentationDataset(TorchDataset):
    def __init__(self, image_paths: List[Path], label_paths: List[Path], n_slices: int, feature_size: int):
        super().__init__(image_paths, label_paths)
        self.n_slices = n_slices
        self.feature_size = feature_size

    @staticmethod
    def read_image(image_path: Path, feature_size: int, n_slices: int) -> np.ndarray:
        image = nib.load(str(image_path)).get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        image = image.astype(np.float32)
        X, Y, Z = image.shape
        cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
        # image = Torch2DSegmentationDataset.crop_3D_image(image, cx, cy, feature_size, cz, n_slices)
        image = np.resize(image, (feature_size, feature_size, n_slices))
        image = np.transpose(image, (2, 0, 1))
        return image

    @staticmethod
    def read_label(label_path: Path, feature_size: int, n_slices: int) -> np.ndarray:
        label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        label = np.transpose(label, axes=(2, 1, 0))
        if label.ndim == 4:
            label = np.squeeze(label, axis=-1).astype(np.int16)
        label = label.astype(np.float32)
        label[label == 4] = 3
        X, Y, Z = label.shape
        cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
        # label = Torch2DSegmentationDataset.crop_3D_image(label, cx, cy, feature_size, cz, n_slices)
        label = np.resize(label, (feature_size, feature_size, n_slices))

        labels = []
        for i in range(1, 4):
            blank_image = np.zeros((feature_size, feature_size, n_slices))
            blank_image[label == i] = 1
            labels.append(blank_image)
        label = np.array(labels)
        label = np.transpose(label, (0, 3, 1, 2))
        return label

    def __getitem__(self, index: int):
        image = self.read_image(self.image_paths[index], self.feature_size, self.n_slices)
        label = self.read_label(self.label_paths[index], self.feature_size, self.n_slices)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label

    @staticmethod
    def crop_image(image, cx, cy, size):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        r = int(size / 2)
        x1, x2 = cx - r, cx + r
        y1, y2 = cy - r, cy + r
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)), 'constant')
        elif crop.ndim == 4:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)), 'constant')
        elif crop.ndim == 2:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_)), 'constant')
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def augment(self):
        pass

    @staticmethod
    def crop_3D_image(image, cx, cy, size_xy, cz, size_z):
        """ Crop a 3D image using a bounding box centred at (cx, cy, cz) with specified size """
        X, Y, Z = image.shape[:3]
        rxy = int(size_xy / 2)
        r_z = int(size_z / 2)
        x1, x2 = cx - rxy, cx + rxy
        y1, y2 = cy - rxy, cy + rxy
        z1, z2 = cz - r_z, cz + r_z
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)), 'constant')
        elif crop.ndim == 4:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_), (0, 0)), 'constant')
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop
