import os
from pathlib import Path
from CMRSegment.common.config import DatasetConfig, DataConfig, AugmentationConfig
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
from scipy.ndimage import zoom
from CMRSegment.common.nn.torch.augmentation import augment
from CMRSegment.common.nn.torch.data import Torch2DSegmentationDataset, generate_dataframe, read_dataframe, rescale_intensity, resize_image


def construct_training_validation_dataset(
    data_config: DataConfig, feature_size: int, n_slices: int, output_dir: Path, is_3d: bool = False,
    augmentation_config: AugmentationConfig = None, seed: int = None, template_path: Path = None,
    template_image_path: Path = None,
) -> Tuple[List["Torch2DSegmentationDataset"], List["Torch2DSegmentationDataset"], List["Torch2DSegmentationDataset"]]:
    training_set_configs = [
        DatasetConfig.from_conf(name, mode=data_config.data_mode, mount_prefix=data_config.mount_prefix)
        for name in data_config.training_datasets
    ]

    extra_val_set_configs = [
        DatasetConfig.from_conf(name, mode=data_config.data_mode, mount_prefix=data_config.mount_prefix)
        for name in data_config.extra_validation_datasets
    ]
    training_sets = []
    validation_sets = []
    for config in training_set_configs:
        train, val, template_path = train_val_dataset_from_config(
            dataset_config=config,
            augmentation_config=augmentation_config,
            augmentation_prob=data_config.augmentation_prob,
            validation_split=data_config.validation_split,
            feature_size=feature_size,
            n_slices=n_slices,
            is_3d=is_3d,
            renew_dataframe=data_config.renew_dataframe,
            seed=seed,
            output_dir=output_dir,
            template_path=template_path,
            template_image_path=template_image_path,
        )
        training_sets.append(train)
        validation_sets.append(val)
    extra_val_sets = []
    for config in extra_val_set_configs:
        __, val, __ = train_val_dataset_from_config(
            dataset_config=config,
            validation_split=data_config.validation_split,
            feature_size=feature_size,
            n_slices=n_slices,
            is_3d=is_3d,
            only_val=True,
            renew_dataframe=data_config.renew_dataframe,
            seed=seed,
            output_dir=output_dir,
            template_path=template_path,
            template_image_path=template_image_path,
        )
        extra_val_sets.append(val)
    return training_sets, validation_sets, extra_val_sets


def train_val_dataset_from_config(dataset_config: DatasetConfig, validation_split: float, feature_size: int,
                                  n_slices: int, is_3d: bool, output_dir: Path, only_val: bool = False,
                                  renew_dataframe: bool = False, seed: int = None,
                                  augmentation_config: AugmentationConfig = None, augmentation_prob: float = 0,
                                  template_path: Path = None, template_image_path: Path = None):
    if dataset_config.dataframe_path.exists():
        print("Dataframe {} exists.".format(dataset_config.dataframe_path))
    if not dataset_config.dataframe_path.exists() or renew_dataframe:
        generate_dataframe(dataset_config)
    image_paths, label_paths = read_dataframe(dataset_config.dataframe_path)
    c = list(zip(image_paths, label_paths))
    random.shuffle(c)
    shuffled_image_paths, shuffled_label_paths = zip(*c)
    print("Dataset {} has {} images.".format(dataset_config.name, len(shuffled_image_paths)))
    if dataset_config.size is None:
        size = len(shuffled_image_paths)
    else:
        size = dataset_config.size

    if not only_val:
        train_image_paths = image_paths[:int((1 - validation_split) * size)]
        val_image_paths = image_paths[int((1 - validation_split) * size):size]

        train_label_paths = label_paths[:int((1 - validation_split) * size)]
        val_label_paths = label_paths[int((1 - validation_split) * size):size]
        print("Selecting {} trainig images, {} validation images.".format(len(train_image_paths), len(val_image_paths)))
        if template_path is None:
            template_path = label_paths[0]
        if template_image_path is None:
            template_image_path = image_paths[0]
        print("Template Path: {}".format(template_path))
        print("Template Image Path: {}".format(template_image_path))

        train_set = DefSegDataset(
            template_path=template_path,
            template_image_path=template_image_path,
            name=dataset_config.name,
            image_paths=train_image_paths,
            label_paths=train_label_paths,
            augmentation_prob=augmentation_prob,
            augmentation_config=augmentation_config,
            feature_size=feature_size, n_slices=n_slices, is_3d=is_3d, seed=seed,
            output_dir=output_dir.joinpath("train"),
        )

    else:
        train_set = None
        val_image_paths = image_paths[:size]
        val_label_paths = label_paths[:size]
        print("Selecting {} validation images.".format(len(val_image_paths)))

    val_set = DefSegDataset(
        template_path=template_path,
        template_image_path=template_image_path,
        name=dataset_config.name,
        image_paths=val_image_paths,
        label_paths=val_label_paths,
        feature_size=feature_size, n_slices=n_slices, is_3d=is_3d, seed=seed,
        output_dir=output_dir.joinpath("val"),
    )
    return train_set, val_set, template_path


class DefSegDataset(Torch2DSegmentationDataset):
    def __init__(self, template_path: Path, template_image_path: Path, name: str, image_paths: List[Path], label_paths: List[Path],
                 n_slices: int, feature_size: int, augmentation_prob: float = 0,
                 augmentation_config: AugmentationConfig = None, is_3d: bool = False, seed: int = None,
                 output_dir: Path = None):
        super().__init__(
            name, image_paths, label_paths, n_slices, feature_size, augmentation_prob,
            augmentation_config, is_3d, seed, output_dir
        )
        self.template = self.read_label(template_path, self.feature_size, self.n_slices)
        # self.template_image = self.read_image(template_image_path, self.feature_size, self.n_slices)

    @staticmethod
    def read_image(image_path: Path, feature_size: int, n_slices: int, crop: bool = False) -> np.ndarray:
        image = nib.load(str(image_path)).get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        image = image.astype(np.float32)
        X, Y, Z = image.shape
        cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
        if crop:
            image = Torch2DSegmentationDataset.crop_3D_image(image, cx, cy, feature_size, cz, n_slices)
        else:
            image = resize_image(image, (feature_size, feature_size, n_slices), 0)
        image = np.transpose(image, (2, 0, 1))
        image = rescale_intensity(image, (1.0, 99.0))
        return image

    @staticmethod
    def read_label(label_path: Path, feature_size: int, n_slices: int, crop: bool = False) -> np.ndarray:
        label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        label = np.transpose(label, axes=(2, 1, 0))
        if label.ndim == 4:
            label = np.squeeze(label, axis=-1).astype(np.int16)
        label = label.astype(np.float32)
        label[label == 4] = 3

        X, Y, Z = label.shape
        cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
        if crop:
            label = Torch2DSegmentationDataset.crop_3D_image(label, cx, cy, feature_size, cz, n_slices)
        else:
            label = resize_image(label, (feature_size, feature_size, n_slices), 0)
        blank_image = np.zeros((feature_size, feature_size, n_slices))
        # blank_image = np.zeros((X, Y, Z))

        blank_image[label == 2] = 1
        label = np.expand_dims(blank_image, axis=0)

        label = np.transpose(label, (0, 3, 1, 2))
        return label

    def __getitem__(self, index: int):
        image = self.read_image(self.image_paths[index], self.feature_size, self.n_slices)
        label = self.read_label(self.label_paths[index], self.feature_size, self.n_slices)
        # template_index = np.random.randint(0, len(self))
        # template_index = 0
        # template = self.read_label(self.label_paths[template_index], self.feature_size, self.n_slices)
        if self.augmentation_prob > 0 and self.augmentation_config is not None:
            prob = torch.FloatTensor(1).uniform_(0, 1)
            if prob.item() <= self.augmentation_prob:
                augment(
                    image, label, self.augmentation_config,
                    (self.n_slices, self.feature_size, self.feature_size),
                    seed=self.seed
                )
        self.save(image, label, index)
        if self.is_3d:
            image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        template = torch.from_numpy(self.template).float()
        # template_image = np.expand_dims(self.template_image, 0)
        # template_image = torch.from_numpy(template_image).float()
        return (image, template), (label, template)

    def save(self, image: np.ndarray, label: np.ndarray, index: int):
        if index % 100 == 0:
            nim = nib.load(str(self.image_paths[index]))
            image = np.transpose(image, [1, 2, 0])
            nim2 = nib.Nifti1Image(image, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_path = self.output_dir.joinpath(self.name, "image_{}.nii.gz".format(index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))

            final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
            for i in range(label.shape[0]):
                final_label[label[i, :, :, :] == 1.0] = i + 1
            final_label = np.transpose(final_label, [1, 2, 0])
            nim2 = nib.Nifti1Image(final_label, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_path = self.output_dir.joinpath(self.name, "label_{}.nii.gz".format(index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))
