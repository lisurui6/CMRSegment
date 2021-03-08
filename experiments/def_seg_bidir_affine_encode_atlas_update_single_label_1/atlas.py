import torch
from tqdm import tqdm
from typing import List
from pathlib import Path
import nibabel as nib
import numpy as np
from CMRSegment.common.nn.torch import prepare_tensors
from CMRSegment.common.nn.torch.data import MultiDataLoader


def mean_image_label(data_loader: MultiDataLoader):
    pbar = tqdm(enumerate(data_loader))
    mean_labels = []
    mean_images = []
    for idx, (inputs, outputs) in pbar:
        image, __ = inputs
        label, __ = outputs
        mean_label = torch.squeeze(torch.mean(label, dim=0))
        mean_label = torch.unsqueeze(mean_label, dim=0)
        mean_image = torch.squeeze(torch.mean(image, dim=0))

        mean_labels.append(mean_label)
        mean_images.append(mean_image)
    mean_labels = torch.stack(mean_labels, dim=0)
    mean_label = torch.mean(mean_labels, dim=0)
    mean_images = torch.stack(mean_images, dim=0)
    mean_image = torch.mean(mean_images, dim=0)
    return mean_image, mean_label


class Atlas:
    def __init__(self, image: np.ndarray, label: np.ndarray, eta: float = 0.01):
        self._image = image
        self._label = label
        self.eta = eta

    def image(self):
        return self._image

    def label(self):
        return self._label

    def update(self, warped_images: List[np.ndarray], warped_templates: List[np.ndarray]):
        mean_image = np.stack(warped_images, axis=0)
        mean_image = np.mean(mean_image, axis=0)

        mean_atlas = np.stack(warped_templates, axis=0)
        mean_atlas = np.mean(mean_atlas, axis=0)
        mean_image = (1 - self.eta) * self.image() + self.eta * mean_image
        mean_atlas = (1 - self.eta) * self.label() + self.eta * mean_atlas
        self._image = mean_image
        self._label = mean_atlas

    @classmethod
    def from_data_loader(cls, data_loader: MultiDataLoader):
        image, label = mean_image_label(data_loader)
        image = image.detach().numpy()
        label = label.detach().numpy()
        return cls(image, label)

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        final_image = self._image
        # final_image = np.squeeze(final_image, 0)
        final_image = np.transpose(final_image, [1, 2, 0])
        # print(final_image.shape)
        nim2 = nib.Nifti1Image(final_image, None)
        nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))

        final_label = np.zeros((self._image.shape[0], self._image.shape[1], self._image.shape[2]))
        label = self._label > 0.5

        for i in range(label.shape[0]):
            final_label[label[i, :, :, :] == 1.0] = i + 1

        final_label = np.transpose(final_label, [1, 2, 0])
        # print(final_label.shape)
        nim2 = nib.Nifti1Image(final_label, None)
        output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))
