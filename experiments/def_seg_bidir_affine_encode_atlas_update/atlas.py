import torch
from tqdm import tqdm
from typing import List
from pathlib import Path
import nibabel as nib
import numpy as np
from CMRSegment.common.nn.torch import prepare_tensors
from CMRSegment.common.nn.torch.data import MultiDataLoader
from matplotlib import pyplot as plt


def mean_image_label(data_loader: MultiDataLoader):
    pbar = tqdm(enumerate(data_loader))
    labels = None
    images = None
    n = 0
    for idx, (inputs, outputs) in pbar:
        image, __ = inputs
        label = outputs
        n += image.shape[0]

        # mean_label = torch.squeeze(torch.mean(label, dim=0))
        # mean_image = torch.squeeze(torch.mean(image, dim=0))
        image = np.squeeze(torch.sum(image, dim=0).cpu().detach().numpy(), axis=0)
        label = torch.sum(label, dim=0).cpu().detach().numpy()

        if images is None:
            images = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        if labels is None:
            labels = np.zeros((label.shape[0], label.shape[1], label.shape[2], label.shape[3]))
        images += image
        labels += label
    mean_image = images / n
    mean_label = labels / n
    print(n)
    # labels = np.concatenate(labels, axis=0)
    # mean_label = np.mean(labels, axis=0)
    # images = np.concatenate(images, axis=0)
    # mean_image = np.mean(images, axis=0)
    # mean_labels = torch.stack(mean_labels, dim=0)
    # mean_label = torch.mean(mean_labels, dim=0)
    # mean_images = torch.stack(mean_images, dim=0)
    # mean_image = torch.mean(mean_images, dim=0)
    return mean_image, mean_label


class Atlas:
    def __init__(self, image: np.ndarray, label: np.ndarray):
        self._image = image
        self._label = label

    def image(self):
        return self._image

    def label(self):
        return self._label

    def update(self, mean_image: np.ndarray, mean_atlas: np.ndarray, eta: float = 0.01):
        mean_image = (1 - eta) * self.image() + eta * mean_image
        mean_atlas = (1 - eta) * self.label() + eta * mean_atlas
        self._image = mean_image
        self._label = mean_atlas

    @classmethod
    def from_data_loader(cls, data_loader: MultiDataLoader):
        image, label = mean_image_label(data_loader)
        # image = image.cpu().detach().numpy()
        # label = label.cpu().detach().numpy()
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

        mean_atlas = self._label
        for i in range(mean_atlas.shape[0]):
            slice = mean_atlas[i, 30, :, :]
            fig = plt.figure()
            plt.imshow(slice)
            plt.savefig(str(output_dir.joinpath("label_{}_slice_1.png".format(i))))
            plt.close(fig)
            slice = mean_atlas[i, :, 64, :]
            fig = plt.figure()
            plt.imshow(slice)
            plt.savefig(str(output_dir.joinpath("label_{}_slice_2.png".format(i))))
            plt.close(fig)
            slice = mean_atlas[i, :, :, 64]
            fig = plt.figure()
            plt.imshow(slice)
            plt.savefig(str(output_dir.joinpath("label_{}_slice_3.png".format(i))))
            plt.close(fig)
