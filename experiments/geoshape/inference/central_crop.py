import torch
from pathlib import Path
from argparse import ArgumentParser
from pyhocon import ConfigFactory
from CMRSegment.common.config import get_conf
from experiments.geoshape.nets import ShapeDeformNet
from CMRSegment.common.nn.torch.data import Torch2DSegmentationDataset, rescale_intensity, resize_image, resize_label
from CMRSegment.common.nn.torch.augmentation import central_crop_with_padding
from CMRSegment.common.config import DatasetConfig, DataConfig
import numpy as np
import nibabel as nib
from CMRSegment.common.nn.torch import prepare_tensors


def inference(image: np.ndarray, label: torch.Tensor, image_path: Path, network: torch.nn.Module, output_dir: Path,
              gpu, device, crop_size, voxel_size, epoch):
    # crop_size: (H, W, D)
    import math
    from matplotlib import pyplot as plt
    label = label.detach().cpu().numpy()

    original_image = image
    voxel_width = network.voxel_width
    voxel_height = network.voxel_height
    Z, X, Y = image.shape  # (H, W, D)
    cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
    # plt.figure("original image")
    # print(original_image.shape)
    # plt.imshow(original_image[34, :, :])
    cropped_image, cropped_label = central_crop_with_padding(image, label, crop_size)
    # plt.figure("central cropped image")
    # plt.imshow(cropped_image[32, :, :])
    resized_image = resize_image(cropped_image, voxel_size, 0)
    resized_label = resize_label(cropped_label, voxel_size, 0)
    torch_label = torch.from_numpy(resized_label).float()
    # image = np.transpose(image, [1, 2, 0])  # (W, D, H)
    # image = Torch2DSegmentationDataset.crop_3D_image(image, cx, cy, voxel_width, cz, voxel_height)
    # image = np.transpose(image, (2, 0, 1))  # (H, W, D)

    resized_image = rescale_intensity(resized_image, (1.0, 99.0))

    image = np.expand_dims(resized_image, 0)
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image, 0)
    image = prepare_tensors(image, gpu, device)

    predicted = network(image, epoch, torch_label.unsqueeze(0))
    init_masks, affine_masks, deform_masks, nodes, flow, preint_flow = predicted
    init_mask = torch.cat(init_masks, dim=1).squeeze(0).detach().cpu().numpy()
    affine_mask = torch.cat(affine_masks, dim=1).squeeze(0).detach().cpu().numpy()
    deform_mask = torch.cat(deform_masks, dim=1).squeeze(0).detach().cpu().numpy()
    nim = nib.load(str(image_path))
    # Transpose and crop the segmentation to recover the original size
    for predicted, prefix in zip([init_mask], ["init"]):
        # predicted: (3, W, D, H)

        predicted = np.transpose(predicted, (0, 3, 1, 2))
        predicted = resize_label(predicted, crop_size, 0)
        print(predicted.shape)
        # plt.figure("resized predicted")
        # alpha = 0.5
        # plt.imshow(cropped_image[32, :, :], alpha=1-alpha)
        # display_predicted = np.zeros((cropped_image.shape[1], cropped_image.shape[2]))
        # display_predicted[predicted[2, 32, :, :] > 0.5] = 3
        # display_predicted[predicted[1, 32, :, :] > 0.5] = 2
        # display_predicted[predicted[0, 32, :, :] > 0.5] = 1
        # plt.imshow(display_predicted, alpha=alpha)

        __, predicted = central_crop_with_padding(None, predicted, (Z, X, Y))
        print(predicted.shape)
        # plt.figure("padded LV predicted")
        # plt.imshow(original_image[34, :, :], alpha=1-alpha)
        # display_predicted = np.zeros((original_image.shape[1], original_image.shape[2]))
        # display_predicted[predicted[2, 34, :, :] > 0.5] = 3
        # display_predicted[predicted[1, 34, :, :] > 0.5] = 2
        # display_predicted[predicted[0, 34, :, :] > 0.5] = 1
        # plt.imshow(display_predicted, alpha=alpha)
        # plt.show()
        predicted = np.transpose(predicted, (0, 2, 3, 1))

        # map back to original size
        final_predicted = np.zeros((original_image.shape[1], original_image.shape[2], original_image.shape[0]))
        final_predicted[predicted[2] > 0.5] = 3
        final_predicted[predicted[1] > 0.5] = 2
        final_predicted[predicted[0] > 0.5] = 1

        nim2 = nib.Nifti1Image(final_predicted, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, '{0}/{1}_seg.nii.gz'.format(str(output_dir), prefix))

    final_image = np.transpose(original_image, [1, 2, 0])
    nim2 = nib.Nifti1Image(final_image, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))

    resized_image = resize_image(resized_image, crop_size, 0)
    print("here", resized_image.shape)
    image, __ = central_crop_with_padding(resized_image, None, (Z, X, Y))
    print(image.shape)
    final_image = np.transpose(image, [1, 2, 0])
    nim2 = nib.Nifti1Image(final_image, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/padded_image.nii.gz'.format(str(output_dir)))

    # label = torch.movedim(label, 1, -1)
    final_label = np.zeros(original_image.shape)
    for i in range(label.shape[0]):
        final_label[label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    nim2 = nib.Nifti1Image(final_label, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))
    # from matplotlib import pyplot as plt
    # plt.figure("label")
    # plt.imshow(final_label[:, :, 16])
    # plt.figure("predicted")
    # plt.imshow(final_predicted[:, :, 16])
    # plt.show()
