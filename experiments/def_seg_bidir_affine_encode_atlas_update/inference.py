import torch
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
from CMRSegment.common.nn.torch import prepare_tensors
from CMRSegment.common.nn.torch.augmentation import central_crop_with_padding

TRAIN_CONF_PATH = Path(__file__).parent.joinpath("train.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="model_path", required=True, type=str)
    parser.add_argument("-i", "--input-dir", dest="input_dir", required=True, type=str)
    parser.add_argument("-t", "--template", dest="template_path", required=True)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("-n", "--network-conf", dest="network_conf_path", default=None, type=str)
    parser.add_argument("-d", "--device", dest="device", default=0, type=int)
    parser.add_argument("-p", "--phase", dest="phase", default="ED", type=str)
    return parser.parse_args()


def inference(image: np.ndarray, label: np.ndarray, network: torch.nn.Module, output_dir: Path,
              atlas, output_size, gpu, device):
    np_cropped_image, np_cropped_label = central_crop_with_padding(image, label, output_size=output_size)

    cropped_label = torch.from_numpy(np_cropped_label).float()
    cropped_image = np.expand_dims(np_cropped_image, 0)
    cropped_image = torch.from_numpy(cropped_image).float()

    cropped_image = torch.unsqueeze(cropped_image, 0)
    cropped_label = torch.unsqueeze(cropped_label, 0)
    cropped_image = prepare_tensors(cropped_image, gpu, device)
    cropped_label = prepare_tensors(cropped_label, gpu, device)

    affine_warped_template, warped_template, pred_maps, flow, warped_image, warped_label, affine_warped_label, \
    batch_atlas, affine_warped_image = network((cropped_image, cropped_label), atlas)
    for prefix, predicted in zip(
            ["warped_template", "affine_warped_template", "pred_maps"],
            [warped_template, affine_warped_template, pred_maps]
    ):

        # predicted = torch.sigmoid(predicted)
        # print("sigmoid", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = (predicted > 0.5).float()
        # print("0.5", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = predicted.cpu().detach().numpy()

        # nim = nib.load(str(image_path))
        # Transpose and crop the segmentation to recover the original size
        predicted = np.squeeze(predicted, axis=0)
        # print(predicted.shape)
        # predicted = np.pad(predicted, ((0, 0), (z1_, Z - z2_), (0, 0), (0, 0)), "constant")
        # predicted = predicted[:, z1_ - z1:z1_ - z1 + Z, x_pre:x_pre + X, y_pre:y_pre + Y]

        # map back to original size
        # final_predicted = np.zeros((image.shape[2], image.shape[3], image.shape[4]))
        __, predicted = central_crop_with_padding(None, predicted, image.shape)
        final_predicted = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

        # print(predicted.shape, final_predicted.shape)
        for i in range(predicted.shape[0]):
            # print(a.shape)
            final_predicted[predicted[i, :, :, :] > 0.5] = i + 1
        # image = nim.get_data()
        final_predicted = np.transpose(final_predicted, [1, 2, 0])

        # print(predicted.shape, final_predicted.shape)
        # final_predicted = np.resize(final_predicted, (image.shape[0], image.shape[1], image.shape[2]))

        # print(predicted.shape, final_predicted.shape, np.max(final_predicted), np.mean(final_predicted),
        #       np.min(final_predicted))
        # if Z < 64:
        #     pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, z1_ - z1:z1_ - z1 + Z]
        # else:
        #     pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, :]
        #     pred_segt = np.pad(pred_segt, ((0, 0), (0, 0), (z_pre, z_post)), 'constant')

        # nim2 = nib.Nifti1Image(final_predicted, nim.affine)
        # nim2.header['pixdim'] = nim.header['pixdim']
        nim2 = nib.Nifti1Image(final_predicted, None)

        output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, '{}/{}_seg.nii.gz'.format(str(output_dir), prefix))
    # final_image = image.cpu().detach().numpy()
    # final_image = np.squeeze(final_image)
    # final_image, __ = central_crop_with_padding(final_image, None, image.shape)

    final_image = np.transpose(image, [1, 2, 0])
    # print(final_image.shape)
    # nim2 = nib.Nifti1Image(final_image, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    nim2 = nib.Nifti1Image(final_image, None)

    nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))
    # shutil.copy(str(input_path), str(output_dir.joinpath("image.nii.gz")))
    final_cropped_image = np.transpose(np_cropped_image, [1, 2, 0])
    nim2 = nib.Nifti1Image(final_cropped_image, None)
    nib.save(nim2, '{0}/cropped_image.nii.gz'.format(str(output_dir)))

    final_cropped_image, final_cropped_label = central_crop_with_padding(np_cropped_image, np_cropped_label, image.shape)
    final_cropped_image = np.transpose(final_cropped_image, [1, 2, 0])
    nim2 = nib.Nifti1Image(final_cropped_image, None)
    nib.save(nim2, '{0}/padded_cropped_image.nii.gz'.format(str(output_dir)))

    warped_image = warped_image.cpu().detach().numpy()
    warped_image = np.squeeze(warped_image, axis=0)
    warped_image = np.squeeze(warped_image, axis=0)
    warped_image, __ = central_crop_with_padding(warped_image, None, image.shape)
    final_warped_image = np.transpose(warped_image, [1, 2, 0])
    nim2 = nib.Nifti1Image(final_warped_image, None)
    nib.save(nim2, '{0}/padded_warped_image.nii.gz'.format(str(output_dir)))

    affine_warped_image = affine_warped_image.cpu().detach().numpy()
    affine_warped_image = np.squeeze(affine_warped_image, axis=0)
    affine_warped_image = np.squeeze(affine_warped_image, axis=0)
    affine_warped_image, __ = central_crop_with_padding(affine_warped_image, None, image.shape)
    final_warped_image = np.transpose(affine_warped_image, [1, 2, 0])
    nim2 = nib.Nifti1Image(final_warped_image, None)
    nib.save(nim2, '{0}/padded_affine_warped_image.nii.gz'.format(str(output_dir)))

    affine_warped_label = affine_warped_label.cpu().detach().numpy()
    affine_warped_label = np.squeeze(affine_warped_label, axis=0)
    affine_warped_label = affine_warped_label > 0.5
    __, affine_warped_label = central_crop_with_padding(None, affine_warped_label, image.shape)

    final_label = np.zeros((affine_warped_label.shape[1], affine_warped_label.shape[2], affine_warped_label.shape[3]))
    # label = label.cpu().detach().numpy()
    # label = np.squeeze(label, 0)

    for i in range(affine_warped_label.shape[0]):
        final_label[affine_warped_label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    # nim2 = nib.Nifti1Image(final_label, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    nim2 = nib.Nifti1Image(final_label, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/padded_affine_warped_label.nii.gz'.format(str(output_dir)))


    warped_label = warped_label.cpu().detach().numpy()
    warped_label = np.squeeze(warped_label, axis=0)
    warped_label = warped_label > 0.5
    __, warped_label = central_crop_with_padding(None, warped_label, image.shape)

    final_label = np.zeros((warped_label.shape[1], warped_label.shape[2], warped_label.shape[3]))
    # label = label.cpu().detach().numpy()
    # label = np.squeeze(label, 0)

    for i in range(warped_label.shape[0]):
        final_label[warped_label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    # nim2 = nib.Nifti1Image(final_label, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    nim2 = nib.Nifti1Image(final_label, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/padded_warped_label.nii.gz'.format(str(output_dir)))


    final_label = np.zeros((np_cropped_label.shape[1], np_cropped_label.shape[2], np_cropped_label.shape[3]))
    # label = label.cpu().detach().numpy()
    # label = np.squeeze(label, 0)

    for i in range(np_cropped_label.shape[0]):
        final_label[np_cropped_label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    # nim2 = nib.Nifti1Image(final_label, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    nim2 = nib.Nifti1Image(final_label, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/cropped_label.nii.gz'.format(str(output_dir)))

    final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
    # label = label.cpu().detach().numpy()
    # label = np.squeeze(label, 0)
    for i in range(label.shape[0]):
        final_label[label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    # nim2 = nib.Nifti1Image(final_label, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    nim2 = nib.Nifti1Image(final_label, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))
