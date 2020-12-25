import torch
import numpy as np
import nibabel as nib
from pathlib import Path


def inference(image: torch.Tensor, label: torch.Tensor, image_path: Path, network: torch.nn.Module, output_dir: Path):

    warped_template, warped_maps, pred_maps, flow = network(image)
    for prefix, predicted in zip(["warped_template", "pred_maps"], [warped_template, pred_maps]):
        # predicted = torch.sigmoid(predicted)
        # print("sigmoid", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = (predicted > 0.5).float()
        # print("0.5", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = predicted.cpu().detach().numpy()

        nim = nib.load(str(image_path))
        # Transpose and crop the segmentation to recover the original size
        predicted = np.squeeze(predicted, axis=0)
        # print(predicted.shape)

        # map back to original size
        final_predicted = np.zeros((image.shape[2], image.shape[3], image.shape[4]))
        # print(predicted.shape, final_predicted.shape)

        for i in range(predicted.shape[0]):
            a = predicted[i, :, :, :] > 0.5
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

        nim2 = nib.Nifti1Image(final_predicted, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, '{}/{}_seg.nii.gz'.format(str(output_dir), prefix))

    final_image = image.cpu().detach().numpy()
    final_image = np.squeeze(final_image, 0)
    final_image = np.squeeze(final_image, 0)
    final_image = np.transpose(final_image, [1, 2, 0])
    # print(final_image.shape)
    nim2 = nib.Nifti1Image(final_image, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))
    # shutil.copy(str(input_path), str(output_dir.joinpath("image.nii.gz")))

    final_label = np.zeros((image.shape[2], image.shape[3], image.shape[4]))
    label = label.cpu().detach().numpy()
    for i in range(label.shape[0]):
        final_label[label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    nim2 = nib.Nifti1Image(final_label, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))
