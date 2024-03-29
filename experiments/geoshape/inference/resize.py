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


TRAIN_CONF_PATH = Path(__file__).parent.parent.joinpath("train.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="model_path", required=True, type=str)
    parser.add_argument("-i", "--input-dir", dest="input_dir", required=True, type=str)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("-n", "--network-conf", dest="network_conf_path", default=None, type=str)
    parser.add_argument("-d", "--device", dest="device", default=0, type=int)
    parser.add_argument("-p", "--phase", dest="phase", default="ED", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if args.network_conf_path is not None:
        train_conf = ConfigFactory.parse_file(str(Path(args.network_conf_path)))
        conf_path = Path(args.network_conf_path)
    else:
        train_conf = ConfigFactory.parse_file(str(TRAIN_CONF_PATH))
        conf_path = TRAIN_CONF_PATH
    data_config = DataConfig.from_conf(conf_path)
    dataset_config = DatasetConfig.from_conf(
        name=data_config.training_datasets[0], mount_prefix=data_config.mount_prefix, mode=data_config.data_mode
    )
    input_path = Path(args.input_dir).joinpath(dataset_config.image_label_format.image.format(phase=args.phase))
    output_dir = Path(args.output_dir)
    checkpoint = torch.load(str(model_path), map_location=torch.device(args.device))

    get_conf(train_conf, group="network", key="experiment_dir")
    network = ShapeDeformNet(
        voxel_width=get_conf(train_conf, group="network", key="voxel_width"),
        voxel_height=get_conf(train_conf, group="network", key="voxel_height"),
        voxel_depth=get_conf(train_conf, group="network", key="voxel_depth"),
        num_lv_slices=get_conf(train_conf, group="network", key="num_lv_slices"),
        num_extra_slices=get_conf(train_conf, group="network", key="num_extra_slices"),
    )
    network.load_state_dict(checkpoint)
    network.cuda(device=args.device)

    dataset = Torch2DSegmentationDataset(
        name=dataset_config.name,
        image_paths=[input_path],
        label_paths=[input_path.parent.joinpath(dataset_config.image_label_format.label.format(phase=args.phase))],
        feature_size=get_conf(train_conf, group="network", key="voxel_width"),
        n_slices=get_conf(train_conf, group="network", key="voxel_height"),
        is_3d=True,
    )

    image = dataset.get_image_tensor_from_index(0)
    image = torch.unsqueeze(image, 0)
    image = prepare_tensors(image, True, args.device)

    label = dataset.get_label_tensor_from_index(0)
    inference(
        image=image,
        label=label,
        image_path=input_path,
        network=network,
        output_dir=output_dir,
    )


def inference(image: np.ndarray, label: torch.Tensor, image_path: Path, network: torch.nn.Module, output_dir: Path,
              gpu, device, crop_size, voxel_size):
    # crop_size: (H, W, D)
    import math
    original_image = image
    voxel_width = network.voxel_width
    voxel_height = network.voxel_height
    Z, X, Y = image.shape  # (H, W, D)
    cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
    # image, __ = central_crop_with_padding(image, None, crop_size)
    resized_image = resize_image(image, voxel_size, 0)

    # image = np.transpose(image, [1, 2, 0])  # (W, D, H)
    # image = Torch2DSegmentationDataset.crop_3D_image(image, cx, cy, voxel_width, cz, voxel_height)
    # image = np.transpose(image, (2, 0, 1))  # (H, W, D)

    resized_image = rescale_intensity(resized_image, (1.0, 99.0))

    image = np.expand_dims(resized_image, 0)
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image, 0)
    image = prepare_tensors(image, gpu, device)

    predicted = network(image)
    init_masks, affine_masks, deform_masks, nodes, flow, preint_flow = predicted
    init_mask = torch.cat(init_masks, dim=1).squeeze(0).detach().cpu().numpy()
    affine_mask = torch.cat(affine_masks, dim=1).squeeze(0).detach().cpu().numpy()
    deform_mask = torch.cat(deform_masks, dim=1).squeeze(0).detach().cpu().numpy()
    # predicted = torch.sigmoid(predicted)
    # # print("sigmoid", torch.mean(predicted).item(), torch.max(predicted).item())
    # predicted = (predicted > 0.5).float()
    # # print("0.5", torch.mean(predicted).item(), torch.max(predicted).item())
    # predicted = predicted.cpu().detach().numpy()

    nim = nib.load(str(image_path))
    # Transpose and crop the segmentation to recover the original size
    for predicted, prefix in zip([init_mask, affine_mask, deform_mask], ["init", "affine", "deform"]):
        # predicted: (3, W, D, H)
        # predicted = np.squeeze(predicted, axis=0)

        # predicted = np.pad(predicted, ((0, 0), (cx - voxel_width//2, X - cx - voxel_width//2), (cy - voxel_width//2, Y - cy - voxel_width//2), (cz - voxel_height//2, Z - cz - voxel_height//2)), "constant")
        predicted = np.transpose(predicted, (0, 3, 1, 2))
        # predicted = resize_label(predicted, crop_size, 0)
        # predicted = resize_label(predicted, original_image.shape, 0)
        # __, predicted = central_crop_with_padding(None, predicted, (Z, X, Y))
        predicted = np.transpose(predicted, (0, 2, 3, 1))

        # predicted = predicted[:, z1_ - z1:z1_ - z1 + Z, x_pre:x_pre + X, y_pre:y_pre + Y]
        # map back to original size
        # final_predicted = np.zeros((original_image.shape[1], original_image.shape[2], original_image.shape[0]))
        final_predicted = np.zeros((resized_image.shape[1], resized_image.shape[2], resized_image.shape[0]))
        # print(predicted.shape, final_predicted.shape)
        final_predicted[predicted[2] > 0.5] = 3
        final_predicted[predicted[1] > 0.5] = 2
        final_predicted[predicted[0] > 0.5] = 1

        # final_predicted = np.transpose(final_predicted, [1, 2, 0])

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
        # nim2.header['pixdim'] = nim.header['pixdim']
        output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, '{0}/{1}_seg.nii.gz'.format(str(output_dir), prefix))

    # final_image = np.transpose(original_image, [1, 2, 0])
    final_image = np.transpose(resized_image, [1, 2, 0])
    # print(final_image.shape)
    nim2 = nib.Nifti1Image(final_image, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))
    # shutil.copy(str(input_path), str(output_dir.joinpath("image.nii.gz")))
    print(label.shape)
    # label = torch.movedim(label, 1, -1)
    label = label.detach().cpu().numpy()
    final_label = np.zeros(resized_image.shape)
    print(final_label.shape)
    label = resize_label(label, resized_image.shape, 0)
    print(label.shape)
    for i in range(label.shape[0]):
        final_label[label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    nim2 = nib.Nifti1Image(final_label, nim.affine)
    # nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))
    # from matplotlib import pyplot as plt
    # plt.figure("label")
    # plt.imshow(final_label[:, :, 16])
    # plt.figure("predicted")
    # plt.imshow(final_predicted[:, :, 16])
    # plt.show()
