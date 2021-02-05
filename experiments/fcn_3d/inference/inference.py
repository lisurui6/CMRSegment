import torch
from pathlib import Path
from argparse import ArgumentParser
from pyhocon import ConfigFactory
from CMRSegment.common.config import get_conf
from experiments.fcn_3d.network import UNet
from CMRSegment.common.nn.torch.data import Torch2DSegmentationDataset
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
    network = UNet(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
    )
    network.load_state_dict(checkpoint)
    network.cuda(device=args.device)
    # image = nib.load(str(input_path)).get_data()
    # if image.ndim == 4:
    #     image = np.squeeze(image, axis=-1).astype(np.int16)
    # image = image.astype(np.int16)
    # image = np.transpose(image, (2, 0, 1))
    dataset = Torch2DSegmentationDataset(
        name=dataset_config.name,
        image_paths=[input_path],
        label_paths=[input_path.parent.joinpath(dataset_config.image_label_format.label.format(phase=args.phase))],
        feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="n_slices"),
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
              gpu, device):
    import math
    original_image = image
    Z, X, Y = image.shape
    n_slices = 96
    X2, Y2 = int(math.ceil(X / 32.0)) * 32, int(math.ceil(Y / 32.0)) * 32
    x_pre, y_pre, z_pre = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
    x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
    z1, z2 = int(Z / 2) - int(n_slices / 2), int(Z / 2) + int(n_slices / 2)
    z1_, z2_ = max(z1, 0), min(z2, Z)
    image = image[z1_: z2_, :, :]
    image = np.pad(image, ((z1_ - z1, z2 - z2_), (x_pre, x_post), (y_pre, y_post)), 'constant')
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image, 0)
    image = prepare_tensors(image, gpu, device)

    predicted = network(image)
    predicted = torch.sigmoid(predicted)
    # print("sigmoid", torch.mean(predicted).item(), torch.max(predicted).item())
    predicted = (predicted > 0.5).float()
    # print("0.5", torch.mean(predicted).item(), torch.max(predicted).item())
    predicted = predicted.cpu().detach().numpy()

    nim = nib.load(str(image_path))
    # Transpose and crop the segmentation to recover the original size
    predicted = np.squeeze(predicted, axis=0)
    # print(predicted.shape)
    predicted = np.pad(predicted, ((0, 0), (z1_, Z - z2_), (0, 0), (0, 0)), "constant")
    predicted = predicted[:, z1_ - z1:z1_ - z1 + Z, x_pre:x_pre + X, y_pre:y_pre + Y]

    # map back to original size
    final_predicted = np.zeros((original_image.shape[0], original_image.shape[1], original_image.shape[2]))
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

    nim2 = nib.Nifti1Image(final_predicted, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/seg.nii.gz'.format(str(output_dir)))

    final_image = np.transpose(original_image, [1, 2, 0])
    # print(final_image.shape)
    nim2 = nib.Nifti1Image(final_image, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))
    # shutil.copy(str(input_path), str(output_dir.joinpath("image.nii.gz")))

    final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
    label = label.cpu().detach().numpy()
    for i in range(label.shape[0]):
        final_label[label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    nim2 = nib.Nifti1Image(final_label, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))
