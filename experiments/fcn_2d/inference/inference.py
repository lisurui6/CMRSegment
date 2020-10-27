import torch
from pathlib import Path
from argparse import ArgumentParser
from pyhocon import ConfigTree, ConfigFactory
from CMRSegment.config import DataConfig, get_conf
from experiments.fcn_2d.networks import FCN2DSegmentationModel
from CMRSegment.nn.torch.data import Torch2DSegmentationDataset
import numpy as np
import nibabel as nib
import shutil
from CMRSegment.nn.torch import prepare_tensors


TRAIN_CONF_PATH = Path(__file__).parent.parent.joinpath("train.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="model_path", required=True, type=str)
    parser.add_argument("-i", "--input-path", dest="input_path", required=True, type=str)
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("-n", "--network-conf", dest="network_conf_path", default=None, type=str)
    parser.add_argument("-d", "--device", dest="device", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    checkpoint = torch.load(str(model_path), map_location=torch.device(args.device))
    if args.network_conf_path is not None:
        train_conf = ConfigFactory.parse_file(str(Path(args.network_conf_path)))
    else:
        train_conf = ConfigFactory.parse_file(str(TRAIN_CONF_PATH))
    get_conf(train_conf, group="network", key="experiment_dir")
    network = FCN2DSegmentationModel(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
        up_conv_filter=get_conf(train_conf, group="network", key="up_conv_filter"),
        final_conv_filter=get_conf(train_conf, group="network", key="final_conv_filter"),
        feature_size=get_conf(train_conf, group="network", key="feature_size")
    )
    network.load_state_dict(checkpoint)
    network.cuda(device=args.device)
    image = Torch2DSegmentationDataset.read_image(
        input_path,
        get_conf(train_conf, group="network", key="feature_size"),
        get_conf(train_conf, group="network", key="in_channels")
    )
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()
    image = prepare_tensors(image, gpu=True, device=args.device)
    predicted = network(image)
    predicted = predicted.cpu().detach().numpy()

    nim = nib.load(str(input_path))
    image = nim.get_data()
    # Transpose and crop the segmentation to recover the original size
    predicted = np.squeeze(predicted, axis=0).astype(np.int16)
    print(predicted.shape)
    # map back to original size
    predicted = np.resize(predicted, image.shape)
    print(predicted.shape)
    # if Z < 64:
    #     pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, z1_ - z1:z1_ - z1 + Z]
    # else:
    #     pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, :]
    #     pred_segt = np.pad(pred_segt, ((0, 0), (0, 0), (z_pre, z_post)), 'constant')

    nim2 = nib.Nifti1Image(predicted, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/seg.nii.gz'.format(str(output_dir)))
    shutil.copy(str(input_path), str(output_dir.joinpath("image.nii.gz")))