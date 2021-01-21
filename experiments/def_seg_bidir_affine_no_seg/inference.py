import torch
from pathlib import Path
from argparse import ArgumentParser
from pyhocon import ConfigFactory
from CMRSegment.common.config import get_conf
from experiments.def_seg_bidir_affine_no_seg.network import DefSegNet
from experiments.def_seg_bidir_affine_no_seg.data import DefSegDataset
from CMRSegment.common.config import DatasetConfig, DataConfig
import numpy as np
import nibabel as nib
from CMRSegment.common.nn.torch import prepare_tensors


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
    network = DefSegNet(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
        feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="n_slices"),
        int_downsize=get_conf(train_conf, group="network", key="integrate_downsize"),
        bidir=True,
    )
    network.load_state_dict(checkpoint)
    network.cuda(device=args.device)
    # image = nib.load(str(input_path)).get_data()
    # if image.ndim == 4:
    #     image = np.squeeze(image, axis=-1).astype(np.int16)
    # image = image.astype(np.int16)
    # image = np.transpose(image, (2, 0, 1))
    dataset = DefSegDataset(
        name=dataset_config.name,
        image_paths=[input_path],
        label_paths=[input_path.parent.joinpath(dataset_config.image_label_format.label.format(phase=args.phase))],
        feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="n_slices"),
        is_3d=True,
        template_path=Path(args.template_path),
    )

    image = dataset.get_image_tensor_from_index(0)
    image = torch.unsqueeze(image, 0)
    image = prepare_tensors(image, True, args.device)

    label = dataset.get_label_tensor_from_index(0)

    template = dataset.template
    template = torch.from_numpy(template).float()
    template = torch.unsqueeze(template, 0)
    template = prepare_tensors(template, True, args.device)

    inference(
        image=(image, template),
        label=label,
        image_path=input_path,
        network=network,
        output_dir=output_dir,
    )


def inference(image: torch.Tensor, label: torch.Tensor, image_path: Path, network: torch.nn.Module, output_dir: Path):
    image, template = image
    # warped_template, warped_maps, pred_maps, flow = network((image, template))
    warped_template, flow = network((image, template))
    for prefix, predicted in zip(["warped_template"], [warped_template]):
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


if __name__ == '__main__':
    main()
