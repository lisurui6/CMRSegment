import torch
from pathlib import Path
from pyhocon import ConfigFactory
from argparse import ArgumentParser

from CMRSegment.common.config import get_conf
from CMRSegment.segmentor.torch.network import UNet


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", dest="checkpoint_path", type=str, required=True)
    parser.add_argument("-n", "--network-conf", dest="network_conf_path", default=None, type=str)

    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("-d", "--device", dest="device", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.network_conf_path is None:
        train_conf = Path(__file__).parent.joinpath("train.conf")
    else:
        train_conf = ConfigFactory.parse_file(str(Path(args.network_conf_path)))
    network = UNet(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
    )
    checkpoint = torch.load(str(Path(args.checkpoint_path)), map_location=torch.device(args.device))
    network.load_state_dict(checkpoint)
    network.cuda(args.device)
    torch.save(network, str(output_dir.joinpath("inference_model.pt")))


if __name__ == '__main__':
    main()
