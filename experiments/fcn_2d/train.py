import torch
import shutil
from pathlib import Path
from argparse import ArgumentParser
from experiments.fcn_2d.networks import FCN2DSegmentationModel
from CMRSegment.nn.torch.experiment import Experiment, ExperimentConfig
from CMRSegment.nn.torch.data import construct_training_validation_dataset
from CMRSegment.nn.torch.loss import FocalLoss, DiceCoeff, BCELoss, DiceCoeffWithLogits
from CMRSegment.config import DataConfig, get_conf
from pyhocon import ConfigTree, ConfigFactory


TRAIN_CONF_PATH = Path(__file__).parent.joinpath("train.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--conf", dest="conf_path", default=None, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.conf_path is None:
        train_conf = ConfigFactory.parse_file(str(TRAIN_CONF_PATH))
    else:
        train_conf = ConfigFactory.parse_file(str(Path(args.conf_path)))

    if get_conf(train_conf, group="experiment", key="experiment_dir") is not None:
        experiment_dir = Path(get_conf(train_conf, group="experiment", key="experiment_dir"))
    else:
        experiment_dir = None
    config = ExperimentConfig(
        experiment_dir=experiment_dir,
        batch_size=get_conf(train_conf, group="experiment", key="batch_size"),
        num_epochs=get_conf(train_conf, group="experiment", key="num_epochs"),
        gpu=get_conf(train_conf, group="experiment", key="gpu"),
        device=get_conf(train_conf, group="experiment", key="device"),
        num_workers=get_conf(train_conf, group="experiment", key="num_workers"),
        pin_memory=get_conf(train_conf, group="experiment", key="pin_memory"),
    )
    shutil.copy(str(TRAIN_CONF_PATH), str(config.experiment_dir.joinpath("train.conf")))
    network = FCN2DSegmentationModel(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
        up_conv_filter=get_conf(train_conf, group="network", key="up_conv_filter"),
        final_conv_filter=get_conf(train_conf, group="network", key="final_conv_filter"),
        feature_size=get_conf(train_conf, group="network", key="feature_size")
    )
    training_set, validation_set = construct_training_validation_dataset(
        DataConfig.from_conf(TRAIN_CONF_PATH), feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="in_channels")
    )
    training_set.export(config.experiment_dir.joinpath("training_set.csv"))
    validation_set.export(config.experiment_dir.joinpath("validation_set.csv"))
    if get_conf(train_conf, group="optimizer", key="type") == "SGD":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=get_conf(train_conf, group="optimizer", key="learning_rate"),
            momentum=get_conf(train_conf, group="optimizer", key="momentum"),
        )
    else:
        optimizer = torch.optim.Adam(
            network.parameters(), lr=get_conf(train_conf, group="optimizer", key="learning_rate")
        )
    if get_conf(train_conf, group="loss", key="type") == "FocalLoss":
        loss = FocalLoss(
            alpha=get_conf(train_conf, group="loss", key="alpha"),
            gamma=get_conf(train_conf, group="loss", key="gamma"),
            logits=True,
        )
    else:
        loss = BCELoss()
    experiment = Experiment(
        config=config,
        network=network,
        training_set=training_set,
        validation_set=validation_set,
        optimizer=optimizer,
        loss=loss,
        other_validation_metrics=[DiceCoeffWithLogits()],
    )
    experiment.train()


if __name__ == '__main__':
    main()
