import torch
from pathlib import Path
from experiments.fcn_2d.networks import FCN2DSegmentationModel
from CMRSegment.nn.torch.experiment import Experiment, ExperimentConfig
from CMRSegment.nn.torch.data import construct_training_validation_dataset
from CMRSegment.nn.torch.loss import FocalLoss, DiceCoeff
from CMRSegment.config import DataConfig, get_conf
from pyhocon import ConfigTree, ConfigFactory

TRAIN_CONF_PATH = Path(__file__).parent.joinpath("train.conf")


def main():
    train_conf = ConfigFactory.parse_file(str(TRAIN_CONF_PATH))
    # TODO: copy train_conf to experiment_dir
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
    )

    network = FCN2DSegmentationModel(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
        up_conv_filter=get_conf(train_conf, group="network", key="up_conv_filter"),
        final_conv_filter=get_conf(train_conf, group="network", key="final_conv_filter"),
    )
    training_set, validation_set = construct_training_validation_dataset(
        DataConfig.from_conf(TRAIN_CONF_PATH)
    )
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=get_conf(train_conf, group="optimizer", key="learning_rate"),
        momentum=get_conf(train_conf, group="optimizer", key="momentum"),
    )
    loss = FocalLoss(
        alpha=get_conf(train_conf, group="loss", key="alpha"),
        gamma=get_conf(train_conf, group="loss", key="gamma"),
        logits=True,
    )
    experiment = Experiment(
        config=config,
        network=network,
        training_set=training_set,
        validation_set=validation_set,
        optimizer=optimizer,
        loss=loss,
        other_validation_metrics=[DiceCoeff()],
    )
    experiment.train()
