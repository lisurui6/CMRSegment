import os

os.environ['VXM_BACKEND'] = 'pytorch'

import torch
import shutil
from pathlib import Path
from argparse import ArgumentParser
from experiments.fcn_3d.network import UNet
from CMRSegment.common.nn.torch.experiment import Experiment, ExperimentConfig
from CMRSegment.common.nn.torch.data import construct_training_validation_dataset, Torch2DSegmentationDataset
from CMRSegment.common.nn.torch.loss import FocalLoss, BCELoss, DiceCoeffWithLogits, DiceCoeff, MSELoss
from CMRSegment.common.config import DataConfig, get_conf, AugmentationConfig
from pyhocon import ConfigFactory
from experiments.fcn_3d.inference.inference import inference
from experiments.def_seg.network import DefSegNet
from experiments.def_seg.loss import DefSegWarpedDice, DefSegPredDice, DefSegLoss


TRAIN_CONF_PATH = Path(__file__).parent.joinpath("train.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--conf", dest="conf_path", default=None, type=str)
    # parser.add_argument("-a", "--atlas", dest="atlas_path", required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.conf_path is None:
        train_conf = ConfigFactory.parse_file(str(TRAIN_CONF_PATH))
        conf_path = TRAIN_CONF_PATH
    else:
        train_conf = ConfigFactory.parse_file(str(Path(args.conf_path)))
        conf_path = Path(args.conf_path)

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
        n_inference=get_conf(train_conf, group="experiment", key="n_inference"),
        seed=get_conf(train_conf, group="experiment", key="seed"),
    )
    augmentation_config = AugmentationConfig.from_conf(conf_path)
    shutil.copy(str(conf_path), str(config.experiment_dir.joinpath("train.conf")))
    atlas_path = "D:\surui\\rbh\cardiac\DL_segmentation\RBH_3D_atlases\IHD586_10RZ04876_RBH_IHD_201410241040_MRC25598\seg_lvsa_SR_ED.nii.gz"
    atlas = Torch2DSegmentationDataset.read_label(
        label_path=Path(atlas_path),
        feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="n_slices"),
    )
    network = DefSegNet(
        in_channels=get_conf(train_conf, group="network", key="in_channels"),
        n_classes=get_conf(train_conf, group="network", key="n_classes"),
        n_filters=get_conf(train_conf, group="network", key="n_filters"),
        feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="n_slices"),
        template=atlas,
        int_downsize=2,
    )

    training_sets, validation_sets, extra_validation_sets = construct_training_validation_dataset(
        DataConfig.from_conf(conf_path), feature_size=get_conf(train_conf, group="network", key="feature_size"),
        n_slices=get_conf(train_conf, group="network", key="n_slices"), is_3d=True, seed=config.seed,
        augmentation_config=augmentation_config, output_dir=config.experiment_dir,
    )
    for train in training_sets:
        train.export(config.experiment_dir.joinpath("training_set_{}.csv".format(train.name)))
    for val in validation_sets:
        val.export(config.experiment_dir.joinpath("validation_set_{}.csv".format(val.name)))
    for val in extra_validation_sets:
        val.export(config.experiment_dir.joinpath("extra_validation_set_{}.csv".format(val.name)))
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
        loss = DefSegLoss(penalty="l2", loss_mult=2, template=atlas)
    experiment = Experiment(
        config=config,
        network=network,
        training_sets=training_sets,
        validation_sets=validation_sets,
        extra_validation_sets=extra_validation_sets,
        optimizer=optimizer,
        loss=loss,
        other_validation_metrics=[DefSegWarpedDice(template=atlas), DefSegPredDice()],
        inference_func=inference
    )
    experiment.train()


if __name__ == '__main__':
    main()
