import torch
import shutil
from pathlib import Path
from argparse import ArgumentParser
from experiments.geoshape.nets import ShapeDeformNet
from experiments.geoshape.nets.lite import LiteShapeDeformNet
from experiments.geoshape.nets.mid import MidShapeDeformNet

from CMRSegment.common.nn.torch.experiment import Experiment, ExperimentConfig
from CMRSegment.common.nn.torch.data import construct_training_validation_dataset
from experiments.geoshape.loss import InitLVDiceCoeff, InitLVMyoDiceCoeff, InitRVDiceCoeff, InitAllDiceCoeff
from experiments.geoshape.loss import AffineLVDiceCoeff, AffineLVMyoDiceCoeff, AffineRVDiceCoeff, AffineAllDiceCoeff
from experiments.geoshape.loss import DeformLVDiceCoeff, DeformLVMyoDiceCoeff, DeformRVDiceCoeff, DeformAllDiceCoeff
from CMRSegment.common.config import DataConfig, get_conf, AugmentationConfig
from pyhocon import ConfigFactory
from experiments.geoshape.inference.inference import inference
from experiments.geoshape.experiment import GeoShapeExperiment
from experiments.geoshape.loss import ShapeDeformLoss

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
    if get_conf(train_conf, group="network", key="name") == "lite":
        network = LiteShapeDeformNet(
            voxel_width=get_conf(train_conf, group="network", key="voxel_width"),
            voxel_height=get_conf(train_conf, group="network", key="voxel_height"),
            voxel_depth=get_conf(train_conf, group="network", key="voxel_depth"),
            num_lv_slices=get_conf(train_conf, group="network", key="num_lv_slices"),
            num_extra_slices=get_conf(train_conf, group="network", key="num_extra_slices"),
            enc_dim=get_conf(train_conf, group="network", key="enc_dim"),
        )
    elif get_conf(train_conf, group="network", key="name") == "full":
        network = ShapeDeformNet(
            voxel_width=get_conf(train_conf, group="network", key="voxel_width"),
            voxel_height=get_conf(train_conf, group="network", key="voxel_height"),
            voxel_depth=get_conf(train_conf, group="network", key="voxel_depth"),
            num_lv_slices=get_conf(train_conf, group="network", key="num_lv_slices"),
            num_extra_slices=get_conf(train_conf, group="network", key="num_extra_slices"),
            enc_dim=get_conf(train_conf, group="network", key="enc_dim"),
        )
    elif get_conf(train_conf, group="network", key="name") == "mid":
        network = MidShapeDeformNet(
            voxel_width=get_conf(train_conf, group="network", key="voxel_width"),
            voxel_height=get_conf(train_conf, group="network", key="voxel_height"),
            voxel_depth=get_conf(train_conf, group="network", key="voxel_depth"),
            num_lv_slices=get_conf(train_conf, group="network", key="num_lv_slices"),
            num_extra_slices=get_conf(train_conf, group="network", key="num_extra_slices"),
            enc_dim=get_conf(train_conf, group="network", key="enc_dim"),
        )
    training_sets, validation_sets, extra_validation_sets = construct_training_validation_dataset(
        DataConfig.from_conf(conf_path), feature_size=get_conf(train_conf, group="network", key="voxel_width"),
        n_slices=get_conf(train_conf, group="network", key="voxel_height"), is_3d=True, seed=config.seed,
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

    if get_conf(train_conf, group="experiment", key="resume") is not None:
        checkpoint = torch.load(get_conf(train_conf, group="experiment", key="resume"))

        # if only model state dict is saved.
        network.load_state_dict(checkpoint)
        start_epoch = int(get_conf(train_conf, group="experiment", key="resume").split("_")[-1][0])

        # if model, optimizer and epoch are saved.
        # network.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0
    loss = ShapeDeformLoss(flow_lambda=get_conf(train_conf, group="loss", key="flow_lambda"))
    experiment = GeoShapeExperiment(
        config=config,
        network=network,
        training_sets=training_sets,
        validation_sets=validation_sets,
        extra_validation_sets=extra_validation_sets,
        optimizer=optimizer,
        loss=loss,
        other_validation_metrics=[
            InitLVDiceCoeff(),
            InitLVMyoDiceCoeff(),
            InitRVDiceCoeff(),
            InitAllDiceCoeff(),

            AffineLVDiceCoeff(),
            AffineLVMyoDiceCoeff(),
            AffineRVDiceCoeff(),
            AffineAllDiceCoeff(),

            DeformLVDiceCoeff(),
            DeformLVMyoDiceCoeff(),
            DeformRVDiceCoeff(),
            DeformAllDiceCoeff(),
        ],
        inference_func=inference,
    )
    experiment.train(start_epoch=start_epoch)


if __name__ == '__main__':
    main()
