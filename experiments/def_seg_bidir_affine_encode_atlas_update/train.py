import os

os.environ['VXM_BACKEND'] = 'pytorch'

import torch
import shutil
from pathlib import Path
from argparse import ArgumentParser
from CMRSegment.common.nn.torch.experiment import ExperimentConfig
from CMRSegment.common.config import DataConfig, get_conf, AugmentationConfig
from pyhocon import ConfigFactory
from experiments.def_seg_bidir_affine_encode_atlas_update.inference import inference
from experiments.def_seg_bidir_affine_encode_atlas_update.network import ImgTemplateEncoderNet
from experiments.def_seg_bidir_affine_encode_atlas_update.network.istn import ISTNNet

from experiments.def_seg_bidir_affine_encode_atlas_update.loss import DefWarpedTemplateDice, \
    DefAffineWarpedTemplateDice, DefLoss, DefPredDice
from experiments.def_seg_bidir_affine_encode_atlas_update.data import construct_training_validation_dataset, DefSegDataset
from experiments.def_seg_bidir_affine_encode_atlas_update.experiment import DefSegExperiment
from experiments.def_seg_bidir_affine_encode_atlas_update.atlas import Atlas


TRAIN_CONF_PATH = Path(__file__).parent.joinpath("train.conf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--conf", dest="conf_path", default=None, type=str)
    parser.add_argument("-s", "--starting-epoch", dest="starting_epoch", default=0, type=int)
    parser.add_argument("-e", "--exp-dir", dest="exp_dir", default=None, type=str)


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.starting_epoch > 0:
        assert args.exp_dir is not None
        exp_dir = Path(args.exp_dir)
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

    if get_conf(train_conf, group="network", key="name") == "encodedecode":
        network = ImgTemplateEncoderNet(
            feature_size=get_conf(train_conf, group="network", key="feature_size"),
            n_slices=get_conf(train_conf, group="network", key="n_slices"),
            n_filters=get_conf(train_conf, group="network", key="n_filters"),
            batch_norm=get_conf(train_conf, group="network", key="batch_norm"),
            group_norm=get_conf(train_conf, group="network", key="group_norm"),
            int_downsize=get_conf(train_conf, group="network", key="integrate_downsize"),
            bidir=True,
            in_channels=get_conf(train_conf, group="network", key="in_channels"),
            n_classes=get_conf(train_conf, group="network", key="n_classes"),
            batch_size=get_conf(train_conf, group="experiment", key="batch_size"),
            gpu=get_conf(train_conf, group="experiment", key="gpu"),
            device=get_conf(train_conf, group="experiment", key="device"),
        )
    elif get_conf(train_conf, group="network", key="name") == "istn":
        network = ISTNNet(
            feature_size=get_conf(train_conf, group="network", key="feature_size"),
            n_slices=get_conf(train_conf, group="network", key="n_slices"),
            n_filters=get_conf(train_conf, group="network", key="n_filters"),
            batch_norm=get_conf(train_conf, group="network", key="batch_norm"),
            group_norm=get_conf(train_conf, group="network", key="group_norm"),
            int_downsize=get_conf(train_conf, group="network", key="integrate_downsize"),
            bidir=True,
            in_channels=get_conf(train_conf, group="network", key="in_channels"),
            n_classes=get_conf(train_conf, group="network", key="n_classes"),
            batch_size=get_conf(train_conf, group="experiment", key="batch_size"),
            gpu=get_conf(train_conf, group="experiment", key="gpu"),
            device=get_conf(train_conf, group="experiment", key="device"),
        )
    else:
        raise ValueError("network name not supported. istn or encodedecode")

    if args.starting_epoch > 0:
        print(exp_dir.joinpath("checkpoints", "CP_{}.pth".format(args.starting_epoch-1)))
        checkpoint = torch.load(
            str(exp_dir.joinpath("checkpoints", "CP_{}.pth".format(args.starting_epoch-1))),
            map_location=torch.device(get_conf(train_conf, group="experiment", key="device")),
        )
        network.load_state_dict(checkpoint)
        network.cuda(device=get_conf(train_conf, group="experiment", key="device"))
        image = DefSegDataset.read_image(
            image_path=exp_dir.joinpath("atlas", "epoch_{}".format(args.starting_epoch-1), "image.nii.gz"),
            feature_size=None, n_slices=None,
        )
        label = DefSegDataset.read_label(
            label_path=exp_dir.joinpath("atlas", "epoch_{}".format(args.starting_epoch-1), "label.nii.gz"),
            feature_size=None, n_slices=None,
        )
        atlas = Atlas(image=image, label=label)
    else:
        atlas = None

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

    loss = DefLoss(
        penalty="l2",
        loss_mult=get_conf(train_conf, group="network", key="integrate_downsize"),
        weights=get_conf(train_conf, group="loss", key="weights"),
    )

    experiment = DefSegExperiment(
        config=config,
        network=network,
        training_sets=training_sets,
        validation_sets=validation_sets,
        extra_validation_sets=extra_validation_sets,
        optimizer=optimizer,
        loss=loss,
        other_validation_metrics=[
            DefAffineWarpedTemplateDice(), DefWarpedTemplateDice(), DefPredDice(),
        ],
        inference_func=inference,
    )
    experiment.train(
        starting_epoch=args.starting_epoch,
        atlas=atlas, atlas_eta=get_conf(train_conf, group="network", key="atlas_eta"),
    )


if __name__ == '__main__':
    main()
