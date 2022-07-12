from CMRSegment.common.nn.torch.experiment import Experiment
import torch
from CMRSegment.common.nn.torch.data import TorchDataset
from CMRSegment.common.nn.torch.data import MultiDataLoader
from CMRSegment.common.nn.torch.augmentation import augment
from CMRSegment.common.nn.torch import prepare_tensors

from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch


def read_image(dataset, idx):
    import math
    image = dataset.read_image(dataset.image_paths[idx], dataset.feature_size, dataset.n_slices)
    X, Y, Z = image.shape
    n_slices = 96
    X2, Y2 = int(math.ceil(X / 32.0)) * 32, int(math.ceil(Y / 32.0)) * 32
    x_pre, y_pre, z_pre = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
    x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
    z1, z2 = int(Z / 2) - int(n_slices / 2), int(Z / 2) + int(n_slices / 2)
    z1_, z2_ = max(z1, 0), min(z2, Z)
    image = image[:, z1_: z2_]
    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z1_ - z1, z2 - z2_)), 'constant')
    image = torch.from_numpy(image).float()
    return image


class GeoShapeExperiment(Experiment):
    def train(self, start_epoch=0):
        self.network.train()
        train_data_loader = MultiDataLoader(
            *self.training_sets,
            batch_size=self.config.batch_size,
            sampler_cls="random",
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        set = False
        for epoch in range(start_epoch, self.config.num_epochs):
            self.network.train()
            self.logger.info("{}: starting epoch {}/{}".format(datetime.now(), epoch, self.config.num_epochs))
            self.loss.reset()
            # if epoch > 10 and not set:
            #     self.optimizer.param_groups[0]['lr'] /= 10
            #     print("-------------Learning rate: {}-------------".format(self.optimizer.param_groups[0]['lr']))
            #     set = True

            # train loop
            pbar = tqdm(enumerate(train_data_loader))
            n = 0
            train_metrics = self.other_validation_metrics
            for metric in train_metrics:
                metric.reset()
            for idx, (inputs, outputs) in pbar:
                inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                predicted = self.network(inputs, epoch, outputs)
                loss = self.loss.cumulate(predicted, outputs)
                for metric in train_metrics:
                    metric.cumulate(predicted, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    "{:.2f} --- {}".format((idx + 1) / len(train_data_loader), self.loss.description())
                )
                n += inputs.shape[0]
            self.logger.info(f"{n} data processed.")
            self.logger.info("Epoch finished !")
            if train_metrics:
                self.logger.info("Other metrics on Training set.")
                for metric in train_metrics:
                    self.logger.info("{}".format(metric.description()))
            for metric in train_metrics:
                self.tensor_board.add_scalar(
                    "other_metrics/training/{}".format(metric.document()), metric.avg(), epoch
                )

            val_metrics = self.eval(self.loss.new(), *self.other_validation_metrics, datasets=self.validation_sets)

            # train_metrics = self.eval(*self.other_validation_metrics, datasets=self.training_sets)

            self.logger.info("Validation loss: {}".format(val_metrics[0].description()))
            if val_metrics[1:]:
                self.logger.info("Other metrics on validation set.")
                for metric in val_metrics[1:]:
                    self.logger.info("{}".format(metric.description()))
            self.tensor_board.add_scalar("loss/training/{}".format(self.loss.document()), self.loss.log(), epoch)
            self.tensor_board.add_scalar(
                "loss/validation/loss_{}".format(val_metrics[0].document()), val_metrics[0].log(), epoch
            )
            for metric in val_metrics[1:]:
                self.tensor_board.add_scalar(
                    "other_metrics/validation/{}".format(metric.document()), metric.avg(), epoch
                )

            # eval extra validation sets
            if self.extra_validation_sets:
                for val in self.extra_validation_sets:
                    val_metrics = self.eval(
                        self.loss.new(), *self.other_validation_metrics, datasets=[val]
                    )
                    self.logger.info(
                        "Extra Validation loss on dataset {}: {}".format(val.name, val_metrics[0].description())
                    )
                    if val_metrics[1:]:
                        self.logger.info("Other metrics on extra validation set.")
                        for metric in val_metrics[1:]:
                            self.logger.info("{}".format(metric.description()))
                    self.tensor_board.add_scalar(
                        "loss/extra_validation_{}/loss_{}".format(val.name, val_metrics[0].document()),
                        val_metrics[0].log(), epoch
                    )
                    for metric in val_metrics[1:]:
                        self.tensor_board.add_scalar(
                            "other_metrics/extra_validation_{}/{}".format(val.name, metric.document()),
                            metric.avg(), epoch
                        )

            checkpoint_dir = self.config.experiment_dir.joinpath("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_path = checkpoint_dir.joinpath("CP_{}.pth".format(epoch))
            torch.save(
                {
                    "epoch": epoch,
                    "model": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                str(output_path)
            )
            self.logger.info("Checkpoint {} saved at {}!".format(epoch, str(output_path)))
            if self.inference_func is not None and epoch % 100 == 0:
                self.inference(epoch)

    def inference(self, epoch: int):
        output_dir = self.config.experiment_dir.joinpath("inference").joinpath("CP_{}".format(epoch))
        for tr in self.training_sets:
            indices = np.random.choice(len(tr.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = tr.image_paths[idx]
                label_path = tr.label_paths[idx]

                image = tr.read_image(tr.image_paths[idx])
                label = tr.get_label_tensor_from_index(idx)

                self.logger.info("Inferencing for {} dataset, image {}.".format(tr.name, idx))
                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(tr.name, "training", image_path.parent.stem),
                    self.config.gpu, self.config.device, tr.crop_size, tr.voxel_size
                )

        for val in self.validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]

                image = val.read_image(val.image_paths[idx])
                label = val.get_label_tensor_from_index(idx)
                # image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(val.name, "validation", image_path.parent.stem),
                    self.config.gpu, self.config.device, val.crop_size, val.voxel_size
                )
        for val in self.extra_validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]

                image = val.read_image(val.image_paths[idx])
                label = val.get_label_tensor_from_index(idx)
                # image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(val.name, "extra_val", image_path.parent.stem),
                    self.config.gpu, self.config.device, val.crop_size, val.voxel_size
                )
