import numpy as np
from CMRSegment.common.nn.torch.experiment import Experiment, ExperimentConfig, Optimizer, SummaryWriter, Callable
import torch
from CMRSegment.common.nn.torch.data import MultiDataLoader
from CMRSegment.common.nn.torch import prepare_tensors
from experiments.def_seg_bidir_affine_encode_atlas_update.atlas import Atlas
from tqdm import tqdm
from datetime import datetime

from CMRSegment.common.nn.torch.loss import TorchLoss
from CMRSegment.common.nn.torch.data import TorchDataset, random_crop
from CMRSegment.common.nn.torch.augmentation import soi_crop
from typing import Iterable, Tuple, List


class DefSegExperiment(Experiment):
    def update_atlas(self, train_data_loader, atlas_label, atlas, eta: float = 0.01):
        pbar = tqdm(enumerate(train_data_loader))
        warped_labels = None
        warped_images = None
        n = 0
        for idx, (inputs, outputs) in pbar:
            inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
            predicted = self.network(inputs, atlas_label)  # pass updated atlas in
            warped_label = torch.sum(predicted[-4], dim=0).cpu().detach().numpy()
            image = np.squeeze(torch.sum(predicted[-5], dim=0).cpu().detach().numpy(), axis=0)
            n += predicted[-5].shape[0]
            if warped_images is None:
                warped_images = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
            if warped_labels is None:
                warped_labels = np.zeros((warped_label.shape[0], warped_label.shape[1], warped_label.shape[2], warped_label.shape[3]))
            warped_images += image
            warped_labels += warped_label
        print(n)

        mean_image = warped_images / n
        mean_label = warped_labels / n
        atlas.update(mean_image, mean_label, eta=eta)
        return atlas

    def train(self, starting_epoch: int = 0, atlas: Atlas = None, atlas_eta: float = 0.01):
        self.network.train()
        train_data_loader = MultiDataLoader(
            *self.training_sets,
            batch_size=self.config.batch_size,
            sampler_cls="random",
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        set = False
        print(self.network)
        if atlas is None:
            atlas = Atlas.from_data_loader(train_data_loader)
            atlas.save(output_dir=self.config.experiment_dir.joinpath("atlas").joinpath("init"))
        real_weights = self.loss.weights
        for epoch in range(self.config.num_epochs):
            if epoch < starting_epoch:
                continue
            self.network.train()
            self.logger.info("{}: starting epoch {}/{}".format(datetime.now(), epoch, self.config.num_epochs))
            self.loss.reset()
            if epoch > 7 and not set:
                self.optimizer.param_groups[0]['lr'] /= 10
                print("-------------Learning rate: {}-------------".format(self.optimizer.param_groups[0]['lr']))
                set = True
                self.loss.weights = real_weights
            else:
                self.loss.weights = [1, 0, 0, 0, 0, 0, 0, 0, 0]

            pbar = tqdm(enumerate(train_data_loader))
            atlas_label = prepare_tensors(torch.from_numpy(atlas.label()).float(), self.config.gpu, self.config.device)
            self.network.update_batch_atlas(atlas_label)

            for idx, (inputs, outputs) in pbar:
                inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                predicted = self.network(inputs, atlas_label)  # pass updated atlas in
                loss = self.loss.cumulate(predicted, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    "{:.2f} --- {}".format((idx + 1) / len(train_data_loader), self.loss.description())
                )
            # update atlas
            self.network.eval()
            atlas = self.update_atlas(train_data_loader, atlas_label, atlas, eta=atlas_eta)
            atlas.save(output_dir=self.config.experiment_dir.joinpath("atlas").joinpath("epoch_{}".format(epoch)))
            self.logger.info("Epoch finished !")
            val_metrics = self.eval(self.loss.new(), *self.other_validation_metrics, datasets=self.validation_sets,
                                    atlas=atlas_label)
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
                        self.loss.new(), *self.other_validation_metrics, datasets=[val], atlas=atlas_label
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
            torch.save(self.network.state_dict(), str(output_path))
            self.logger.info("Checkpoint {} saved at {}!".format(epoch, str(output_path)))
            if self.inference_func is not None:
                self.inference(epoch, atlas_label)

    def inference(self, epoch: int, atlas):
        output_dir = self.config.experiment_dir.joinpath("inference").joinpath("CP_{}".format(epoch))
        for val in self.validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.read_image(val.image_paths[idx], val.feature_size, val.n_slices)
                label = val.read_label(val.label_paths[idx], val.feature_size, val.n_slices)

                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, self.network, output_dir.joinpath(val.name, image_path.parent.stem), atlas,
                    (val.n_slices, val.feature_size, val.feature_size), self.config.gpu, self.config.device,
                )
        for val in self.extra_validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.read_image(val.image_paths[idx], val.feature_size, val.n_slices)
                label = val.read_label(val.label_paths[idx], val.feature_size, val.n_slices)

                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, self.network, output_dir.joinpath(val.name, image_path.parent.stem), atlas,
                    (val.n_slices, val.feature_size, val.feature_size), self.config.gpu, self.config.device
                )

    def eval(self, *metrics: TorchLoss, datasets: List[TorchDataset], atlas) -> Tuple[TorchLoss]:
        """Evaluate on validation set with training loss function if none provided"""
        val_data_loader = MultiDataLoader(
            *datasets,
            batch_size=self.config.batch_size,
            sampler_cls="sequential",
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        self.network.eval()
        if not isinstance(metrics, Iterable) and isinstance(metrics, TorchLoss):
            metrics = [metrics]
        for metric in metrics:
            metric.reset()
        for idx, (inputs, outputs) in enumerate(val_data_loader):
            inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
            preds = self.network(inputs, atlas)
            for metric in metrics:
                preds = prepare_tensors(preds, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                metric.cumulate(preds, outputs)
        return metrics
