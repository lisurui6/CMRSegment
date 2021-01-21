
import torch
import numpy as np
from CMRSegment.common.nn.torch import prepare_tensors
from CMRSegment.common.nn.torch.experiment import Experiment
from CMRSegment.common.nn.torch.data import MultiDataLoader
from datetime import datetime
from tqdm import tqdm


class DefSegExperiment(Experiment):

    def train(self):
        self.network.train()
        train_data_loader = MultiDataLoader(
            *self.training_sets,
            batch_size=self.config.batch_size,
            sampler_cls="random",
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        set = False
        for epoch in range(self.config.num_epochs):
            self.network.train()
            self.logger.info("{}: starting epoch {}/{}".format(datetime.now(), epoch, self.config.num_epochs))
            self.loss.reset()
            if epoch > 10 and not set:
                self.optimizer.param_groups[0]['lr'] /= 10
                print("-------------Learning rate: {}-------------".format(self.optimizer.param_groups[0]['lr']))
                set = True

            # train loop
            pbar = tqdm(enumerate(train_data_loader))
            for idx, (inputs, outputs) in pbar:
                inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                predicted = self.network(inputs)
                loss = self.loss.cumulate(predicted, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    "{:.2f} --- {}".format((idx + 1) / len(train_data_loader), self.loss.description())
                )
            self.logger.info("Epoch finished !")
            val_metrics = self.eval(self.loss.new(), *self.other_validation_metrics, datasets=self.validation_sets)
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
            torch.save(self.network.state_dict(), str(output_path))
            self.logger.info("Checkpoint {} saved at {}!".format(epoch, str(output_path)))
            if self.inference_func is not None:
                self.inference(epoch)

    def inference(self, epoch: int):
        output_dir = self.config.experiment_dir.joinpath("inference").joinpath("CP_{}".format(epoch))
        for val in self.validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.get_image_tensor_from_index(idx)
                label = val.get_label_tensor_from_index(idx)
                # template = val.get_label_tensor_from_index(np.random.randint(0, len(val)))
                template = val.template
                template = torch.from_numpy(template).float()
                template = torch.unsqueeze(template, 0)
                template = prepare_tensors(template, self.config.gpu, self.config.device)

                # template_image = val.template_image
                # template_image = np.expand_dims(template_image, 0)
                # template_image = torch.from_numpy(template_image).float()
                # template_image = torch.unsqueeze(template_image, 0)
                # template_image = prepare_tensors(template_image, self.config.gpu, self.config.device)

                image = torch.unsqueeze(image, 0)
                image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    (image, template), label, image_path, self.network, output_dir.joinpath(val.name, image_path.parent.stem)
                )
        for val in self.extra_validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.get_image_tensor_from_index(idx)
                label = val.get_label_tensor_from_index(idx)
                # template = val.get_label_tensor_from_index(np.random.randint(0, len(val)))
                template = val.template
                template = torch.from_numpy(template).float()
                template = torch.unsqueeze(template, 0)
                template = prepare_tensors(template, self.config.gpu, self.config.device)

                # template_image = val.template_image
                # template_image = np.expand_dims(template_image, 0)
                # template_image = torch.from_numpy(template_image).float()
                # template_image = torch.unsqueeze(template_image, 0)
                # template_image = prepare_tensors(template_image, self.config.gpu, self.config.device)

                image = torch.unsqueeze(image, 0)
                image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    (image, template), label, image_path, self.network,
                    output_dir.joinpath(val.name, image_path.parent.stem),
                )
