import torch
from CMRSegment.nn.torch.data import Torch2DSegmentationDataset, construct_training_validation_dataset, TorchDataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader

from CMRSegment.nn.torch import prepare_tensors
from CMRSegment.nn.torch.loss import TorchLoss
from torch.optim.optimizer import Optimizer
from CMRSegment.config import ExperimentConfig
from tqdm import tqdm
from typing import Iterable, Union, Tuple, List
from torch.utils.tensorboard import SummaryWriter
import logging
from argparse import ArgumentParser
from torch.utils.data.dataset import Dataset

logging.basicConfig(level=logging.INFO)


class Experiment:
    def __init__(
        self,
        config: ExperimentConfig,
        network: torch.nn.Module,
        training_set: TorchDataset,
        validation_set: TorchDataset,
        loss: TorchLoss,
        optimizer: Optimizer,
        other_validation_metrics: List = None,
        tensor_board: SummaryWriter = None,
        logger=None,
    ):
        self.config = config
        self.network = network
        self.training_set = training_set
        self.validation_set = validation_set
        self.optimizer = optimizer
        self.loss = loss
        self.other_validation_metrics = other_validation_metrics if other_validation_metrics is not None else []
        self.tensor_board = tensor_board or SummaryWriter(str(self.config.experiment_dir.joinpath("tb_runs")))
        self.logger = logger or logging.getLogger("CMRSegment.nn.torch.Experiment")
        self.set_device()

    def set_device(self):
        if self.config.gpu:
            if self.config.device is None or isinstance(self.config.device, str):
                device = 0
            else:
                device = self.config.device
        else:
            device = "cpu"
        if self.config.gpu:
            self.network.cuda(device=device)
        return self

    def train(self):
        self.network.train()
        train_data_loader = self.training_set.random_loader(self.config.batch_size)
        for epoch in range(self.config.num_epochs):
            self.network.train()
            self.logger.info("Starting epoch {}/{}".format(epoch, self.config.num_epochs))
            self.loss.reset()
            pbar = tqdm(enumerate(train_data_loader))
            for idx, (inputs, outputs) in pbar:
                inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                predicted = self.network(inputs)
                loss = self.loss.cumulate(predicted, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description("{} --- {}".format((idx + 1) / len(train_data_loader), self.loss.description()))
            self.logger.info("Epoch finished !")
            metrics = self.eval(self.loss.new(), *self.other_validation_metrics)
            self.logger.info("Validation loss: {}".format(metrics[0].description()))
            if metrics[1:]:
                self.logger.info("Other metrics on validation set.")
                for metric in metrics[1:]:
                    self.logger.info("{}".format(metric.description()))
            self.tensor_board.add_scalar("loss/training/{}".format(self.loss.document()), self.loss.log(), epoch)
            self.tensor_board.add_scalar(
                "loss/validation/loss_{}".format(metrics[0].document()), metrics[0].log(), epoch
            )
            for metric in metrics[1:]:
                self.tensor_board.add_scalar("other_metrics/{}".format(metric.document()), metric.avg(), epoch)

            checkpoint_dir = self.config.experiment_dir.joinpath("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_path = checkpoint_dir.joinpath("CP_{}.pth".format(epoch))
            torch.save(self.network.state_dict(), str(output_path))
            self.logger.info("Checkpoint {} saved at {}!".format(epoch, str(output_path)))

    def eval(self, *metrics: TorchLoss) -> Tuple[TorchLoss]:
        """Evaluate on validation set with training loss function if none provided"""
        data_loader = self.validation_set.sequential_loader(self.config.batch_size)
        self.network.eval()
        if not isinstance(metrics, Iterable) and isinstance(metrics, TorchLoss):
            metrics = [metrics]
        for metric in metrics:
            metric.reset()
        for idx, (inputs, outputs) in enumerate(data_loader):
            inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
            preds = self.network(inputs)
            for metric in metrics:
                preds = prepare_tensors(preds, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                metric.cumulate(preds, outputs)
        return metrics

    @staticmethod
    def parser() -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("-i", "--datasets", dest="dataset_names", required=True, nargs="+", type=str)
        parser.add_argument("--mount-prefix", dest="mount_prefix", type=str, required=True)
        parser.add_argument("-o", "--experiment-dir", dest="experiment_dir", type=str, default=None)
        parser.add_argument("--data-mode", dest="data_mode", type=str, default="2D")

        parser.add_argument("-v", "--validation-split", dest="validation_split", type=float, default=0.2)
        parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=10)
        parser.add_argument("-n", "--num-epochs", dest="num_epochs", type=int, default=10)
        parser.add_argument("-lr", dest="learning_rate", type=float, default=0.0001)
        parser.add_argument("-g", "--gpu", action="store_true", dest="gpu", default=False, help="use cuda")
        parser.add_argument("--device", dest="device", type=int, default=0)
        return parser
