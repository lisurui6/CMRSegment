import math
from abc import ABC
from typing import Union, Iterable

import torch
import torch.nn.functional as F
from CMRSegment.common.nn.torch.loss import TorchLoss, MSELoss, BCELoss, DiceCoeff, DiceLoss
import numpy as np
from typing import List


class DefSegLoss(TorchLoss):
    def __init__(self, weights: List[float], penalty="l2", loss_mult=None):
        super().__init__()
        self.weights = weights

        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.deform_mse_loss = MSELoss()

        self.label_dice_loss = DiceLoss()
        self.label_mse_loss = MSELoss()


    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        """predicted = (warped template, flow)"""
        label, template = outputs
        weights = self.weights

        grad_loss = self.grad_loss.cumulate(predicted[1], None)
        deform_loss = self.deform_mse_loss.cumulate(predicted[1], torch.zeros(predicted[1].shape).cuda())

        label_dice_loss = self.label_dice_loss.cumulate(predicted[0], label)
        label_mse_loss = self.label_mse_loss.cumulate(predicted[0], label)
        label_loss = weights[0] * label_dice_loss + weights[1] * label_mse_loss

        loss = label_loss + grad_loss * self.weights[2] + deform_loss * self.weights[3]
        self._cum_loss += loss.item()
        self._count += 1
        return loss

    def new(self):
        new_loss = self.__class__(
            penalty=self.grad_loss.penalty,
            loss_mult=self.grad_loss.loss_mult, weights=self.weights
        )
        new_loss.reset()
        return new_loss

    def description(self):
        return "total {:.4f}, label {}, label {}, grad {}, deform {}, ".format(
            self.log(),
            self.label_dice_loss.description(), self.label_mse_loss.description(),
            self.grad_loss.description(), self.deform_mse_loss.description(),
        )

    def reset(self):
        super().reset()
        self.grad_loss.reset()
        self.deform_mse_loss.reset()
        self.label_dice_loss.reset()
        self.label_mse_loss.reset()


class DefSegWarpedTemplateDice(DiceCoeff, ABC):
    def forward(self, input, target):
        label, template = target
        # input = (warped template, flow)
        pred = (input[0] > 0.5).float()
        return super().forward(pred, label)


class Grad(TorchLoss):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        super().__init__()

    def forward(self, y_pred, _):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
