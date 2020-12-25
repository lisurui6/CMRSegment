import math
from abc import ABC
from typing import Union, Iterable

import torch
import torch.nn.functional as F
from CMRSegment.common.nn.torch.loss import TorchLoss, MSELoss, BCELoss, DiceCoeff
import numpy as np


class DefSegLoss(TorchLoss):
    def __init__(self, template, penalty="l2", loss_mult=None, weight=0.01):
        super().__init__()
        self.label_mse_loss = MSELoss()
        self.template_mse_loss = MSELoss()
        self.bce_loss = BCELoss(logit=False)
        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.weight = weight
        if isinstance(template, np.ndarray):
            self.template = torch.from_numpy(template).float().cuda().unsqueeze(0)
        else:
            self.template = template

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        # predicted = (warped template, warped maps, pred maps, flow)

        label_mse_loss = self.label_mse_loss.cumulate(predicted[0], outputs)
        template_mse_loss = self.template_mse_loss.cumulate(predicted[1], self.template)

        bce_loss = self.bce_loss.cumulate(predicted[2], outputs)
        grad_loss = self.grad_loss.cumulate(predicted[3], None)
        loss = label_mse_loss + bce_loss + grad_loss * self.weight + template_mse_loss * 0.1
        self._cum_loss += loss.item()
        self._count += 1
        return loss

    def new(self):
        new_loss = self.__class__(
            template=self.template, penalty=self.grad_loss.penalty,
            loss_mult=self.grad_loss.loss_mult, weight=self.weight
        )
        new_loss.reset()
        return new_loss

    def description(self):
        return "{}, {}, {}, {}".format(
            self.label_mse_loss.description(), self.template_mse_loss.description(),
            self.bce_loss.description(), self.grad_loss.description()
        )

    def reset(self):
        super().reset()
        self.label_mse_loss.reset()
        self.template_mse_loss.reset()
        self.bce_loss.reset()
        self.grad_loss.reset()


class DefSegWarpedTemplateDice(DiceCoeff, ABC):
    def __init__(self, template):
        if isinstance(template, np.ndarray):
            self.template = torch.from_numpy(template).float().cuda().unsqueeze(0)
        else:
            self.template = template
        super().__init__()

    def forward(self, input, target):
        # predicted = (warped template, warped maps, pred maps, flow)
        pred = (input[0] > 0.5).float()
        return super().forward(pred, target)

    def new(self):
        new_loss = self.__class__(template=self.template)
        new_loss.reset()
        return new_loss


class DefSegPredDice(DiceCoeff):
    def forward(self, input, target):
        # predicted = (warped template, warped maps, pred maps, flow)
        pred = (input[2] > 0.5).float()
        return super().forward(pred, target)


class DefSegWarpedMapsDice(DiceCoeff):
    def __init__(self, template):
        if isinstance(template, np.ndarray):
            self.template = torch.from_numpy(template).float().cuda().unsqueeze(0)
        else:
            self.template = template
        super().__init__()

    def forward(self, input, target):
        # predicted = (warped template, warped maps, pred maps, flow)
        pred = (input[1] > 0.5).float()
        return super().forward(pred, self.template.contiguous())

    def new(self):
        new_loss = self.__class__(template=self.template)
        new_loss.reset()
        return new_loss


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
