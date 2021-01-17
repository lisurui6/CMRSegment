import math
from typing import Union, Iterable

import torch
import torch.nn.functional as F
from CMRSegment.common.nn.torch.loss import TorchLoss, MSELoss, BCELoss, DiceCoeff
import numpy as np


class DefSegLoss(TorchLoss):
    def __init__(self, template, penalty="l2", loss_mult=None, weight=0.01):
        super().__init__()
        self.mse_loss = MSELoss()
        self.bce_loss = BCELoss(logit=True)
        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.weight = weight
        if isinstance(template, np.ndarray):
            self.template = torch.from_numpy(template).float().cuda().unsqueeze(0)
        else:
            self.template = template

    # def cumulate(
    #     self,
    #     predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
    #     outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    # ):
    #     # predicted = (warped_maps, pred maps, flow)
    #
    #     mse_loss = self.mse_loss.cumulate(predicted[0], self.template)
    #     bce_loss = self.bce_loss.cumulate(predicted[1], outputs)
    #     grad_loss = self.grad_loss.cumulate(predicted[2], None)
    #     loss = mse_loss * 0.5 + bce_loss * 0.5 + grad_loss * self.weight
    #     self._cum_loss += loss.item()
    #     self._count += 1
    #     return loss

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        # predicted = (warped template, pred maps, flow)

        mse_loss = self.mse_loss.cumulate(predicted[0], outputs)
        # bce_loss = self.bce_loss.cumulate(predicted[1], outputs)
        grad_loss = self.grad_loss.cumulate(predicted[2], None)
        loss = mse_loss + grad_loss * self.weight
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

    # def description(self):
    #     return "{}, {}, {}".format(self.mse_loss.description(), self.bce_loss.description(), self.grad_loss.description())

    def description(self):
        return "{}, {}".format(self.mse_loss.description(), self.grad_loss.description())

    def reset(self):
        super().reset()
        self.mse_loss.reset()
        self.bce_loss.reset()
        self.grad_loss.reset()


class DefSegWarpedDice(DiceCoeff):
    def __init__(self, template):
        if isinstance(template, np.ndarray):
            self.template = torch.from_numpy(template).float().cuda().unsqueeze(0)
        else:
            self.template = template
        super().__init__()

    def forward(self, input, target):
        # input = (warped template, pred maps, flow)
        return super().forward(input[0], target)

    # def forward(self, input, _):
    #     # input = (warped maps, pred maps, flow)
    #     return super().forward(input[0], self.template.contiguous())

    def new(self):
        new_loss = self.__class__(template=self.template)
        new_loss.reset()
        return new_loss


class DefSegPredDice(DiceCoeff):
    def forward(self, input, target):
        # input = (warped template, pred maps)
        return super().forward(input[1], target)


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
