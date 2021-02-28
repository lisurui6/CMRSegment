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

        self.pred_maps_bce_loss = BCELoss(logit=False)
        self.pred_maps_dice_loss = DiceLoss()
        self.pred_maps_mse_loss = MSELoss()

        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.deform_mse_loss = MSELoss()

        self.label_dice_loss = DiceLoss()
        self.label_mse_loss = MSELoss()

        self.template_dice_loss = DiceLoss()
        self.template_mse_loss = MSELoss()

        # self.label_bce_loss = BCELoss(logit=False)
        # self.template_bce_loss = BCELoss(logit=False)
        self.epoch = 0

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        """predicted = (warped template, warped maps, pred maps, flow)"""
        label, template = outputs
        if self.epoch <= 10:
            weights = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            weights = self.weights

        pred_map_bce_loss = self.pred_maps_bce_loss.cumulate(predicted[2], label)
        pred_map_dice_loss = self.pred_maps_dice_loss.cumulate(predicted[2], label)
        pred_map_mse_loss = self.pred_maps_mse_loss.cumulate(predicted[2], label)
        pred_map_loss = weights[0] * pred_map_bce_loss + weights[1] * pred_map_dice_loss + weights[2] * pred_map_mse_loss

        grad_loss = self.grad_loss.cumulate(predicted[3], None)
        deform_loss = self.deform_mse_loss.cumulate(predicted[3], torch.zeros(predicted[3].shape).cuda())

        label_dice_loss = self.label_dice_loss.cumulate(predicted[0], label)
        label_mse_loss = self.label_mse_loss.cumulate(predicted[0], label)
        label_loss = weights[3] * label_dice_loss + weights[4] * label_mse_loss

        template_dice_loss = self.template_dice_loss.cumulate(predicted[1], template)
        template_mse_loss = self.template_mse_loss.cumulate(predicted[1], template)
        template_loss = weights[5] * template_dice_loss + weights[6] * template_mse_loss

        loss = pred_map_loss + label_loss + template_loss + grad_loss * self.weights[7] + deform_loss * self.weights[8]
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
        return "total {:.4f}, pred map {}, pred map {}, pred map {}, label {}, label {}, grad {}, deform {}, ".format(
            self.log(),
            self.pred_maps_bce_loss.description(), self.pred_maps_dice_loss.description(), self.pred_maps_mse_loss.description(),
            self.label_dice_loss.description(), self.label_mse_loss.description(),
            # self.template_dice_loss.description(), self.template_mse_loss.description(),
            self.grad_loss.description(), self.deform_mse_loss.description(),
        )

    def reset(self):
        super().reset()
        self.epoch += 1
        self.pred_maps_bce_loss.reset()
        self.pred_maps_mse_loss.reset()
        self.pred_maps_dice_loss.reset()

        self.grad_loss.reset()
        self.deform_mse_loss.reset()
        self.label_dice_loss.reset()
        self.label_mse_loss.reset()
        self.template_mse_loss.reset()
        self.template_dice_loss.reset()


class DefSegWarpedTemplateDice(DiceCoeff, ABC):
    def forward(self, input, target):
        label, template = target
        # input = (warped template, warped maps, pred maps, flow)
        pred = (input[0] > 0.5).float()
        return super().forward(pred, label)


class DefSegPredDice(DiceCoeff):
    def forward(self, input, target):
        label, template = target

        # input = (warped template, warped maps, pred maps, flow)
        pred = (input[2] > 0.5).float()
        return super().forward(pred, label)


class DefSegWarpedMapsDice(DiceCoeff):
    def forward(self, input, target):
        label, template = target
        # input = (warped template, warped maps, pred maps, flow)
        pred = (input[1] > 0.5).float()
        return super().forward(pred, template)


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
