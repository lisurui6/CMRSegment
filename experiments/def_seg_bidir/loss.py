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

        self.pred_maps_bce_loss = BCELoss(logit=False)
        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.weights = weights
        self.deform_mse_loss = MSELoss()

        self.label_dice_loss = DiceLoss()

        # self.label_bce_loss = BCELoss(logit=False)
        # self.template_bce_loss = BCELoss(logit=False)
        self.warped_image_loss = MSELoss()
        self.warped_template_image_loss = MSELoss()

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        """predicted = (warped template, warped maps, pred maps, flow, warped image, warped template image)"""
        label, template, image, template_image = outputs

        bce_loss = self.pred_maps_bce_loss.cumulate(predicted[2], label)
        grad_loss = self.grad_loss.cumulate(predicted[3], None)
        deform_loss = self.deform_mse_loss.cumulate(predicted[3], torch.zeros(predicted[3].shape).cuda())

        label_dice_loss = self.label_dice_loss.cumulate(predicted[0], label)

        warped_image_loss = self.warped_image_loss.cumulate(predicted[4], template_image)
        warped_template_image_loss = self.warped_template_image_loss.cumulate(predicted[5], image)

        loss = bce_loss * self.weights[0] + grad_loss * self.weights[1] + deform_loss * self.weights[2] + \
            label_dice_loss * self.weights[3] + warped_image_loss * self.weights[4] + \
            warped_template_image_loss * self.weights[5]
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
        return "{}, {}, deform {}, label {}".format(
            self.pred_maps_bce_loss.description(), self.grad_loss.description(), self.deform_mse_loss.description(),
            self.label_dice_loss.description()
        )

    def reset(self):
        super().reset()
        self.label_mse_loss.reset()
        self.template_mse_loss.reset()
        self.pred_maps_bce_loss.reset()
        self.grad_loss.reset()
        self.deform_mse_loss.reset()
        self.label_dice_loss.reset()
        self.template_dice_loss.reset()
        # self.label_bce_loss.reset()
        # self.template_bce_loss.reset()


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
