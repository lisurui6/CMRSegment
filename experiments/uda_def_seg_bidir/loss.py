from typing import Union, Iterable

import torch
from CMRSegment.common.nn.torch.loss import TorchLoss, MSELoss, BCELoss, DiceCoeff, DiceLoss
from experiments.def_seg_bidir.loss import Grad
from typing import List


class UDADefSegLoss(TorchLoss):
    def __init__(self, weights: List[float], penalty="l2", loss_mult=None):
        super().__init__()
        self.template_dice_loss = DiceLoss()
        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.deform_mse_loss = MSELoss()
        self.weights = weights

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        """predicted = (warped template, warped maps, pred maps, flow)"""
        __, template = outputs

        template_dice_loss = self.template_dice_loss.cumulate(predicted[1], template)
        grad_loss = self.grad_loss.cumulate(predicted[3], None)
        deform_loss = self.deform_mse_loss.cumulate(predicted[3], torch.zeros(predicted[3].shape).cuda())
        loss = template_dice_loss * self.weights[0] + grad_loss * self.weights[1] + deform_loss * self.weights[2]

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
        return "template {}, grad {}, deform {}".format(
            self.template_dice_loss.description(), self.grad_loss.description(), self.deform_mse_loss.description()
        )

    def reset(self):
        super().reset()
        self.template_dice_loss.reset()
        self.grad_loss.reset()
        self.deform_mse_loss.reset()
