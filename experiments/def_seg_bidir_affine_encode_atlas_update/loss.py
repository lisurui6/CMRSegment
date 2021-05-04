from abc import ABC
from typing import Union, Iterable

import torch
from CMRSegment.common.nn.torch.loss import TorchLoss, MSELoss, BCELoss, DiceCoeff, DiceLoss
from typing import List


class DefLoss(TorchLoss):
    def __init__(self, weights: List[float], penalty="l2", loss_mult=None):
        super().__init__()
        self.weights = weights

        self.pred_maps_mse_loss = MSELoss()

        self.grad_loss = Grad(penalty=penalty, loss_mult=loss_mult)
        self.deform_mse_loss = MSELoss()

        self.label_mse_loss = MSELoss()
        self.label_mse_loss_affine = MSELoss()

        self.atlas_mse_loss = MSELoss()
        self.atlas_mse_loss_affine = MSELoss()

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        """predicted = affine_warped_template 0, warped_template 1, pred_maps 2, preint_flow 3, warped_image 4,
        warped_label 5, affine_warped_label 6, batch_atlas 7"""
        label = outputs
        # weights = self.weights
        if self.epoch <= 7:
            weights = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            weights = self.weights

        grad_loss = self.grad_loss.cumulate(predicted[3], None)
        deform_loss = self.deform_mse_loss.cumulate(predicted[3], torch.zeros(predicted[3].shape).cuda())

        pred_map_mse_loss = self.pred_maps_mse_loss.cumulate(predicted[2], label)
        pred_map_loss = weights[0] * pred_map_mse_loss

        affine_label_mse_loss = self.label_mse_loss_affine.cumulate(predicted[0], label)
        affine_label_loss = weights[1] * affine_label_mse_loss

        label_mse_loss = self.label_mse_loss.cumulate(predicted[1], label)
        label_loss = weights[1] * label_mse_loss + affine_label_loss

        atlas_mse_loss = self.atlas_mse_loss.cumulate(predicted[5], predicted[7])
        atlas_mse_loss_affine = self.atlas_mse_loss_affine.cumulate(predicted[6], predicted[7])
        atlas_loss = atlas_mse_loss * weights[4] + atlas_mse_loss_affine * weights[4]
        loss = label_loss + grad_loss * weights[2] + deform_loss * weights[3] + pred_map_loss + atlas_loss

        # loss = label_loss
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
        return "total {:.4f}, pred map {}, label {}, grad {}, deform {}, ".format(
            self.log(),
            self.pred_maps_mse_loss.description(),
            self.label_mse_loss.description(),
            self.grad_loss.description(), self.deform_mse_loss.description(),
        )

    def reset(self):
        super().reset()
        self.pred_maps_mse_loss.reset()

        self.grad_loss.reset()
        self.deform_mse_loss.reset()
        self.label_mse_loss.reset()
        self.label_mse_loss_affine.reset()
        self.atlas_mse_loss.reset()
        self.atlas_mse_loss_affine.reset()


class DefWarpedTemplateDice(DiceCoeff, ABC):
    def forward(self, input, target):
        label = target
        """predicted = affine_warped_template, warped_template, pred_maps, preint_flow, warped_image, warped_label, 
        affine_warped_label, batch_atlas"""
        pred = (input[1] > 0.5).float()
        return super().forward(pred, label)


class DefAffineWarpedTemplateDice(DiceCoeff, ABC):
    def forward(self, input, target):
        label = target
        """predicted = affine_warped_template, warped_template, pred_maps, preint_flow, warped_image, warped_label,
         affine_warped_label, batch_atlas"""
        pred = (input[0] > 0.5).float()
        return super().forward(pred, label)


class DefPredDice(DiceCoeff):
    def forward(self, input, target):
        label = target
        """predicted = affine_warped_template, warped_template, pred_maps, preint_flow, warped_image, warped_label,
         affine_warped_label, batch_atlas"""
        pred = (input[2] > 0.5).float()
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
