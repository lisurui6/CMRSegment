import torch
import torch.nn.functional as F

from CMRSegment.common.nn.torch.loss import TorchLoss


class Grad3D(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        super().__init__()

    def forward(self, y_pred):
        # y_pred (B, 3, W, H, D)
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


class ShapeDeformLoss(TorchLoss):
    def __init__(self, flow_lambda: int = 100):
        super().__init__()
        self.flow_lambda = flow_lambda
        self.flow_grad_loss = Grad3D()

    def forward(self, predicted, label):
        # label: (B, 3, H(z), W(x), D(y))
        label = torch.movedim(label, 2, -1)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicted
        loss = (label[:, 0] - init_mask0).pow(2).mean()
        loss += (label[:, 1] - init_mask1).pow(2).mean()
        loss += (label[:, 2] - init_mask2).pow(2).mean()
        loss += (label[:, 0] - affine_mask0).pow(2).mean()
        loss += (label[:, 1] - affine_mask1).pow(2).mean()
        loss += (label[:, 2] - affine_mask2).pow(2).mean()
        loss += (label[:, 0] - deform_mask0).pow(2).mean()
        loss += (label[:, 1] - deform_mask1).pow(2).mean()
        loss += (label[:, 2] - deform_mask2).pow(2).mean()
        loss += self.flow_lambda * self.flow_grad_loss(flow)
        return loss

    def new(self):
        """Copy and reset obj"""
        new_loss = self.__class__(flow_lambda=self.flow_lambda)
        new_loss.reset()
        return new_loss


class DiceCoeff(TorchLoss):
    def forward(self, predicted, target):
        # predicted: (B, 1, W, D, H), target: (B, H, W, D)
        target = torch.movedim(target, 1, -1)
        eps = 0.0001
        m1 = predicted.view(predicted.shape[0], -1).float()
        m2 = target.reshape(target.shape[0], -1).float()
        inter = (m1 * m2).sum().float()
        union = m1.sum() + m2.sum()
        return (2 * inter + eps) / (union + eps)


class InitLVDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(init_mask0, target[:, 0])

    def description(self):
        return "Init LV Dice: {:.4f}".format(self.avg())


class InitLVMyoDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(init_mask1, target[:, 1])

    def description(self):
        return "Init LV Myo Dice: {:.4f}".format(self.avg())


class InitRVDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(init_mask2, target[:, 2])

    def description(self):
        return "Init RV Dice: {:.4f}".format(self.avg())


class InitAllDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts
        init_mask = torch.cat([init_mask0, init_mask1, init_mask2], dim=1)
        return super().forward(init_mask, target)

    def description(self):
        return "Init All Dice: {:.4f}".format(self.avg())


class AffineLVDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(affine_mask0, target[:, 0])

    def description(self):
        return "Affine LV Dice: {:.4f}".format(self.avg())


class AffineLVMyoDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(affine_mask1, target[:, 1])

    def description(self):
        return "Affine LV Myo Dice: {:.4f}".format(self.avg())


class AffineRVDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(affine_mask2, target[:, 2])

    def description(self):
        return "Affine RV Dice: {:.4f}".format(self.avg())


class AffineAllDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts
        affine_mask = torch.cat([affine_mask0, affine_mask1, affine_mask2], dim=1)
        return super().forward(affine_mask, target)

    def description(self):
        return "Affine All Dice: {:.4f}".format(self.avg())


class DeformLVDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(deform_mask0, target[:, 0])

    def description(self):
        return "Deform LV Dice: {:.4f}".format(self.avg())


class DeformLVMyoDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(deform_mask1, target[:, 1])

    def description(self):
        return "Deform LV Myo Dice: {:.4f}".format(self.avg())


class DeformRVDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts

        return super().forward(deform_mask2, target[:, 2])

    def description(self):
        return "Deform RV Dice: {:.4f}".format(self.avg())


class DeformAllDiceCoeff(DiceCoeff):
    def forward(self, predicts, target):
        # target: (B, 3, H, W, D)
        [init_mask0, init_mask1, init_mask2], \
        [affine_mask0, affine_mask1, affine_mask2], \
        [deform_mask0, deform_mask1, deform_mask2], \
        [nodes0, nodes1, nodes2], flow, preint_flow = predicts
        deform_mask = torch.cat([deform_mask0, deform_mask1, deform_mask2], dim=1)
        return super().forward(deform_mask, target)

    def description(self):
        return "Deform All Dice: {:.4f}".format(self.avg())
