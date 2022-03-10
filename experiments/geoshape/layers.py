import torch
import torch.nn.functional as F
import torch.nn.functional as nnf


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


class SpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        # should it be minus?
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(torch.nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class AffineSpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        # vectors = [torch.arange(0, s) for s in size]
        # grids = torch.meshgrid(vectors)
        # grid = torch.stack(grids)
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor)
        #
        # # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # # adds it to the state dict. this is annoying since everything in the state dict
        # # is included when saving weights to disk, so the model files are way bigger
        # # than they need to be. so far, there does not appear to be an elegant solution.
        # # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)

    def forward(self, src, theta):
        # theta = (N, 3, 4)
        # new locations
        new_locs = nnf.affine_grid(theta, size=src.shape)
        # grid = self.grid.permute(0, 2, 3, 4, 1)
        # new_locs = grid + new_locs
        # new_locs = new_locs[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

