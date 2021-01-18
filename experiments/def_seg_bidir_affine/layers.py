import torch
import torch.nn as nn
import torch.nn.functional as nnf


class AffineSpatialTransformer(nn.Module):
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

    def forward(self, src, theta):
        # theta = (N, 3, 4)
        # new locations
        grid = nnf.affine_grid(theta, size=src.shape)
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(grid) == 2:
            grid = grid.permute(0, 2, 3, 1)
            grid = grid[..., [1, 0]]
        elif len(grid) == 3:
            grid = grid.permute(0, 2, 3, 4, 1)
            grid = grid[..., [2, 1, 0]]

        new_locs = self.grid + grid
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
