import torch
import numpy as np
from rasterizor.voxelize import Voxelize
from experiments.geoshape.unet_3d import Encoder, Decoder
import torch.nn as nn
from experiments.geoshape.layers import SpatialTransformer, VecInt
from experiments.geoshape.sample import sample_lv_rv_points
from torch.distributions.normal import Normal
import torch.nn.functional as F
import math
from experiments.geoshape.nets import similarity_transform_points, norm_tensor

kernel_size = 5


class MidShapeDeformNet(torch.nn.Module):
    def __init__(self, voxel_width, voxel_depth, voxel_height, num_lv_slices, num_extra_slices, enc_dim):
        super().__init__()
        self.voxel_width = voxel_width
        self.voxel_depth = voxel_depth
        self.voxel_height = voxel_height
        self.num_lv_slices = num_lv_slices
        self.num_extra_slices = num_extra_slices

        padding = int((kernel_size - 1) / 2)

        self.shape_encoder = Encoder(enc_dim, drop=False, kernel_size=kernel_size, in_channels=1)

        self.shape_regressor = nn.Sequential(
            nn.Conv3d(
                self.shape_encoder.dims[-2], self.shape_encoder.dims[-2] // 2,
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm3d(self.shape_encoder.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(self.shape_encoder.dims[-2] // 2, self.shape_encoder.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm3d(self.shape_encoder.dims[-2] // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )

        self.shape_end_lv1 = nn.Sequential(nn.Linear(200, 4), nn.Tanh())  # c0_x, c0_y, c0_z, c0_z_end
        bias = torch.from_numpy(np.array([0, 0, -0.5, 0.5])).float()
        self.shape_end_lv1[0].weight.data.zero_()
        self.shape_end_lv1[0].bias.data.copy_(bias)

        self.shape_end_lv2 = nn.Sequential(nn.Linear(200, 2), nn.Sigmoid())  # r1, r0_r1_ratio
        bias = torch.from_numpy(np.array([-1.7, 0.690])).float()
        self.shape_end_lv2[0].weight.data.zero_()
        self.shape_end_lv2[0].bias.data.copy_(bias)

        num_tanh_delta = (num_lv_slices + num_extra_slices - 1) * 3  # dx, dy, dr1: dx = [-1, 1] * d_max
        self.shape_end_lv3 = nn.Sequential(nn.Linear(200, num_tanh_delta), nn.Tanh())
        d_bias = np.random.normal(0, 1, num_tanh_delta)
        d_bias[(num_lv_slices + num_extra_slices - 1) * 2:] = np.abs(d_bias[(num_lv_slices + num_extra_slices - 1) * 2:])
        bias = torch.from_numpy(d_bias).float()
        self.shape_end_lv3[0].weight.data.zero_()
        self.shape_end_lv3[0].bias.data.copy_(bias)

        num_tanh_delta = (num_lv_slices - 1)  # dr0
        self.shape_end_lv4 = nn.Sequential(nn.Linear(200, num_tanh_delta), nn.Tanh())
        d_bias = np.random.normal(0, 1, num_tanh_delta)
        d_bias[(num_lv_slices - 1) * 2:] = np.abs(d_bias[(num_lv_slices - 1) * 2:])
        bias = torch.from_numpy(d_bias).float()
        self.shape_end_lv4[0].weight.data.zero_()
        self.shape_end_lv4[0].bias.data.copy_(bias)

        num_tanh_delta = (num_lv_slices + num_extra_slices - 1) * 3  # dtheta2, dtheta_c2, dd_c2_c0
        self.shape_end_rv_tanh = nn.Sequential(nn.Linear(200, num_tanh_delta), nn.Tanh())
        d_bias = np.random.normal(0, 1, num_tanh_delta)
        d_bias[:(num_lv_slices + num_extra_slices - 1)] = np.abs(d_bias[:(num_lv_slices + num_extra_slices - 1)])
        bias = torch.from_numpy(d_bias).float()
        self.shape_end_rv_tanh[0].weight.data.zero_()
        self.shape_end_rv_tanh[0].bias.data.copy_(bias)

        self.shape_end_rv_sig = nn.Sequential(nn.Linear(200, 3), nn.Sigmoid())  # theta_c2, theta2_ratio, d_c2_c0_ratio
        bias = torch.from_numpy(np.array([1.9459, -1.6094, 0])).float()
        self.shape_end_rv_sig[0].weight.data.zero_()
        self.shape_end_rv_sig[0].bias.data.copy_(bias)

        self.tri0 = None
        self.tri1 = None
        self.tri2 = None
        self.voxeliser = Voxelize(voxel_width=self.voxel_width, voxel_depth=self.voxel_depth, voxel_height=self.voxel_height, eps=1e-4, eps_in=20)

        self.transform_encoder = Encoder(enc_dim, drop=False, kernel_size=kernel_size, in_channels=4)

        # affine transformation
        self.affine_regressor = nn.Sequential(
            nn.Conv3d(
                self.transform_encoder.dims[-2], self.transform_encoder.dims[-2] // 2,
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm3d(self.transform_encoder.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(self.transform_encoder.dims[-2] // 2, self.transform_encoder.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm3d(self.transform_encoder.dims[-2] // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 400),
            # nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        self.affine_end1 = nn.Sequential(nn.Linear(200, 6), nn.Tanh())  # rotation rx, ry, rz, translation x, y, z
        bias = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0])).float()
        self.affine_end1[0].weight.data.zero_()
        self.affine_end1[0].bias.data.copy_(bias)

        self.affine_end2 = nn.Sequential(nn.Linear(200, 3))  # scale x, y, z
        bias = torch.from_numpy(np.array([1, 1, 1])).float()
        self.affine_end2[0].weight.data.zero_()
        self.affine_end2[0].bias.data.copy_(bias)

        self.flow_decoder = Decoder(
            self.transform_encoder.dims, drop=False, kernel_size=kernel_size, output_act=None,
            output_dim=self.transform_encoder.dims[-4]
        )

        self.flow_conv = torch.nn.Conv3d(self.transform_encoder.dims[-4], 3, kernel_size=3, padding=1)
        # # init flow layer with small weights and bias
        self.flow_conv.weight = torch.nn.Parameter(Normal(0, 1e-5).sample(self.flow_conv.weight.shape))
        self.flow_conv.bias = torch.nn.Parameter(torch.zeros(self.flow_conv.bias.shape))

        self.integrate = VecInt(
            inshape=(self.voxel_width, self.voxel_depth, self.voxel_height),
            nsteps=7,
        )

        self.deform_transformer = SpatialTransformer(size=(self.voxel_width, self.voxel_depth, self.voxel_height), mode="bilinear")

    def forward(self, img, epoch=0):
        # x = (B, 1, H, W)
        x = norm_tensor(img)
        x = torch.movedim(x, 2, -1)
        em = self.shape_encoder(x)
        out = self.shape_regressor(em[-1])
        lv_par1 = self.shape_end_lv1(out)  # (B, 3), tanh
        lv_par2 = self.shape_end_lv2(out)  # (B, 5), sig
        lv_par3 = self.shape_end_lv3(out)  # (B, 5), sig
        lv_par4 = self.shape_end_lv4(out)  # (B, 5), sig

        rv_par_tanh = self.shape_end_rv_tanh(out)
        rv_par_sig = self.shape_end_rv_sig(out)

        [nodes0, tetras0, self.tri0], [nodes1, tetras1, self.tri1], [nodes2, tetras2, self.tri2] = sample_lv_rv_points(
            lv_par1=lv_par1, lv_par2=lv_par2, lv_par3=lv_par3, lv_par4=lv_par4, rv_par1=rv_par_tanh, rv_par2=rv_par_sig,
            num_lv_slices=self.num_lv_slices, num_extra_lv_myo_slices=self.num_extra_slices,
            voxel_width=self.voxel_width, voxel_depth=self.voxel_depth, voxel_height=self.voxel_height,
            num_points=32, batch_size=img.shape[0],
            lv_tetras=self.tri0, lv_myo_tetras=self.tri1, rv_tetras=self.tri2, epoch=epoch,
        )
        init_mask0 = self.voxelize_mask(nodes0, tetras0)
        init_mask1 = self.voxelize_mask(nodes1, tetras1)
        init_mask2 = self.voxelize_mask(nodes2, tetras2)

        transform_in = torch.cat([x, init_mask0, init_mask1, init_mask2], dim=1).detach()
        em = self.transform_encoder(transform_in)

        # affine transform
        affine_pars = self.affine_regressor(em[-1])

        affine_pars1 = self.affine_end1(affine_pars)
        affine_pars2 = self.affine_end2(affine_pars)

        affine_node0 = similarity_transform_points(nodes0, affine_pars1, affine_pars2)
        affine_node1 = similarity_transform_points(nodes1, affine_pars1, affine_pars2)
        affine_node2 = similarity_transform_points(nodes2, affine_pars1, affine_pars2)

        affine_mask0 = self.voxelize_mask(affine_node0, tetras0)
        affine_mask1 = self.voxelize_mask(affine_node1, tetras1)
        affine_mask2 = self.voxelize_mask(affine_node2, tetras2)

        flow = self.flow_conv(self.flow_decoder(em))  # (B, 3, W, H, D) (dx, dy, dz)
        flow = flow / img.shape[2]

        preint_flow = flow
        flow = self.integrate(preint_flow)
        # vertices: (B, D, 3)
        affine_node0[..., 1] = affine_node0[..., 1] * -1
        Pxx = F.grid_sample(flow[:, 0:1], affine_node0.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)  # Pxx (B, D, 1)
        Pyy = F.grid_sample(flow[:, 1:2], affine_node0.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)
        Pzz = F.grid_sample(flow[:, 2:3], affine_node0.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)

        dP0 = torch.cat((Pxx, Pyy, Pzz), -1)
        deform_node0 = affine_node0 + dP0

        affine_node1[..., 1] = affine_node1[..., 1] * -1
        Pxx = F.grid_sample(flow[:, 0:1], affine_node1.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)  # Pxx (B, D, 1)
        Pyy = F.grid_sample(flow[:, 1:2], affine_node1.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)
        Pzz = F.grid_sample(flow[:, 2:3], affine_node1.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)

        dP0 = torch.cat((Pxx, Pyy, Pzz), -1)
        deform_node1 = affine_node1 + dP0

        affine_node2[..., 1] = affine_node2[..., 1] * -1
        Pxx = F.grid_sample(flow[:, 0:1], affine_node2.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)  # Pxx (B, D, 1)
        Pyy = F.grid_sample(flow[:, 1:2], affine_node2.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)
        Pzz = F.grid_sample(flow[:, 2:3], affine_node2.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)

        dP0 = torch.cat((Pxx, Pyy, Pzz), -1)
        deform_node2 = affine_node2 + dP0

        deform_mask0 = self.voxelize_mask(deform_node0, tetras0)
        deform_mask1 = self.voxelize_mask(deform_node1, tetras1)
        deform_mask2 = self.voxelize_mask(deform_node2, tetras2)

        return [init_mask0, init_mask1, init_mask2], \
               [affine_mask0, affine_mask1, affine_mask2], \
               [deform_mask0, deform_mask1, deform_mask2], \
               [deform_node0, deform_node1, deform_node2], flow, preint_flow

    def voxelize_mask(self, nodes, faces):
        P3d = torch.squeeze(nodes, dim=1)
        faces = torch.squeeze(faces, dim=1).to(nodes.device)
        mask = self.voxeliser(P3d, faces).unsqueeze(1)
        return mask
