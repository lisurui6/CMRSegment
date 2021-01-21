import torch
import torch.nn as nn
import numpy as np
from voxelmorph.torch.networks import VxmDense
from experiments.fcn_3d.network import UNet
from experiments.def_seg_bidir_affine_no_seg.layers import AffineSpatialTransformer


def conv_block_2_3d(in_dim, out_dim, activation, batch_norm: bool = True, group_norm=0):
    if batch_norm:
        return nn.Sequential(
            # conv_block_3d(in_dim, out_dim, activation),
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
        )
    elif group_norm > 0:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_norm, out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_norm, out_dim),
        )
    else:
        return nn.Sequential(
            # conv_block_3d(in_dim, out_dim, activation),
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )


class AffineLocalNet(torch.nn.Module):
    def __init__(self, in_dim, batch_norm, group_norm):
        super().__init__()
        activation = torch.nn.ReLU
        self.conv1 = conv_block_2_3d(in_dim, 16, activation, batch_norm, group_norm)
        # self.maxpoo1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = conv_block_2_3d(16, 32, activation, batch_norm, group_norm)
        self.maxpoo2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv3 = conv_block_2_3d(32, 64, activation, batch_norm, group_norm)
        # self.maxpoo3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv4 = conv_block_2_3d(64, 32, activation, batch_norm, group_norm)
        self.maxpoo4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv5 = conv_block_2_3d(32, 16, activation, batch_norm, group_norm)
        # self.maxpoo5 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            activation(),
        )
        self.regress = torch.nn.Sequential(
            torch.nn.Linear(32, 50),
            activation(),
            torch.nn.Linear(50, 12),
            torch.nn.Tanh()  # affine grid seems to want [-1, 1]
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])).float()
        self.regress[2].weight.data.zero_()
        self.regress[2].bias.data.copy_(bias)

    def forward(self, pred_maps, atlas):
        x = torch.cat([pred_maps, atlas], dim=1)
        out = self.conv1(x)
        out = self.maxpoo1(out)
        out = self.conv2(out)
        out = self.maxpoo2(out)
        out = self.conv3(out)
        out = self.maxpoo3(out)
        out = self.conv4(out)
        out = self.maxpoo4(out)
        out = self.conv5(out)
        out = self.maxpoo5(out)

        out = self.conv6(out)
        print(out.shape)
        out = out.view(out.shape[0], -1)
        print(out.shape)
        theta = self.regress(out)
        theta = theta.view(theta.shape[0], 3, 4)
        return theta


class DefSegNet(torch.nn.Module):
    def __init__(self, in_channels, n_classes, n_filters, feature_size, n_slices, int_steps=7,
                 int_downsize=1, bidir=False, batch_norm=True, group_norm=0):
        assert not (batch_norm and group_norm)
        super().__init__()

        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.vxm_dense = VxmDense(
            inshape=(n_slices, feature_size, feature_size),
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=int_steps,
            int_downsize=int_downsize,
            mode="bilinear",
            batch_norm=batch_norm,
            group_norm=group_norm,
            in_dim=4,
        )
        self.affine_local = AffineLocalNet(4, batch_norm=batch_norm, group_norm=group_norm)
        self.affine_transformer = AffineSpatialTransformer(
            size=(n_slices, feature_size, feature_size), mode="bilinear"
        )

    def forward(self, inputs):
        image, template = inputs

        affine_theta = self.affine_local(image, template)
        affine_transformed_template = self.affine_transformer(template, affine_theta)
        warped_template, warped_maps, flow, pos_flow, neg_flow = self.vxm_dense(affine_transformed_template, image)

        return warped_template, affine_transformed_template, flow

    def freeze_vxm(self):
        for param in self.vxm_dense.parameters():
            param.requires_grad = False


def visualise(image, template, pred: torch.Tensor, warped: torch.Tensor):
    import neurite as ne
    from matplotlib import pyplot as plt
    image = image.squeeze().cpu().detach().numpy()
    template = template.squeeze().cpu().detach().numpy()
    template = maps_to_volume(template)
    pred = pred.squeeze().cpu().detach().numpy()
    pred = maps_to_volume(pred)
    warped = warped.squeeze().cpu().detach().numpy()
    warped = maps_to_volume(warped)

    ne.plot.volume3D([image, template, pred, warped])
    plt.close()
    pass


def maps_to_volume(image: np.ndarray):
    final_predicted = np.zeros((image.shape[1], image.shape[2], image.shape[3]))
    for i in range(image.shape[0]):
        final_predicted[image[i, :, :, :] > 0.5] = i + 1
    return final_predicted

