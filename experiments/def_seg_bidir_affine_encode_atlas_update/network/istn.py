import torch
import torch.nn as nn
import numpy as np
from voxelmorph.torch.networks import VxmDense, LoadableModel, store_config_args, Normal, layers
from experiments.fcn_3d.network import UNet, max_pooling_3d, conv_trans_block_3d
from experiments.def_seg_bidir_affine_encode_atlas_update.layers import AffineSpatialTransformer
from CMRSegment.common.nn.torch import prepare_tensors


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


def visualise(image, template, pred: torch.Tensor, warped: torch.Tensor):
    import neurite as ne
    from matplotlib import pyplot as plt
    image = image.squeeze().cpu().detach().numpy()
    template = template.squeeze().cpu().detach().numpy()
    template = np.expand_dims(template, axis=0)
    template = maps_to_volume(template)
    pred = pred.squeeze().cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=0)

    pred = maps_to_volume(pred)
    warped = warped.squeeze().cpu().detach().numpy()
    warped = np.expand_dims(warped, axis=0)
    warped = maps_to_volume(warped)

    ne.plot.volume3D([image, template, pred, warped])
    plt.close()
    pass


def maps_to_volume(image: np.ndarray):
    final_predicted = np.zeros((image.shape[1], image.shape[2], image.shape[3]))
    for i in range(image.shape[0]):
        final_predicted[image[i, :, :, :] > 0.5] = i + 1
    # final_predicted[image[1, :, :, :] > 0.5] = 1

    return final_predicted


class Encoder(nn.Module):
    def __init__(self, in_channels, n_filters, batch_norm: bool = True, group_norm=0):
        super().__init__()

        self.in_dim = in_channels
        self.num_filters = n_filters
        activation = nn.ReLU
        self.batch_norm = batch_norm

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation, self.batch_norm, group_norm)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters, activation, self.batch_norm, group_norm)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation, self.batch_norm, group_norm)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation, self.batch_norm, group_norm)
        self.pool_4 = max_pooling_3d()
        # self.down_5 = conv_block_2_3d(self.num_filters, self.num_filters, activation, self.batch_norm, group_norm)
        # self.pool_5 = max_pooling_3d()
        self.bridge = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation, self.batch_norm, group_norm)

    def forward(self, x):
        # x -> [None, 1, 64, 128, 128]
        # Down sampling
        down_1 = self.down_1(x)       # -> [None, 16, 64, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [None, 16, 32, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [None, 32, 32, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [None, 32, 16, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [None, 64, 16, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [None, 64, 8, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [None, 128, 8, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [None, 128, 4, 8, 8]

        # down_5 = self.down_5(pool_4)  # -> [None, 256, 4, 8, 8]
        # pool_5 = self.pool_5(down_5)  # -> [None, 256, 2, 4, 4]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [None, 512, 2, 4, 4]
        return down_1, down_2, down_3, down_4, bridge


class SegDecoder(torch.nn.Module):
    def __init__(self, num_filters, batch_norm, group_norm, out_dim):
        super().__init__()
        activation = torch.nn.ReLU
        self.num_filters = num_filters
        self.batch_norm = batch_norm
        self.out_dim = out_dim

        # Up sampling
        self.trans_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_1 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 6, activation, self.batch_norm, group_norm)
        self.trans_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_2 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 4, activation, self.batch_norm, group_norm)
        self.trans_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_3 = conv_block_2_3d(self.num_filters * 5, self.num_filters * 1, activation, self.batch_norm, group_norm)
        self.trans_4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_4 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation, self.batch_norm, group_norm)
        # self.trans_5 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up_5 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 1, activation, self.batch_norm, group_norm)

        # Output
        self.out = nn.Conv3d(self.num_filters, self.out_dim, kernel_size=1)

    def forward(self, img_down1, img_down2, img_down3, img_down4, img_bridge):
        # Up sampling
        trans_1 = self.trans_1(img_bridge)  # -> [None, 512, 4, 8, 8]
        concat_1 = torch.cat([trans_1, img_down4], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, img_down3], dim=1)  # -> [1, 192, 8, 8, 8]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, img_down2], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, img_down1], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        # trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        # concat_5 = torch.cat([trans_5, img_down1], dim=1)  # -> [1, 12, 128, 128, 128]
        # up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_4)  # -> [1, 3, 128, 128, 128]
        return out


class STN(torch.nn.Module):
    def __init__(self, num_filters, group_norm: int):
        super().__init__()
        activation = torch.nn.ReLU
        self.down1 = max_pooling_3d()
        self.down_conv1 = conv_block_2_3d(6, num_filters, activation, group_norm=group_norm)
        self.down2 = max_pooling_3d()
        self.down_conv2 = conv_block_2_3d(num_filters, num_filters, activation, group_norm=group_norm)
        self.down3 = max_pooling_3d()
        self.down_conv3 = conv_block_2_3d(num_filters, num_filters*2, activation, group_norm=group_norm)

        self.affine_down = max_pooling_3d()
        self.affine_conv = conv_block_2_3d(num_filters*2, num_filters*2, activation, group_norm=group_norm)
        self.affine_down2 = max_pooling_3d()
        self.affine_conv2 = conv_block_2_3d(num_filters*2, num_filters*2, activation, group_norm=group_norm)

        self.affine_regressor = torch.nn.Sequential(
            torch.nn.Linear(2048, 400),
            # nn.BatchNorm1d(200),
            nn.GroupNorm(group_norm, 400),
            activation(),
            torch.nn.Linear(400, 12),
            torch.nn.Tanh()  # affine grid seems to want [-1, 1]
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])).float()
        self.affine_regressor[-2].weight.data.zero_()
        self.affine_regressor[-2].bias.data.copy_(bias)

        self.up0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = conv_block_2_3d(num_filters*4, num_filters*2, activation, group_norm=group_norm)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv2 = conv_block_2_3d(num_filters*3, num_filters, activation, group_norm=group_norm)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv3 = conv_block_2_3d(num_filters*2, num_filters, activation, group_norm=group_norm)
        self.flow = nn.Conv3d(num_filters, 3, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, pred_maps, atlas):
        x = torch.cat([pred_maps, atlas], dim=1)
        print("cat", x.shape)
        x = self.down1(x)
        print("down 1", x.shape)
        map1 = self.down_conv1(x)
        x = self.down2(map1)
        print("down 2", x.shape)
        map2 = self.down_conv2(x)
        x = self.down3(map2)
        print("down 3", x.shape)
        map3 = self.down_conv3(x)
        print("map 3", map3.shape)
        affine = self.affine_down(map3)
        print("affine", affine.shape)
        affine_map = self.affine_conv(affine)
        affine1 = self.affine_down2(affine_map)
        affine_map = self.affine_conv2(affine1)
        print("affine map", affine_map.shape)
        out = affine_map.view(affine_map.shape[0], -1)
        print(out.shape)
        affine_params = self.affine_regressor(out)
        affine_params = affine_params.view(affine_params.shape[0], 3, 4)

        x = torch.cat([map3, self.up1(self.up0(affine_map))], dim=1)
        x = self.up_conv1(x)
        x = torch.cat([map2, self.up2(x)], dim=1)
        x = self.up_conv2(x)
        x = torch.cat([map1, self.up3(x)], dim=1)
        x = self.up_conv3(x)
        x = self.flow(x)
        return x, affine_params


class FFDTransformer(torch.nn.Module):
    def __init__(self, inshape, int_steps=7, int_downsize=2, bidir=False, mode="bilinear"):
        super().__init__()
        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, 3) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, 3) if resize else None
        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape, mode=mode)

    def forward(self, source, target, flow_field, bidir: bool = False, resize: bool = False):
        # resize flow for integration
        pos_flow = flow_field
        if resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if bidir else None

        # return non-integrated flow field if training
        return (y_source, y_target, preint_flow, pos_flow, neg_flow) if bidir else (y_source, preint_flow)


class ISTNNet(torch.nn.Module):
    def __init__(self, in_channels, n_classes, n_slices, feature_size, n_filters, batch_norm: bool, group_norm, bidir,
                 int_downsize, batch_size, gpu, device):
        super().__init__()
        # ITN
        self.seg_encoder = Encoder(in_channels, n_filters, batch_norm=batch_norm, group_norm=group_norm)
        self.seg_decoder = SegDecoder(n_filters, batch_norm, group_norm, n_classes)

        # STN
        self.stn = STN(num_filters=n_filters, group_norm=group_norm)
        self.batch_size = batch_size
        self.batch_atlas = None
        self.gpu = gpu
        self.device = device
        # configure transformer
        self.ffd_transformer = FFDTransformer(
            inshape=(n_slices, feature_size, feature_size),
            int_downsize=2, bidir=bidir, mode="bilinear"
        )
        self.affine_transformer = AffineSpatialTransformer(
            size=(n_slices, feature_size, feature_size), mode="bilinear"
        )
        self.batch_affine_added = prepare_tensors(
            torch.stack([torch.from_numpy(np.array([[0, 0, 0, 1]])) for _ in range(self.batch_size)], dim=0),
            self.gpu, self.device,
        )

    def update_batch_atlas(self, atlas):
        self.batch_atlas = torch.stack([atlas for _ in range(self.batch_size)], dim=0)

    def forward(self, inputs, atlas):
        image, label = inputs

        if image.shape[0] == self.batch_size:
            batch_atlas = self.batch_atlas
            batch_affine_added = self.batch_affine_added
        else:
            batch_atlas = torch.stack([atlas for _ in range(image.shape[0])], dim=0)
            batch_affine_added = prepare_tensors(
                torch.stack([torch.from_numpy(np.array([[0, 0, 0, 1]])) for _ in range(image.shape[0])], dim=0),
                self.gpu, self.device,
            )

        img_down1, img_down2, img_down3, img_down4, img_bridge = self.seg_encoder(image)
        pred_maps_logits = self.seg_decoder(img_down1, img_down2, img_down3, img_down4, img_bridge)
        # pred_maps = torch.sigmoid(pred_maps)
        flow, affine_params = self.stn(pred_maps_logits, batch_atlas)
        # inverse transform of label to template space
        inverse_affine_par = inverse_affine_params(affine_params, batch_affine_added)

        affine_warped_atlas = self.affine_transformer(batch_atlas, affine_params)
        affine_warped_label = self.affine_transformer(label, inverse_affine_par)

        warped_atlas, warped_label, preint_flow, pos_flow, neg_flow = self.ffd_transformer(
            affine_warped_atlas, affine_warped_label, flow, bidir=True, resize=False
        )
        # inverse transform of image to template space
        affine_warped_image = self.affine_transformer(image, inverse_affine_par)
        warped_image = self.ffd_transformer.transformer(affine_warped_image, neg_flow)

        return affine_warped_atlas, warped_atlas, pred_maps_logits, preint_flow, warped_image, warped_label, \
               affine_warped_label, batch_atlas, affine_warped_image


def inverse_affine_params(affine_params, affine_added):
    theta = torch.cat([affine_params, affine_added], dim=1)
    inverse_theta = torch.inverse(theta)
    return inverse_theta[:, :3, :]
