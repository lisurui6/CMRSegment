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


class AffineRegressor(torch.nn.Module):
    def __init__(self, group_norm):
        super().__init__()
        activation = torch.nn.ReLU
        self.conv = torch.nn.Sequential(
            # torch.nn.Conv3d(256, 128, kernel_size=1),
            # nn.GroupNorm(group_norm, 128),
            # activation(),
            torch.nn.Conv3d(128, 64, kernel_size=1),
            nn.GroupNorm(group_norm, 64),
            activation(),
            max_pooling_3d(),
            torch.nn.Conv3d(64, 32, kernel_size=1),
            nn.GroupNorm(group_norm, 32),
            activation(),
            torch.nn.Conv3d(32, 8, kernel_size=1),
            nn.GroupNorm(group_norm, 8),
            activation(),
            # torch.nn.Conv3d(16, 8, kernel_size=1),
            # nn.BatchNorm3d(8),
            # activation(),
        )
        self.regress = torch.nn.Sequential(
            torch.nn.Linear(400, 200),
            # nn.BatchNorm1d(200),
            nn.GroupNorm(group_norm, 200),
            activation(),
            torch.nn.Linear(200, 96),
            # nn.BatchNorm1d(100),
            nn.GroupNorm(group_norm, 96),
            activation(),
            torch.nn.Linear(96, 12),
            torch.nn.Tanh()  # affine grid seems to want [-1, 1]
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])).float()
        self.regress[-2].weight.data.zero_()
        self.regress[-2].bias.data.copy_(bias)

    def forward(self, image_code, template_code):
        x = torch.cat([image_code, template_code], dim=1)

        out = self.conv(x)
        out = out.view(out.shape[0], -1)

        theta = self.regress(out)
        theta = theta.view(theta.shape[0], 3, 4)
        return theta


class FlowDecoder(torch.nn.Module):
    def __init__(self, num_filters, batch_norm, group_norm):
        super().__init__()
        activation = torch.nn.ReLU
        self.num_filters = num_filters
        self.batch_norm = batch_norm

        # Up sampling
        self.trans_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_1 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.trans_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_2 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation, self.batch_norm, group_norm)
        self.trans_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_3 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation, self.batch_norm, group_norm)
        self.trans_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 1, activation, self.batch_norm, group_norm)
        # self.trans_5 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation, self.batch_norm, group_norm)

    def forward(self, img_down1, img_down2, img_down3, img_down4, img_bridge,
               temp_down1, temp_down2, temp_down3, temp_down4, temp_bridge):
        bridge = torch.cat([img_bridge, temp_bridge], dim=1)

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [None, 512, 4, 8, 8]
        down_5 = torch.cat([img_down4, temp_down4], dim=1)
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        down_4 = torch.cat([img_down3, temp_down3], dim=1)
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 192, 8, 8, 8]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        down_3 = torch.cat([img_down2, temp_down2], dim=1)
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        down_2 = torch.cat([img_down1, temp_down1], dim=1)
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        # trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        # down_1 = torch.cat([img_down1, temp_down1], dim=1)
        # concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        # up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        # out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return up_4


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


class DecoderVxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(
        self,
        inshape,
        n_filters,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False,
        mode="bilinear",
        batch_norm=False,
        group_norm=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or int egrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.flow_decoder = FlowDecoder(n_filters, batch_norm, group_norm)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(n_filters, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape, mode=mode)

    def forward(self, source, target,
                img_down1, img_down2, img_down3, img_down4, img_bridge,
                temp_down1, temp_down2, temp_down3, temp_down4, temp_bridge, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = self.flow_decoder(
            img_down1, img_down2, img_down3, img_down4, img_bridge,
            temp_down1, temp_down2, temp_down3, temp_down4, temp_bridge
        )
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow, pos_flow, neg_flow) if self.bidir else (y_source, preint_flow, pos_flow, neg_flow)
        else:
            return y_source, pos_flow


class ImgTemplateEncoderNet(torch.nn.Module):
    def __init__(self, in_channels, n_classes, n_slices, feature_size, n_filters, batch_norm: bool, group_norm, bidir,
                 int_downsize, batch_size, gpu, device):
        super().__init__()
        self.image_encoder = Encoder(in_channels, n_filters, batch_norm=batch_norm, group_norm=group_norm)
        self.template_encoder = Encoder(n_classes, n_filters, batch_norm=batch_norm, group_norm=group_norm)
        self.affine_regressor = AffineRegressor(group_norm=group_norm)
        self.affine_transformer = AffineSpatialTransformer(
            size=(n_slices, feature_size, feature_size), mode="bilinear"
        )
        self.decoder_vxm = DecoderVxmDense(
            inshape=(n_slices, feature_size, feature_size),
            n_filters=n_filters, batch_norm=batch_norm, group_norm=group_norm, int_downsize=int_downsize, bidir=False
        )
        self.seg_decoder = SegDecoder(n_filters, batch_norm, group_norm, n_classes)
        self.batch_size = batch_size
        self.batch_atlas = None
        self.gpu = gpu
        self.device = device
        self.batch_affine_added = prepare_tensors(
            torch.stack([torch.from_numpy(np.array([[0, 0, 0, 1]])) for _ in range(self.batch_size)], dim=0),
            self.gpu, self.device,
        )

    def update_batch_atlas(self, atlas):
        self.batch_atlas = torch.stack([atlas for _ in range(self.batch_size)], dim=0)

    def forward(self, inputs, atlas):
        image, label = inputs
        img_down1, img_down2, img_down3, img_down4, img_bridge = self.image_encoder(image)
        pred_maps = self.seg_decoder(img_down1, img_down2, img_down3, img_down4, img_bridge)
        pred_maps = torch.sigmoid(pred_maps)

        if image.shape[0] == self.batch_size:
            batch_atlas = self.batch_atlas
            batch_affine_added = self.batch_affine_added
        else:
            batch_atlas = torch.stack([atlas for _ in range(image.shape[0])], dim=0)
            batch_affine_added = prepare_tensors(
                torch.stack([torch.from_numpy(np.array([[0, 0, 0, 1]])) for _ in range(image.shape[0])], dim=0),
                self.gpu, self.device,
            )

        temp_down1, temp_down2, temp_down3, temp_down4, temp_bridge = self.template_encoder(batch_atlas)
        affine_params = self.affine_regressor(img_bridge, temp_bridge)
        affine_warped_template = self.affine_transformer(batch_atlas, affine_params)

        temp_down1, temp_down2, temp_down3, temp_down4, temp_bridge = self.template_encoder(affine_warped_template)
        warped_template, preint_flow, pos_flow, neg_flow = self.decoder_vxm(
            affine_warped_template, None,
            img_down1, img_down2, img_down3, img_down4, img_bridge,
            temp_down1, temp_down2, temp_down3, temp_down4, temp_bridge
        )

        # inverse transform of label to template space
        inverse_affine_par = inverse_affine_params(affine_params, batch_affine_added)
        affine_warped_label = self.affine_transformer(label, inverse_affine_par)
        warped_label = self.decoder_vxm.transformer(affine_warped_label, neg_flow)

        # inverse transform of image to template space
        affine_warped_image = self.affine_transformer(image, inverse_affine_par)
        warped_image = self.decoder_vxm.transformer(affine_warped_image, neg_flow)

        return affine_warped_template, warped_template, pred_maps, preint_flow, warped_image, warped_label, \
               affine_warped_label, batch_atlas, affine_warped_image


def inverse_affine_params(affine_params, affine_added):
    theta = torch.cat([affine_params, affine_added], dim=1)
    inverse_theta = torch.inverse(theta)
    return inverse_theta[:, :3, :]
