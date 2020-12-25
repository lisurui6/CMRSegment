import torch
import numpy as np
from voxelmorph.torch.networks import VxmDense
from experiments.fcn_3d.network import UNet


class DefSegNet(torch.nn.Module):
    def __init__(self, template: np.ndarray, in_channels, n_classes, n_filters, feature_size, n_slices, int_steps=7,
                 int_downsize=1, bidir=False):
        super().__init__()
        self.seg_unet = UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            n_filters=n_filters,
        )
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        self.vxm_dense = VxmDense(
            inshape=(n_slices, feature_size, feature_size),
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=int_steps,
            int_downsize=int_downsize
        )
        self.template = torch.from_numpy(template).float().cuda().unsqueeze(0)

    def forward(self, image):
        pred_maps = self.seg_unet(image)
        pred_maps = torch.sigmoid(pred_maps)
        # out = (3, 64, 128, 128)
        # warped_maps, flow = self.vxm_dense(pred_maps, self.template)
        warped_template, flow = self.vxm_dense(self.template, pred_maps)

        # if not self.training:
        #     visualise(image, self.template, pred_maps, warped_template)

        return warped_template, pred_maps, flow


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
    # print(predicted.shape, final_predicted.shape)

    for i in range(image.shape[0]):
        final_predicted[image[i, :, :, :] > 0.5] = i + 1
    return final_predicted
