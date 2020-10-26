import torch
import math
from typing import List


class FCN2DSegmentationModel(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_filters: List[int], up_conv_filter: int,
                 final_conv_filter: int, feature_size: int = 192):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            self.conv2d_bn_relu(in_channels, n_filters[0]),
            self.conv2d_bn_relu(n_filters[0], n_filters[0]),
        )

        self.block2 = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[0], n_filters[1]),
            self.conv2d_bn_relu(n_filters[1], n_filters[1]),
        )

        self.block3 = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[1], n_filters[2]),
            self.conv2d_bn_relu(n_filters[2], n_filters[2]),
            self.conv2d_bn_relu(n_filters[2], n_filters[2]),
        )

        self.block4 = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[2], n_filters[3]),
            self.conv2d_bn_relu(n_filters[3], n_filters[3]),
            self.conv2d_bn_relu(n_filters[3], n_filters[3]),
        )

        self.block5 = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[3], n_filters[4]),
            self.conv2d_bn_relu(n_filters[4], n_filters[4]),
            self.conv2d_bn_relu(n_filters[4], n_filters[4]),
        )

        self.block1_up = self.conv2d_bn_relu(n_filters[0], up_conv_filter)

        self.block2_up = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[1], up_conv_filter),
            self.conv2d_transpose_same_padding(up_conv_filter, up_conv_filter, kernel_size=4, stride=2, output_size=feature_size),
        )

        self.block3_up = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[2], up_conv_filter),
            self.conv2d_transpose_same_padding(up_conv_filter, up_conv_filter, kernel_size=8, stride=4, output_size=feature_size),
        )

        self.block4_up = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[3], up_conv_filter),
            self.conv2d_transpose_same_padding(up_conv_filter, up_conv_filter, kernel_size=15, stride=7, output_size=feature_size),
        )

        self.block5_up = torch.nn.Sequential(
            self.conv2d_bn_relu(n_filters[4], up_conv_filter),
            self.conv2d_transpose_same_padding(up_conv_filter, up_conv_filter, kernel_size=31, stride=15, output_size=feature_size),
        )

        self.final_block = torch.nn.Sequential(
            self.conv2d_bn_relu(5*up_conv_filter, final_conv_filter),
            self.conv2d_bn_relu(final_conv_filter, final_conv_filter),
            torch.nn.Conv2d(
                in_channels=final_conv_filter,
                out_channels=n_classes * 12,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor):
        conv1 = self.block1(x)
        conv2 = self.block2(conv1)
        conv3 = self.block3(conv2)
        conv4 = self.block4(conv3)
        conv5 = self.block5(conv4)

        conv1_up = self.block1_up(conv1)
        conv2_up = self.block2_up(conv2)
        conv3_up = self.block3_up(conv3)
        conv4_up = self.block4_up(conv4)
        conv5_up = self.block5_up(conv5)
        x = torch.cat([conv1_up, conv2_up, conv3_up, conv4_up, conv5_up], 1)
        logits = self.final_block(x)
        logits = logits.reshape((logits.shape[0], 3, 12, logits.shape[2], logits.shape[3]))
        return logits

    @staticmethod
    def conv2d_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=bias
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    @staticmethod
    def conv2d_transpose_same_padding(in_channels: int, out_channels: int, kernel_size: int, stride: int, output_size: int):
        padding = ((output_size - 1) * stride + kernel_size - output_size) // 2
        layer = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding
        )
        return layer
