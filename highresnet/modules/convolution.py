import torch.nn as nn
import torch.nn.functional as F
from .padding import MixedPad
from .split import SplitReLU

PADDING_MODES = {
    'reflect',
    'replicate',
    'constant',
    'circular'
}


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            dimensions,
            batch_norm=True,
            instance_norm=False,
            norm_affine=True,
            padding_mode='constant',
            preactivation=True,
            kernel_size=3,
            activation=True,
            split_relu=False,
            ):
        assert padding_mode in PADDING_MODES
        assert not (batch_norm and instance_norm)
        super().__init__()

        padding_instance = MixedPad(dilation, padding_mode)

        conv_class = nn.Conv2d if dimensions == 2 else nn.Conv3d

        if batch_norm:
            norm_class = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d
        if instance_norm:
            norm_class = nn.InstanceNorm2d if dimensions == 2 else nn.InstanceNorm3d

        layers = nn.ModuleList()

        if preactivation:
            if batch_norm or instance_norm:
                layers.append(norm_class(in_channels, affine=norm_affine))
            if activation:
                if split_relu:
                    layers.append(SplitReLU())
                else:
                    layers.append(nn.ReLU())

        if kernel_size > 1:
            layers.append(padding_instance)

        if split_relu and activation:
            if preactivation:
                in_channels = in_channels * 2
            else:
                out_channels = out_channels // 2

        use_bias = not (instance_norm or batch_norm)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=use_bias,
            )
        layers.append(conv_layer)

        if not preactivation:
            if batch_norm or instance_norm:
                layers.append(norm_class(out_channels, affine=norm_affine))
                if split_relu:
                    layers.append(SplitReLU())
                else:
                    layers.append(nn.ReLU())

        self.convolutional_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolutional_block(x)

