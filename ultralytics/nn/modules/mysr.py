import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class sr(nn.Module):
    def __init__(self, ch, c1=128):
        super(sr, self).__init__()
        self.decoder = Decoder(c1)
        self.edsr = EnhancedSRNet(num_channels=ch, input_channel=64, factor=4)

    def forward(self, x):
        x_sr = self.decoder(x)
        x_sr_up = self.edsr(x_sr)
        return x_sr_up


class Decoder(nn.Module):
    def __init__(self, c1):
        super().__init__()

        self.conv1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(c1 // 2, c1 // 2, 3, padding=1, bias=False)
        self.attention = SEBlock(c1 // 2)
        self.shortcut = nn.Conv2d(c1, c1 // 2, 1, bias=False) if c1 != c1 // 2 else nn.Identity()

        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.attention(x)
        return x + identity

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def make_model(args, parent=False):
    return EnhancedSRNet(args)


def standard_conv(in_channels, out_channels, kernel_size, use_bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=use_bias)


class UpscaleModule(nn.Sequential):
    def __init__(self, conv_layer, scale_factor, feature_dim, batch_norm=False, activation=False, bias=True):
        layers = []
        if (scale_factor & (scale_factor - 1)) == 0:
            for _ in range(int(math.log(scale_factor, 2))):
                layers.append(conv_layer(feature_dim, 4 * feature_dim, 3, use_bias=bias))
                layers.append(nn.PixelShuffle(2))
                if batch_norm: layers.append(nn.BatchNorm2d(feature_dim))
                if activation: layers.append(activation())
        super(UpscaleModule, self).__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(
            self, conv_layer, feature_dim, kernel_size,
            use_bias=True, batch_norm=False, activation=nn.ReLU(True), scaling=1):

        super(ResidualBlock, self).__init__()
        components = []
        for idx in range(2):
            components.append(conv_layer(feature_dim, feature_dim, kernel_size, use_bias=use_bias))
            if batch_norm: components.append(nn.BatchNorm2d(feature_dim))
            if idx == 0: components.append(activation)

        self.core_block = nn.Sequential(*components)
        self.scaling_factor = scaling

    def forward(self, input_tensor):
        residual = self.core_block(input_tensor).mul(self.scaling_factor)
        return residual + input_tensor


class EnhancedSRNet(nn.Module):
    def __init__(self, num_channels=3, input_channel=64, factor=4, width=64, depth=4, kernel_size=3,
                 conv=standard_conv):
        super(EnhancedSRNet, self).__init__()
        num_blocks = depth
        feature_dim = width
        kernel_dim = kernel_size
        scale_factor = factor
        activation_func = nn.ReLU()

        input_layers = [conv(input_channel, feature_dim, kernel_dim)]

        body_layers = [
            ResidualBlock(
                conv, feature_dim, kernel_dim,
                activation=activation_func,
                scaling=1.
            ) for _ in range(num_blocks)
        ]
        body_layers.append(conv(feature_dim, feature_dim, kernel_dim))

        output_layers = [
            UpscaleModule(conv, scale_factor, feature_dim),
            conv(feature_dim, num_channels, kernel_dim)
        ]

        self.input_block = nn.Sequential(*input_layers)
        self.feature_block = nn.Sequential(*body_layers)
        self.output_block = nn.Sequential(*output_layers)

    def forward(self, input_tensor):
        x = self.input_block(input_tensor)
        features = self.feature_block(x)
        features += x
        return self.output_block(features)

    def load_state_dict(self, state_dict, strict=True):
        current_params = self.state_dict()
        for param_name, weights in state_dict.items():
            if param_name in current_params:
                if isinstance(weights, nn.Parameter):
                    weights = weights.data
                try:
                    current_params[param_name].copy_(weights)
                except Exception:
                    if 'output_block' not in param_name:
                        raise RuntimeError(f'Dimension mismatch for {param_name}, '
                                           f'model: {current_params[param_name].size()}, '
                                           f'checkpoint: {weights.size()}')
            elif strict:
                if 'output_block' not in param_name:
                    raise KeyError(f'Unexpected parameter {param_name} in state_dict')