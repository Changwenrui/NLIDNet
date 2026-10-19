import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import functional, layer, base, neuron


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
            step_mode="m",
        )

    def forward(self, x):
        return self.conv(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            step_mode="m",
        )

    def forward(self, x):
        return self.conv(x)


class BN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = layer.BatchNorm2d(channels, step_mode="m")

    def forward(self, x):
        return self.bn(x)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = layer.Linear(in_features, out_features, step_mode="m")

    def forward(self, x):
        return self.fc(x)


class LIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode="m")

    def forward(self, x):
        return self.lif(x)


class SpikingMatmul(nn.Module):
    def __init__(self, mode="r"):
        super().__init__()

    def forward(self, mat1, mat2):
        return torch.matmul(mat1, mat2)


class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=LIF):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(), Conv1x1(in_channels, inner_channels), BN(inner_channels)
        )
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(),
                    Conv3x3(
                        inner_channels,
                        inner_channels,
                        groups=inner_channels // group_size,
                    ),
                    BN(inner_channels),
                )
            )
        self.down = nn.Sequential(
            activation(), Conv1x1(inner_channels, in_channels), BN(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        return self.down(x + x_feat_in) + x_feat_out


class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, patch_size, activation=LIF):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth
        self.register_buffer("firing_rate_x", torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer("firing_rate_attn", torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999
        self.activation_in = activation()
        self.W = layer.Conv2d(
            dim, dim * 2, patch_size, patch_size, bias=False, step_mode="m"
        )
        self.norm = BN(dim * 2)
        self.matmul1 = SpikingMatmul("r")
        self.matmul2 = SpikingMatmul("r")
        self.activation_attn = activation()
        self.activation_out = activation()
        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)
        y = self.W(x)
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = (
            y[:, :, :, : C // self.num_heads, :],
            y[:, :, :, C // self.num_heads :, :],
        )
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (
                1 - self.momentum
            )
        scale1 = 1.0 / torch.sqrt(
            self.firing_rate_x * (self.dim // self.num_heads) + 1e-8
        )
        attn = self.matmul1(y1.transpose(-1, -2), x) * scale1
        attn = self.activation_attn(attn)

        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = (
                self.firing_rate_attn * self.momentum
                + firing_rate_attn * (1 - self.momentum)
            )
        scale2 = 1.0 / torch.sqrt(self.firing_rate_attn * self.lenth + 1e-8)
        out = self.matmul2(y2, attn) * scale2
        out = out.reshape(T, B, C, H, W)
        out = self.activation_out(out)
        out = self.Wproj(out)
        out = self.norm_proj(out)
        return out + x_feat


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        self.norm = BN(out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.activation(x)))


class SpikingResformer(nn.Module):
    def __init__(self, num_classes=16, T=20, in_channels=2):
        super().__init__()
        self.T = T
        planes = [64, 192, 384]
        num_heads = [1, 3, 6]
        patch_sizes = [4, 2, 1]

        self.prologue = nn.Sequential(
            layer.Conv2d(
                in_channels,
                planes[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
                step_mode="m",
            ),
            BN(planes[0]),
        )
        img_size = 224 // 4

        self.layers = nn.Sequential()
        layers_layout = [
            ["DSSA", "GWFFN"] * 1,
            ["DSSA", "GWFFN"] * 2,
            ["DSSA", "GWFFN"] * 3,
        ]
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(
                        planes[idx - 1], planes[idx], stride=2, activation=LIF
                    )
                )
                img_size = img_size // 2
            for name in layers_layout[idx]:
                if name == "DSSA":
                    sub_layers.append(
                        DSSA(
                            planes[idx],
                            num_heads[idx],
                            (img_size // patch_sizes[idx]) ** 2,
                            patch_sizes[idx],
                            activation=LIF,
                        )
                    )
                elif name == "GWFFN":
                    sub_layers.append(GWFFN(planes[idx], group_size=64, activation=LIF))
            self.layers.append(sub_layers)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.classifier = Linear(planes[-1] * T, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(self.T, 224, 224), mode="trilinear", align_corners=False
        )
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        batchsize = x.shape[1]
        x = x.permute(1, 0, 2).contiguous().reshape(batchsize, -1)
        return self.classifier(x)
