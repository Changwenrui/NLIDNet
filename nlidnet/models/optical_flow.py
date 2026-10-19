import torch
from torch import nn
from torch.nn import functional as F

_BASE_CHANNELS = 64


def init_conv(conv):
    torch.nn.init.kaiming_normal_(conv.weight, a=4.4, nonlinearity="relu")
    torch.nn.init.constant_(conv.bias, 0)


def init_bn(bn):
    torch.nn.init.constant_(bn.weight, 0.1)


class GConv2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding=True):
        super(GConv2d, self).__init__()
        padding = tuple(map(lambda x: (x - 1) // 2, kernel_size)) if padding else 0
        self.conv = nn.Conv2d(
            in_size, out_size, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_size)
        init_conv(self.conv)
        init_bn(self.bn)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class GConv2d_right(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride):
        super(GConv2d_right, self).__init__()
        self.padding = tuple(map(lambda x: x - 1, kernel_size))
        self.padding = [0, self.padding[0], 0, self.padding[1]]
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        init_conv(self.conv)
        init_bn(self.bn)

    def forward(self, x):
        return self.bn(F.relu(self.conv(F.pad(x, self.padding))))


class ResNetBlock(nn.Module):
    def __init__(self, io_size, kernel_size, depth):
        super(ResNetBlock, self).__init__()
        self.conv = nn.ModuleList()
        for _ in range(depth):
            self.conv.append(GConv2d(io_size, io_size, kernel_size))

    def forward(self, x):
        h = x.clone()
        for conv in self.conv:
            h = conv(h)
        return x + h


class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size):
        super(UpsampleBlock, self).__init__()
        self.pad_size = tuple(
            map(
                lambda x: (x - 1) // 2,
                [x for sub in zip(kernel_size, kernel_size) for x in sub],
            )
        )
        self.conv = GConv2d(in_size, out_size, kernel_size, padding=False)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=2)

        chunk_size = 4
        if x.shape[0] > chunk_size:
            chunks = torch.split(x, chunk_size, dim=0)
            padded_chunks = [F.pad(c, self.pad_size, mode="replicate") for c in chunks]
            x = torch.cat(padded_chunks, dim=0)
        else:
            x = F.pad(x, self.pad_size, mode="replicate")

        return self.conv(x)


class EVFlowNet(nn.Module):
    def __init__(self):
        super(EVFlowNet, self).__init__()
        in_size, enc_depth, tr_depth, tr_res_depth = 4, 4, 2, 2
        kernel_size = (3, 3)
        sizes = [4]
        self.enc = nn.ModuleList()
        for i in range(enc_depth):
            sizes.append(_BASE_CHANNELS * 2**i)
            self.enc.append(GConv2d_right(sizes[-2], sizes[-1], kernel_size, 2))
        self.tr = nn.ModuleList()
        for i in range(tr_depth):
            self.tr.append(
                ResNetBlock(sizes[-1], kernel_size=kernel_size, depth=tr_res_depth)
            )
        self.dec = nn.ModuleList()
        self.flow = nn.ModuleList()
        sizes[0] = 32
        for i in range(enc_depth):
            in_size = 2 * sizes[-1 - i] + (2 if i > 0 else 0)
            out_size = sizes[-2 - i]
            self.dec.append(UpsampleBlock(in_size, out_size, kernel_size))
            self.flow.append(nn.Conv2d(out_size, 2, kernel_size=(1, 1)))
            init_conv(self.flow[-1])

    def _get_result(self, flow, outsize):
        return tuple(f[..., : s[0], : s[1]] for f, s in zip(flow, outsize))

    def forward(self, x):
        imsize = (x.size(2), x.size(3))
        outsize = [
            tuple(map(lambda d: d // 2**i, imsize)) for i in range(len(self.enc))
        ][::-1]
        y, skip = [], [x]
        for enc_block in self.enc:
            skip.append(enc_block(skip[-1]))
        h = skip[-1]
        for res in self.tr:
            h = res(h)
        n = len(skip)
        for idx, (s, d, f) in enumerate(zip(skip[n:0:-1], self.dec, self.flow)):
            h = torch.cat((h, s), 1)
            h = d(h)
            h_flow = f(h)
            y.append(torch.tanh(h_flow).mul_(256.0))
            h = torch.cat((h, y[-1]), 1)
        result = self._get_result(y, outsize)
        return result[-1]
