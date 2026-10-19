import torch
from torch import nn
from torch.nn import functional as F


class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6.0


class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6.0


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = (
            act(inplace=True)
            if hasattr(act, "inplace") or isinstance(act, type)
            else act()
        )

        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = (
            act(inplace=True)
            if hasattr(act, "inplace") or isinstance(act, type)
            else act()
        )
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size),
            )
        elif stride == 2:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=out_size,
                    kernel_size=1,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        skip = x if self.skip is None else self.skip(x)
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        return out + skip


class ResBlock_3d(nn.Module):
    def __init__(self, nf, k):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = nn.Conv3d(nf, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.dcn1 = nn.Conv3d(nf * 2, nf, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=k, out_channels=k, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = (self.dcn1(self.lrelu(self.dcn0(x))) + x).mean(dim=1)
        return self.conv2d(x1)


class MobileNetV3_Chem(nn.Module):
    def __init__(self, num_classes=16, framenum=20):
        super(MobileNetV3_Chem, self).__init__()
        self.conv3d = ResBlock_3d(2, framenum)
        self.conv1 = nn.Conv2d(
            framenum, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        act = nn.Hardswish
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.outputclassifier = nn.Sequential(self.drop, nn.Linear(1280, num_classes))

    def forward(self, x):
        out_3d = self.conv3d(x)
        out = self.hs1(self.bn1(self.conv1(out_3d)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))
        return self.outputclassifier(out)


class AlexNet(nn.Module):
    def __init__(self, in_channels=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, 512)

    def forward(self, x):
        return self.fc(self.adaptive_pool(self.features(x)).view(x.size(0), -1))


class AlexNetLSTM(nn.Module):
    def __init__(self, num_classes=16, hidden_size=256, num_layers=2, in_channels=2):
        super(AlexNetLSTM, self).__init__()
        self.alexnet = AlexNet(in_channels=in_channels)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.size()
        features = [self.alexnet(x[:, :, t, :, :]) for t in range(num_frames)]
        lstm_out, _ = self.lstm(torch.stack(features, dim=1))
        return self.fc(lstm_out[:, -1, :])
