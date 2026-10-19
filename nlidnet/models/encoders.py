import torch
from torch import nn
from torchvision import models
from nlidnet.models.optical_flow import EVFlowNet


class PatternEncoder(nn.Module):
    def __init__(self, channels=2, checkpoint=None):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                ]
            )
        self.downsampler = nn.Sequential(*layers)
        self.backbone = models.resnet34(weights=None)
        if checkpoint is not None:
            self.backbone.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        if channels != 3:
            weight = self.backbone.conv1.weight.detach().mean(dim=1, keepdim=True)
            self.backbone.conv1 = nn.Conv2d(
                channels, 64, 7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(weight.repeat(1, channels, 1, 1))
        self.backbone.fc = nn.Identity()
        self.output_dim = 512

    def forward(self, frames):
        batch, channels, segments, height, width = frames.shape
        images = frames.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        features = self.backbone(self.downsampler(images))
        return features.reshape(batch, segments, self.output_dim).mean(dim=1)


class VelocityEncoder(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.flow_solver = EVFlowNet()
        if checkpoint is not None:
            self.flow_solver.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        self.flow_solver.requires_grad_(False).eval()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.output_dim = 64

    def train(self, mode=True):
        super().train(mode)
        self.flow_solver.eval()
        return self

    def forward(self, voxels, segment_weights):
        batch, channels, pairs, height, width = voxels.shape
        segments = segment_weights.shape[1]
        if pairs != 2 * segments:
            raise ValueError(
                "Velocity encoding requires two adjacent voxels per segment"
            )
        inputs = voxels.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        with torch.no_grad():
            flow = self.flow_solver(inputs)
        encoded = (
            self.features(flow).reshape(batch, segments, 2, self.output_dim).mean(dim=2)
        )
        return (encoded * segment_weights.unsqueeze(-1)).sum(dim=1)


class ResponseHeads(nn.Module):
    def __init__(self, input_dim=576, embedding_dim=256, classes=16):
        super().__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, classes)
        self.physics = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.GELU(), nn.Linear(64, 3)
        )

    def forward(self, pattern, velocity):
        embedding = self.activation(
            self.projection(torch.cat([pattern, velocity], dim=-1))
        )
        return (
            self.classifier(self.dropout(embedding)),
            embedding,
            self.physics(embedding),
        )
