from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from nlidnet.models.temporal.encoder import TemporalEncoder


class GaussianSmoothing1D(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 21, sigma: float = 3.0):
        super().__init__()
        x = torch.arange(kernel_size).float()
        center = kernel_size // 2
        kernel = torch.exp(-((x - center) ** 2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
        self.register_buffer("weight", kernel)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        x_padded = F.pad(x, (self.padding, self.padding), mode="replicate")
        return F.conv1d(x_padded, self.weight, groups=self.groups)


@dataclass
class TemporalSelection:
    raw_logits: torch.Tensor
    smoothed_logits: torch.Tensor
    fine_weights: torch.Tensor
    segment_weights: torch.Tensor
    key_indices: torch.Tensor


class TimeAttention(nn.Module):
    def __init__(self, channels=6, segments=20, bins_per_segment=35):
        super().__init__()
        self.segments = segments
        self.bins_per_segment = bins_per_segment
        self.length = segments * bins_per_segment
        self.encoder = TemporalEncoder(
            {
                "Data_shape": (None, channels, self.length),
                "emb_size": 32,
                "num_heads": 2,
                "dim_ff": 64,
                "Fix_pos_encode": "tAPE",
                "Rel_pos_encode": "eRPE",
                "dropout": 0.1,
            },
            k=bins_per_segment,
        )
        self.smoother = GaussianSmoothing1D(channels=1, kernel_size=15, sigma=3.0)

    def forward(self, signals):
        if signals.ndim != 3 or signals.shape[-1] != self.length:
            raise ValueError(f"Expected [B,C,{self.length}] ROI signals")
        raw = self.encoder(signals)
        smooth = self.smoother(raw.reshape(-1, 1, self.length)).reshape(-1, self.length)
        blocks = smooth.reshape(-1, self.segments, self.bins_per_segment)
        local = torch.softmax(blocks, dim=-1)
        segment_weights = torch.softmax(blocks.sum(dim=-1), dim=-1)
        positions = torch.arange(self.segments, device=signals.device).unsqueeze(0)
        keys = (positions * self.bins_per_segment + local.argmax(dim=-1)).clamp(
            0, self.length - 2
        )
        return TemporalSelection(
            raw, smooth, local.reshape(-1, self.length), segment_weights, keys
        )
