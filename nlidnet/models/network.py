from nlidnet.models.checkpoints import CheckpointCompatible
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from nlidnet.models.optical_flow import EVFlowNet
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


class NLIDNet(CheckpointCompatible):

    def __init__(
        self,
        num_classes=16,
        in_chans_1d=6,
        T=20,
        k=35,
        ev2d_chans=2,
        img_size=224,
        pattern_dims=96,
        delta_t=2000,
        resnet_checkpoint=None,
        flow_checkpoint=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.T = T
        self.k = k
        self.L = T * k
        self.ev2d_chans = ev2d_chans
        self.img_size = img_size
        self.delta_t = delta_t

        conv_config = {
            "Data_shape": (None, in_chans_1d, self.L),
            "emb_size": 32,
            "num_heads": 2,
            "dim_ff": 64,
            "Fix_pos_encode": "tAPE",
            "Rel_pos_encode": "eRPE",
            "dropout": 0.1,
        }
        self.temporal_encoder = TemporalEncoder(conv_config, k=k)

        self.gaussian_smooth = GaussianSmoothing1D(
            channels=1, kernel_size=15, sigma=3.0
        )

        self.event_downsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ev2d_chans,
                out_channels=self.ev2d_chans,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.BatchNorm2d(self.ev2d_chans),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.ev2d_chans,
                out_channels=self.ev2d_chans,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.BatchNorm2d(self.ev2d_chans),
            nn.GELU(),
            nn.Conv2d(
                self.ev2d_chans, self.ev2d_chans, kernel_size=3, padding=1, stride=2
            ),
            nn.BatchNorm2d(self.ev2d_chans),
            nn.GELU(),
        )
        self.pattern_feature_dim = 512
        resnet = models.resnet34(weights=None)
        if resnet_checkpoint is not None:
            resnet.load_state_dict(
                torch.load(resnet_checkpoint, map_location="cpu", weights_only=True),
                strict=True,
            )

        if ev2d_chans != 3:
            old_conv1_weight = resnet.conv1.weight.data.clone()
            resnet.conv1 = nn.Conv2d(
                ev2d_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                mean_weight = old_conv1_weight.mean(dim=1, keepdim=True)
                resnet.conv1.weight.copy_(mean_weight.repeat(1, ev2d_chans, 1, 1))
        resnet.fc = nn.Identity()
        self.pattern_encoder = resnet

        self.optical_flow = EVFlowNet()
        if flow_checkpoint is not None:
            self.optical_flow.load_state_dict(
                torch.load(flow_checkpoint, map_location="cpu", weights_only=True),
                strict=True,
            )

        self.optical_flow.eval()
        for param in self.optical_flow.parameters():
            param.requires_grad = False

        self.velocity_feature_dim = 64
        self.velocity_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.feature_projection = nn.Linear(
            self.pattern_feature_dim + self.velocity_feature_dim, 256
        )
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

        self.physics_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3)
        )

    def build_event_frames(
        self, timeattention, events_tensor, events_mask, t_start_offsets
    ) -> torch.Tensor:
        B, N, _ = events_tensor.shape
        device = events_tensor.device
        base_x_min, base_x_max = 520, 840
        base_y_min, base_y_max = 100, 580
        W = base_x_max - base_x_min
        H = base_y_max - base_y_min

        if self.training:
            shift_x = torch.randint(-10, 11, (B, 1), device=device)
            shift_y = torch.randint(-10, 11, (B, 1), device=device)
        else:
            shift_x = torch.zeros((B, 1), device=device, dtype=torch.long)
            shift_y = torch.zeros((B, 1), device=device, dtype=torch.long)

        x_min = base_x_min + shift_x
        x_max = base_x_max + shift_x
        y_min = base_y_min + shift_y
        y_max = base_y_max + shift_y
        stride_b, stride_c, stride_t, stride_y = (
            self.ev2d_chans * self.T * H * W,
            self.T * H * W,
            H * W,
            W,
        )

        with torch.no_grad():
            x, y, t, p = (
                events_tensor[..., 0],
                events_tensor[..., 1],
                events_tensor[..., 2],
                events_tensor[..., 3],
            )
            valid_mask = (
                events_mask.bool()
                & (x >= x_min)
                & (x < x_max)
                & (y >= y_min)
                & (y < y_max)
            )
            t_rel = t - t_start_offsets.unsqueeze(1)
            idx_fine = (t_rel / self.delta_t).long()
            time_mask = (idx_fine >= 0) & (idx_fine < self.L)
            valid_mask = valid_mask & time_mask
            idx_fine_safe = idx_fine.clamp(0, self.L - 1)
            idx_coarse = (idx_fine_safe // self.k).clamp(0, self.T - 1)
            x_idx = (x - x_min).long().clamp(0, W - 1)
            y_idx = (y - y_min).long().clamp(0, H - 1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
            final_idx = (
                (batch_idx * stride_b)
                + (p.long() * stride_c)
                + (idx_coarse * stride_t)
                + (y_idx * stride_y)
                + x_idx
            )
            final_idx_flat = final_idx.view(-1)
            valid_mask_flat = valid_mask.view(-1)

        ev_weight = torch.gather(timeattention, dim=1, index=idx_fine_safe)
        vals_flat = ev_weight.view(-1)
        flat_grid = torch.zeros(B * stride_b, device=device, dtype=timeattention.dtype)
        flat_grid.scatter_add_(
            0, final_idx_flat[valid_mask_flat], vals_flat[valid_mask_flat]
        )
        return flat_grid.view(B, self.ev2d_chans, self.T, H, W)

    def build_flow_voxels(
        self, events_tensor, events_mask, t_start_offsets, key_indices
    ) -> torch.Tensor:
        B, N, _ = events_tensor.shape
        device = events_tensor.device
        base_x_min, base_x_max = 520, 840
        base_y_min, base_y_max = 100, 580
        W = base_x_max - base_x_min
        H = base_y_max - base_y_min
        target_indices = torch.stack([key_indices, key_indices + 1], dim=-1).view(B, -1)

        if self.training:
            shift_x = torch.randint(-10, 11, (B, 1), device=device)
            shift_y = torch.randint(-10, 11, (B, 1), device=device)
        else:
            shift_x = torch.zeros((B, 1), device=device, dtype=torch.long)
            shift_y = torch.zeros((B, 1), device=device, dtype=torch.long)

        x_min = base_x_min + shift_x
        x_max = base_x_max + shift_x
        y_min = base_y_min + shift_y
        y_max = base_y_max + shift_y

        with torch.no_grad():
            x, y, t, p = (
                events_tensor[..., 0],
                events_tensor[..., 1],
                events_tensor[..., 2],
                events_tensor[..., 3],
            )
            valid_mask = (
                events_mask.bool()
                & (x >= x_min)
                & (x < x_max)
                & (y >= y_min)
                & (y < y_max)
            )
            t_rel = t - t_start_offsets.unsqueeze(1)
            idx_fine = (t_rel / self.delta_t).long().clamp(0, self.L - 1)
            is_valid_frame = (
                idx_fine.unsqueeze(-1) == target_indices.unsqueeze(1)
            ).any(dim=-1)
            valid_mask = valid_mask & is_valid_frame
            frame_map = torch.full((B, self.L), -1, device=device, dtype=torch.long)
            batch_indices = (
                torch.arange(B, device=device).unsqueeze(1).expand(B, 2 * self.T)
            )
            frame_map[batch_indices, target_indices] = torch.arange(
                2 * self.T, device=device
            ).expand(B, 2 * self.T)
            coarse_idx = torch.gather(frame_map, 1, idx_fine)
            t_norm = (t_rel % self.delta_t) / float(self.delta_t)
            x_idx = (x - x_min).long().clamp(0, W - 1)
            y_idx = (y - y_min).long().clamp(0, H - 1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
            C_flow = 4
            stride_b, stride_c, stride_t, stride_y = (
                C_flow * (2 * self.T) * H * W,
                (2 * self.T) * H * W,
                H * W,
                W,
            )
            p_long = 1 - p.long()
            idx_count = (
                (batch_idx * stride_b)
                + (p_long * stride_c)
                + (coarse_idx * stride_t)
                + (y_idx * stride_y)
                + x_idx
            )
            idx_ts = (
                (batch_idx * stride_b)
                + ((p_long + 2) * stride_c)
                + (coarse_idx * stride_t)
                + (y_idx * stride_y)
                + x_idx
            )
            valid_mask_flat = valid_mask.view(-1)
            idx_count_flat = idx_count.view(-1)[valid_mask_flat]
            idx_ts_flat = idx_ts.view(-1)[valid_mask_flat]
            t_norm_flat = t_norm.view(-1)[valid_mask_flat]
            flat_grid = torch.zeros(
                B * stride_b, device=device, dtype=events_tensor.dtype
            )
            ones_flat = torch.ones_like(idx_count_flat, dtype=torch.float32)
            flat_grid.scatter_add_(0, idx_count_flat, ones_flat)
            try:
                flat_grid.scatter_reduce_(
                    0, idx_ts_flat, t_norm_flat, reduce="amax", include_self=True
                )
            except AttributeError:
                flat_grid.scatter_(0, idx_ts_flat, t_norm_flat)
        return flat_grid.view(B, C_flow, 2 * self.T, H, W)

    def forward(self, x_1d, events_tensor, events_mask, t_start_offsets):

        if x_1d.ndim != 3 or x_1d.shape[-1] != self.L:
            raise ValueError(
                f"Expected [B, C, {self.L}] ROI counts, got {tuple(x_1d.shape)}"
            )
        raw_logits = self.temporal_encoder(x_1d)
        B_size = raw_logits.shape[0]

        smooth_logits = self.gaussian_smooth(raw_logits.view(B_size, 1, self.L))

        attn_local = F.softmax(smooth_logits.view(B_size, self.T, self.k), dim=-1)
        timeattention = attn_local.view(B_size, self.L)

        E_m = smooth_logits.view(B_size, self.T, self.k).sum(dim=-1)
        coarse_attention = F.softmax(E_m, dim=-1)

        x_2d = self.build_event_frames(
            timeattention, events_tensor, events_mask, t_start_offsets
        )
        B_frames, C_img, _, H, W = x_2d.shape
        x_frames = x_2d.permute(0, 2, 1, 3, 4).reshape(-1, C_img, H, W)
        x_frames_down = self.event_downsampler(x_frames)
        pattern_features = self.pattern_encoder(x_frames_down)
        pattern_embedding = pattern_features.view(B_size, self.T, -1).mean(dim=1)

        _, j_star = torch.max(attn_local, dim=-1)
        m_indices = torch.arange(self.T, device=raw_logits.device).unsqueeze(0)
        key_indices = (m_indices * self.k + j_star).clamp(0, self.L - 2)

        flow_voxels = self.build_flow_voxels(
            events_tensor, events_mask, t_start_offsets, key_indices
        )
        flow_input = flow_voxels.permute(0, 2, 1, 3, 4).reshape(
            B_size * (2 * self.T), 4, H, W
        )

        with torch.no_grad():
            self.optical_flow.eval()
            flow_maps = self.optical_flow(flow_input)

        encoded_flow = self.velocity_encoder(flow_maps)
        paired_flow_features = encoded_flow.view(B_size, self.T, 2, -1)
        flow_sequence = paired_flow_features.mean(dim=2)
        velocity_embedding = (flow_sequence * coarse_attention.unsqueeze(-1)).sum(dim=1)

        fused_feat = torch.cat([pattern_embedding, velocity_embedding], dim=-1)

        feat_256 = self.feature_projection(fused_feat)
        feat_256 = self.feature_activation(feat_256)

        feat_drop = self.feature_dropout(feat_256)
        logits = self.classifier(feat_drop)

        phys_pred = self.physics_head(feat_256)

        return logits, timeattention, feat_256, phys_pred
