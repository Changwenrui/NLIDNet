from nlidnet.models.encoders import PatternEncoder, VelocityEncoder
from nlidnet.models.checkpoints import CheckpointCompatible
import torch
from torch import nn
from torch.nn import functional as F
from nlidnet.models.time_attention import GaussianSmoothing1D
from nlidnet.models.event_representation import AttentiveEventRepresentation
from nlidnet.models.temporal.encoder import TemporalEncoder


class NoFlowAblation(CheckpointCompatible):

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
        pattern = PatternEncoder(ev2d_chans, resnet_checkpoint)
        self.event_downsampler = pattern.downsampler
        self.pattern_encoder = pattern.backbone
        self.pattern_feature_dim = pattern.output_dim
        self.velocity_feature_dim = 0
        self.fusion_head = nn.Sequential(
            nn.Linear(self.pattern_feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_1d, events_tensor, events_mask, t_start_offsets):
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
        fused_feat = pattern_embedding
        logits = self.fusion_head(fused_feat)
        return (logits, timeattention)

    build_event_frames = AttentiveEventRepresentation.build_event_frames


class UniformTemporalAblation(CheckpointCompatible):
    def __init__(
        self,
        num_classes=16,
        sample_count=20,
        flow_pair_count=20,
        ev2d_chans=2,
        img_size=224,
        delta_t=2000,
        resnet_checkpoint=None,
        flow_checkpoint=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.sample_count = sample_count
        self.flow_pair_count = flow_pair_count
        self.L = 700
        self.ev2d_chans = ev2d_chans
        self.img_size = img_size
        self.delta_t = delta_t

        pattern = PatternEncoder(ev2d_chans, resnet_checkpoint)
        self.event_downsampler = pattern.downsampler
        self.pattern_encoder = pattern.backbone
        self.pattern_feature_dim = pattern.output_dim

        velocity = VelocityEncoder(flow_checkpoint)
        self.optical_flow = velocity.flow_solver
        self.velocity_encoder = velocity.features
        self.velocity_feature_dim = velocity.output_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(self.pattern_feature_dim + self.velocity_feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _get_uniform_sample_indices(self, device):

        return torch.linspace(
            0, self.L - 1, steps=self.sample_count, dtype=torch.long, device=device
        )

    def _get_flow_key_indices_from_sampled_frames(self, sampled_indices):

        if self.sample_count == self.flow_pair_count:
            pos = torch.arange(self.flow_pair_count, device=sampled_indices.device)
        else:
            pos = torch.linspace(
                0,
                self.sample_count - 1,
                steps=self.flow_pair_count,
                dtype=torch.long,
                device=sampled_indices.device,
            )

        key_indices = sampled_indices[pos].clamp(0, self.L - 2)
        return key_indices

    def build_uniform_event_frames(
        self, events_tensor, events_mask, t_start_offsets
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

        stride_b = self.ev2d_chans * self.sample_count * H * W
        stride_c = self.sample_count * H * W
        stride_t = H * W
        stride_y = W

        sampled_indices = self._get_uniform_sample_indices(device)

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

            in_window_mask = (idx_fine >= 0) & (idx_fine < self.L)
            idx_fine_safe = idx_fine.clamp(0, self.L - 1)

            frame_map = torch.full((self.L,), -1, dtype=torch.long, device=device)
            frame_map[sampled_indices] = torch.arange(self.sample_count, device=device)

            coarse_idx = frame_map[idx_fine_safe]
            valid_mask = valid_mask & in_window_mask & (coarse_idx >= 0)

            x_idx = (x - x_min).long().clamp(0, W - 1)
            y_idx = (y - y_min).long().clamp(0, H - 1)

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

            final_idx = (
                batch_idx * stride_b
                + p.long() * stride_c
                + coarse_idx.clamp(0, self.sample_count - 1) * stride_t
                + y_idx * stride_y
                + x_idx
            )

            final_idx_flat = final_idx.view(-1)
            valid_mask_flat = valid_mask.view(-1)

        flat_grid = torch.zeros(B * stride_b, device=device, dtype=torch.float32)
        ones_flat = torch.ones_like(final_idx_flat, dtype=torch.float32)
        flat_grid.scatter_add_(
            0, final_idx_flat[valid_mask_flat], ones_flat[valid_mask_flat]
        )

        return flat_grid.view(B, self.ev2d_chans, self.sample_count, H, W)

    def build_uniform_flow_voxels(
        self, events_tensor, events_mask, t_start_offsets
    ) -> torch.Tensor:
        B, N, _ = events_tensor.shape
        device = events_tensor.device

        base_x_min, base_x_max = 520, 840
        base_y_min, base_y_max = 100, 580
        W = base_x_max - base_x_min
        H = base_y_max - base_y_min

        sampled_indices = self._get_uniform_sample_indices(device)
        key_indices = self._get_flow_key_indices_from_sampled_frames(sampled_indices)
        target_indices_one = torch.stack([key_indices, key_indices + 1], dim=-1).view(
            -1
        )
        target_indices = target_indices_one.unsqueeze(0).expand(B, -1)

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
            idx_fine = (t_rel / self.delta_t).long()
            in_window_mask = (idx_fine >= 0) & (idx_fine < self.L)
            idx_fine_safe = idx_fine.clamp(0, self.L - 1)

            is_valid_frame = (
                idx_fine_safe.unsqueeze(-1) == target_indices.unsqueeze(1)
            ).any(dim=-1)
            valid_mask = valid_mask & in_window_mask & is_valid_frame

            frame_map = torch.full((B, self.L), -1, device=device, dtype=torch.long)
            batch_indices = (
                torch.arange(B, device=device)
                .unsqueeze(1)
                .expand(B, self.flow_pair_count * 2)
            )
            frame_map[batch_indices, target_indices] = torch.arange(
                self.flow_pair_count * 2, device=device
            ).expand(B, self.flow_pair_count * 2)
            coarse_idx = torch.gather(frame_map, 1, idx_fine_safe)

            t_norm = (t_rel % self.delta_t) / float(self.delta_t)

            x_idx = (x - x_min).long().clamp(0, W - 1)
            y_idx = (y - y_min).long().clamp(0, H - 1)

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

            C_flow = 4
            flow_frames = self.flow_pair_count * 2
            stride_b = C_flow * flow_frames * H * W
            stride_c = flow_frames * H * W
            stride_t = H * W
            stride_y = W

            p_long = 1 - p.long()

            idx_count = (
                batch_idx * stride_b
                + p_long * stride_c
                + coarse_idx.clamp(0, flow_frames - 1) * stride_t
                + y_idx * stride_y
                + x_idx
            )

            idx_ts = (
                batch_idx * stride_b
                + (p_long + 2) * stride_c
                + coarse_idx.clamp(0, flow_frames - 1) * stride_t
                + y_idx * stride_y
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

        return flat_grid.view(B, C_flow, self.flow_pair_count * 2, H, W)

    def forward(self, events_tensor, events_mask, t_start_offsets):
        B_size = events_tensor.shape[0]

        x_2d = self.build_uniform_event_frames(
            events_tensor, events_mask, t_start_offsets
        )

        B_frames, C_img, _, H, W = x_2d.shape
        x_frames = x_2d.permute(0, 2, 1, 3, 4).reshape(-1, C_img, H, W)

        x_frames_down = self.event_downsampler(x_frames)
        pattern_features = self.pattern_encoder(x_frames_down)
        pattern_embedding = pattern_features.view(B_size, self.sample_count, -1).mean(
            dim=1
        )

        flow_voxels = self.build_uniform_flow_voxels(
            events_tensor, events_mask, t_start_offsets
        )
        _, _, flow_frames, _, _ = flow_voxels.shape

        flow_input = flow_voxels.permute(0, 2, 1, 3, 4).reshape(
            B_size * flow_frames, 4, H, W
        )

        with torch.no_grad():
            self.optical_flow.eval()
            flow_maps = self.optical_flow(flow_input)

        encoded_flow = self.velocity_encoder(flow_maps)
        paired_flow_features = encoded_flow.view(B_size, self.flow_pair_count, 2, -1)
        flow_sequence = paired_flow_features.mean(dim=2)

        velocity_embedding = flow_sequence.mean(dim=1)

        fused_feat = torch.cat([pattern_embedding, velocity_embedding], dim=-1)
        logits = self.fusion_head(fused_feat)

        return logits
