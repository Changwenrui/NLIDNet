import torch
from torch import nn


class AttentiveEventRepresentation(nn.Module):
    def __init__(self, T=20, k=35, delta_t=2000, channels=2):
        super().__init__()
        self.T, self.k, self.L = T, k, T * k
        self.delta_t, self.ev2d_chans = delta_t, channels

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
