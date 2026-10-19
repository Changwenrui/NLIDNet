import torch
from nlidnet.models.ablations import UniformTemporalAblation


class SparseTemporalAblation(UniformTemporalAblation):
    def __init__(self, sparse_bin_us=200, **kwargs):
        super().__init__(**kwargs)
        if sparse_bin_us not in (20, 200):
            raise ValueError("sparse_bin_us must be 20 or 200")
        self.sparse_bin_us = sparse_bin_us

    def _get_anchor_indices(self, device):
        return torch.linspace(
            0, self.L - 2, steps=self.flow_pair_count, dtype=torch.long, device=device
        )

    def _get_sparse_2d_sampling_plan(self, device):
        anchors = self._get_anchor_indices(device)
        per_anchor = []
        if self.sample_count == self.flow_pair_count:
            per_anchor = [1 for _ in range(self.flow_pair_count)]
        elif self.sample_count == 50 and self.flow_pair_count == 20:
            per_anchor = [3 if i % 2 == 0 else 2 for i in range(self.flow_pair_count)]
        elif self.sample_count == 100 and self.flow_pair_count == 20:
            per_anchor = [5 for _ in range(self.flow_pair_count)]
        else:
            base = self.sample_count // self.flow_pair_count
            rem = self.sample_count % self.flow_pair_count
            per_anchor = [
                base + (1 if i < rem else 0) for i in range(self.flow_pair_count)
            ]
        frame_indices = []
        sub_slots = []
        for anchor_idx, count in enumerate(per_anchor):
            for s in range(count):
                frame_indices.append(anchors[anchor_idx])
                sub_slots.append(s)
        frame_indices = torch.stack(frame_indices).long()
        sub_slots = torch.tensor(sub_slots, dtype=torch.long, device=device)
        if frame_indices.numel() != self.sample_count:
            raise RuntimeError(
                f"Sparse sampling plan length mismatch: got {frame_indices.numel()}, expected {self.sample_count}"
            )
        return (frame_indices, sub_slots)

    def _get_sparse_flow_sampling_plan(self, device):
        anchors = self._get_anchor_indices(device).clamp(0, self.L - 2)
        target_frames = torch.stack([anchors, anchors + 1], dim=-1).view(-1)
        target_sub_slots = torch.zeros_like(
            target_frames, dtype=torch.long, device=device
        )
        return (target_frames, target_sub_slots)

    def build_uniform_event_frames(
        self, events_tensor, events_mask, t_start_offsets
    ) -> torch.Tensor:
        B, N, _ = events_tensor.shape
        device = events_tensor.device
        base_x_min, base_x_max = (520, 840)
        base_y_min, base_y_max = (100, 580)
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
        sparse_us = self.sparse_bin_us
        slots_per_frame = self.delta_t // sparse_us
        sample_frames, sample_sub_slots = self._get_sparse_2d_sampling_plan(device)
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
            idx_fine = torch.div(t_rel, self.delta_t, rounding_mode="floor").long()
            sub_us = t_rel - idx_fine.to(t_rel.dtype) * float(self.delta_t)
            sub_slot = torch.div(sub_us, sparse_us, rounding_mode="floor").long()
            in_window_mask = (
                (idx_fine >= 0)
                & (idx_fine < self.L)
                & (sub_slot >= 0)
                & (sub_slot < slots_per_frame)
            )
            idx_fine_safe = idx_fine.clamp(0, self.L - 1)
            sub_slot_safe = sub_slot.clamp(0, slots_per_frame - 1)
            frame_sub_map = torch.full(
                (self.L, slots_per_frame), -1, dtype=torch.long, device=device
            )
            frame_sub_map[sample_frames, sample_sub_slots] = torch.arange(
                self.sample_count, device=device
            )
            coarse_idx = frame_sub_map[idx_fine_safe, sub_slot_safe]
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
        base_x_min, base_x_max = (520, 840)
        base_y_min, base_y_max = (100, 580)
        W = base_x_max - base_x_min
        H = base_y_max - base_y_min
        sparse_us = self.sparse_bin_us
        slots_per_frame = self.delta_t // sparse_us
        target_frames_one, target_sub_slots_one = self._get_sparse_flow_sampling_plan(
            device
        )
        flow_frames = self.flow_pair_count * 2
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
            idx_fine = torch.div(t_rel, self.delta_t, rounding_mode="floor").long()
            sub_us = t_rel - idx_fine.to(t_rel.dtype) * float(self.delta_t)
            sub_slot = torch.div(sub_us, sparse_us, rounding_mode="floor").long()
            t_norm = (sub_us - sub_slot.to(sub_us.dtype) * float(sparse_us)) / float(
                sparse_us
            )
            in_window_mask = (
                (idx_fine >= 0)
                & (idx_fine < self.L)
                & (sub_slot >= 0)
                & (sub_slot < slots_per_frame)
            )
            idx_fine_safe = idx_fine.clamp(0, self.L - 1)
            sub_slot_safe = sub_slot.clamp(0, slots_per_frame - 1)
            frame_sub_map = torch.full(
                (self.L, slots_per_frame), -1, dtype=torch.long, device=device
            )
            frame_sub_map[target_frames_one, target_sub_slots_one] = torch.arange(
                flow_frames, device=device
            )
            coarse_idx = frame_sub_map[idx_fine_safe, sub_slot_safe]
            valid_mask = valid_mask & in_window_mask & (coarse_idx >= 0)
            x_idx = (x - x_min).long().clamp(0, W - 1)
            y_idx = (y - y_min).long().clamp(0, H - 1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
            C_flow = 4
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
            t_norm_flat = t_norm.view(-1)[valid_mask_flat].clamp(0.0, 1.0)
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
        return flat_grid.view(B, C_flow, flow_frames, H, W)
