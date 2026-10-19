from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from nlidnet.models.optical_flow import EVFlowNet

ROI_X_MIN, ROI_X_MAX = 520, 840
ROI_Y_MIN, ROI_Y_MAX = 0, 600
ROI_W, ROI_H = 320, 600
DELTA_T_US = 2000


def roi_bin_counts(events, t0, start_frame, end_frame):
    x, y, t = (events[:, 0], events[:, 1], events[:, 2])
    frame_idx = torch.floor((t - t0) / DELTA_T_US).long()
    valid = (
        (frame_idx >= start_frame)
        & (frame_idx < end_frame)
        & (x >= ROI_X_MIN)
        & (x < ROI_X_MAX)
        & (y >= ROI_Y_MIN)
        & (y < ROI_Y_MAX)
    )
    relative = frame_idx[valid] - start_frame
    return torch.bincount(relative, minlength=end_frame - start_frame)


def select_frames(counts, start_frame, end_frame, num_frames, mode):
    if mode == "uniform":
        return (
            np.rint(np.linspace(start_frame, end_frame - 1, num_frames))
            .astype(int)
            .tolist()
        )
    edges = np.rint(np.linspace(start_frame, end_frame, num_frames + 1)).astype(int)
    selected = []
    for i in range(num_frames):
        left, right = (int(edges[i]), int(edges[i + 1]))
        right = max(right, left + 1)
        local = counts[left - start_frame : right - start_frame]
        selected.append(left + int(torch.argmax(local).item()))
    return selected


def build_input(events, t0, frame_idx, device):
    t_start = t0 + frame_idx * DELTA_T_US
    t_end = t_start + DELTA_T_US
    x, y, t, p = (events[:, 0], events[:, 1], events[:, 2], events[:, 3])
    mask = (
        (t >= t_start)
        & (t < t_end)
        & (x >= ROI_X_MIN)
        & (x < ROI_X_MAX)
        & (y >= ROI_Y_MIN)
        & (y < ROI_Y_MAX)
    )
    x, y, t, p = (x[mask], y[mask], t[mask], p[mask])
    n_events = int(x.numel())
    flat = torch.zeros(4 * ROI_H * ROI_W, dtype=torch.float32, device=device)
    if n_events == 0:
        return (flat.view(1, 4, ROI_H, ROI_W), 0)
    x = (x - ROI_X_MIN).long().to(device)
    y = (y - ROI_Y_MIN).long().to(device)
    pixel = y * ROI_W + x
    p_long = (1 - p.long()).to(device)
    count_idx = p_long * (ROI_H * ROI_W) + pixel
    flat.scatter_add_(
        0, count_idx, torch.ones_like(count_idx, dtype=torch.float32, device=device)
    )
    ts_norm = ((t - t_start) % DELTA_T_US / float(DELTA_T_US)).to(
        device=device, dtype=torch.float32
    )
    ts_idx = (p_long + 2) * (ROI_H * ROI_W) + pixel
    try:
        flat.scatter_reduce_(0, ts_idx, ts_norm, reduce="amax", include_self=True)
    except AttributeError:
        flat.scatter_(0, ts_idx, ts_norm)
    return (flat.view(1, 4, ROI_H, ROI_W), n_events)


def export_flow(
    event_path,
    checkpoint,
    output,
    start_bin=2950,
    end_bin=3650,
    frames=20,
    selection="segment_peak",
    device="cuda",
):
    if not 0 <= start_bin < end_bin or not 1 <= frames <= end_bin - start_bin:
        raise ValueError("Invalid temporal interval")
    events = load_file(str(event_path), device="cpu")["events"]
    if not len(events):
        raise ValueError("Empty event recording")
    model = EVFlowNet().to(device).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    t0 = events[0, 2].item()
    counts = roi_bin_counts(events, t0, start_bin, end_bin)
    chosen = select_frames(counts, start_bin, end_bin, frames, selection)
    flows = []
    maps = []
    with torch.inference_mode():
        for index in chosen:
            voxel, _ = build_input(events, t0, index, device)
            height, width = voxel.shape[-2:]
            padded = torch.nn.functional.pad(
                voxel, (0, (-width) % 16, 0, (-height) % 16)
            )
            flows.append(model(padded)[0, :, :height, :width].cpu())
            maps.append(voxel[0, :2].sum(0).cpu())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(flows), out / "flow.pt")
    torch.save(torch.stack(maps), out / "event_counts.pt")
    np.save(out / "selected_bins.npy", np.asarray(chosen))
