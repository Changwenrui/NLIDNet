from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from nlidnet.models.optical_flow import EVFlowNet
import argparse
from nlidnet.config import load_config

ROI_X_MIN, ROI_X_MAX = (520, 840)
ROI_Y_MIN, ROI_Y_MAX = (0, 600)
ROI_W, ROI_H = (320, 600)
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
            padded = torch.nn.functional.pad(voxel, (0, -width % 16, 0, -height % 16))
            flows.append(model(padded)[0, :, :height, :width].cpu())
            maps.append(voxel[0, :2].sum(0).cpu())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(flows), out / "flow.pt")
    torch.save(torch.stack(maps), out / "event_counts.pt")
    np.save(out / "selected_bins.npy", np.asarray(chosen))


def export_flow_cache(
    event_path,
    checkpoint,
    output,
    stages=None,
    window_bins=700,
    delta_t_us=2000,
    context_px=64,
    batch_size=4,
    device="cuda",
    roi=(520, 100, 840, 580),
):
    import json
    import math
    from torch.nn import functional as F
    from nlidnet.data.raw import read_events
    from nlidnet.data.representations import flow_voxel
    from nlidnet.protocol import STAGES

    if min(window_bins, delta_t_us, batch_size) < 1 or context_px < 0:
        raise ValueError("Invalid dense optical-flow export settings")
    stages = stages or STAGES
    events = read_events(event_path)
    if not len(events):
        raise ValueError("Cannot extract flow from an empty recording")
    origin = int(events[0, 2])
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720):
        raise ValueError("Target ROI must lie inside the 1280x720 event sensor")
    inference_roi = (
        max(0, x0 - context_px),
        max(0, y0 - context_px),
        min(1280, x1 + context_px),
        min(720, y1 + context_px),
    )
    ix0, iy0, ix1, iy1 = inference_roi
    height, width = (y1 - y0, x1 - x0)
    crop_y, crop_x = (y0 - iy0, x0 - ix0)
    model = EVFlowNet()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    model.to(device).requires_grad_(False).eval()
    out = Path(output)
    for stage, start_bin in stages.items():
        dest = out / f"{stage}_{start_bin}"
        if any(
            (
                (dest / name).exists()
                for name in ("event_count.npy", "flow_uv.npy", "meta.json")
            )
        ):
            raise FileExistsError(dest)
    for stage, start_bin in stages.items():
        dest = out / f"{stage}_{start_bin}"
        dest.mkdir(parents=True, exist_ok=True)
        counts = np.lib.format.open_memmap(
            dest / "event_count.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(window_bins, height, width),
        )
        flows = np.lib.format.open_memmap(
            dest / "flow_uv.npy",
            mode="w+",
            dtype=np.float16,
            shape=(window_bins, 2, height, width),
        )
        for start in range(0, window_bins, batch_size):
            indices = list(range(start, min(start + batch_size, window_bins)))
            voxels = []
            for index in indices:
                begin_us = origin + (start_bin + index) * delta_t_us
                begin, end = np.searchsorted(
                    events[:, 2], [begin_us, begin_us + delta_t_us]
                )
                voxel = flow_voxel(
                    events[begin:end], begin_us, delta_t_us, inference_roi
                )
                count = voxel[
                    :2, crop_y : crop_y + height, crop_x : crop_x + width
                ].sum(axis=0)
                if count.max() > np.iinfo(np.uint16).max:
                    raise ValueError("Per-pixel event count exceeds uint16 storage")
                counts[index] = count.astype(np.uint16)
                voxels.append(voxel)
            inputs = torch.from_numpy(np.stack(voxels)).to(device)
            ih, iw = inputs.shape[-2:]
            padded = F.pad(
                inputs,
                (0, math.ceil(iw / 16) * 16 - iw, 0, math.ceil(ih / 16) * 16 - ih),
            )
            with torch.no_grad():
                predicted = model(padded)[
                    :, :, crop_y : crop_y + height, crop_x : crop_x + width
                ]
            flows[indices] = predicted.cpu().numpy().astype(np.float16)
        counts.flush()
        flows.flush()
        del counts, flows
        meta = {
            "stage": stage,
            "start_frame": start_bin,
            "start_sec_relative_to_recording_t0": start_bin * delta_t_us / 1000000.0,
            "duration_sec": window_bins * delta_t_us / 1000000.0,
            "delta_ms": delta_t_us / 1000,
            "num_bins": window_bins,
            "target_roi": {
                "x_min": x0,
                "y_min": y0,
                "x_max": x1,
                "y_max": y1,
                "height": height,
                "width": width,
            },
            "inference_context_px": context_px,
            "inference_roi": {"x_min": ix0, "y_min": iy0, "x_max": ix1, "y_max": iy1},
            "event_count_shape": [window_bins, height, width],
            "flow_uv_shape": [window_bins, 2, height, width],
            "event_count_dtype": "uint16",
            "flow_uv_dtype": "float16",
            "source_recording": Path(event_path).name,
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(stage, dest, flush=True)
    return out


def add_config(parser):
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--device")
    return parser


def configuration(args):
    cfg = load_config(args.config, args.stage)
    if args.device:
        cfg["device"] = args.device
    return cfg


def main(argv=None):
    p = add_config(argparse.ArgumentParser())
    p.add_argument("--events", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument(
        "--selection", choices=["uniform", "segment_peak"], default="segment_peak"
    )
    p.add_argument("--output", default="outputs/optical_flow")
    p.add_argument(
        "--dense",
        action="store_true",
        help="Export paired event_count.npy and flow_uv.npy for video rendering",
    )
    p.add_argument("--all-stages", action="store_true")
    p.add_argument("--context-px", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args(argv)
    cfg = configuration(args)
    if args.all_stages and (not args.dense):
        p.error("--all-stages requires --dense")
    if args.dense:
        from nlidnet.protocol import STAGES

        stages = STAGES if args.all_stages else {cfg["stage"]: cfg["start_frame"]}
        return export_flow_cache(
            args.events,
            args.checkpoint or cfg["flow_checkpoint"],
            args.output,
            stages,
            cfg["window_bins"],
            cfg["delta_t_us"],
            args.context_px,
            args.batch_size,
            cfg["device"],
        )
    export_flow(
        args.events,
        args.checkpoint or cfg["flow_checkpoint"],
        args.output,
        cfg["start_frame"],
        cfg["start_frame"] + cfg["window_bins"],
        args.frames,
        args.selection,
        cfg["device"],
    )


if __name__ == "__main__":
    main()
