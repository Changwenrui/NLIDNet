import numpy as np
from nlidnet.data.raw import validate_events


DEFAULT_ROI = (520, 0, 840, 600)
COUNT_CHANNELS = ("negative", "positive")
TIME_SURFACE_CHANNELS = ("positive", "negative")
FLOW_CHANNELS = ("positive_count", "negative_count", "positive_time", "negative_time")


def local_events(events, roi=DEFAULT_ROI):
    events = validate_events(events)
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have positive width and height")
    keep = (
        (events[:, 0] >= x0)
        & (events[:, 0] < x1)
        & (events[:, 1] >= y0)
        & (events[:, 1] < y1)
    )
    local = events[keep].copy()
    local[:, 0] -= x0
    local[:, 1] -= y0
    return local, y1 - y0, x1 - x0


def event_frame(events, roi=DEFAULT_ROI, accumulation="count"):
    local, height, width = local_events(events, roi)
    image = np.zeros((2, height, width), dtype=np.float32)
    if not len(local):
        return image
    if accumulation == "count":
        np.add.at(image, (local[:, 3], local[:, 1], local[:, 0]), 1)
    elif accumulation == "last":
        pixel = local[:, 1] * width + local[:, 0]
        _, reverse = np.unique(pixel[::-1], return_index=True)
        last = len(pixel) - 1 - reverse
        chosen = local[last]
        image[chosen[:, 3], chosen[:, 1], chosen[:, 0]] = 1
    else:
        raise ValueError("accumulation must be count or last")
    return image


def time_surface(events, roi=DEFAULT_ROI, normalize=True):
    events = validate_events(events)
    local, height, width = local_events(events, roi)
    signed = np.zeros((height, width), dtype=np.float32)
    if not len(local):
        return np.stack([signed, signed])
    duration = int(events[-1, 2]) - int(events[0, 2])
    relative = (
        (local[:, 2] - events[0, 2]) / duration if duration else np.zeros(len(local))
    )
    values = (local[:, 3] * 2 - 1) * (np.exp2(relative) - 1)
    pixels = local[:, 1] * width + local[:, 0]
    _, reverse = np.unique(pixels[::-1], return_index=True)
    last = len(pixels) - 1 - reverse
    signed.reshape(-1)[pixels[last]] = values[last]
    image = np.stack([signed.clip(min=0), (-signed).clip(min=0)])
    if normalize:
        scale = image.max() - image.min()
        if scale > 0:
            image = (image - image.min()) / scale
    return image


def voxel_grid(events, roi=DEFAULT_ROI, bins=5, normalize=False):
    if bins < 1:
        raise ValueError("bins must be positive")
    events = validate_events(events)
    local, height, width = local_events(events, roi)
    output = np.zeros((bins, height, width), dtype=np.float32)
    if not len(local):
        return output
    duration = int(events[-1, 2]) - int(events[0, 2])
    times = (
        (bins - 1) * (local[:, 2] - events[0, 2]) / duration
        if duration
        else np.zeros(len(local))
    )
    lower = np.floor(times).astype(np.int64)
    signs = local[:, 3] * 2 - 1
    for index in (lower, lower + 1):
        valid = (index >= 0) & (index < bins)
        weights = signs * (1 - np.abs(index - times))
        np.add.at(
            output, (index[valid], local[valid, 1], local[valid, 0]), weights[valid]
        )
    if normalize:
        mask = output != 0
        values = output[mask]
        if len(values):
            std = values.std(ddof=1) if len(values) > 1 else 0
            output[mask] = (
                (values - values.mean()) / std if std > 0 else values - values.mean()
            )
    return output


def flow_voxel(events, start_us, delta_t_us=2000, roi=DEFAULT_ROI):
    if delta_t_us <= 0:
        raise ValueError("delta_t_us must be positive")
    local, height, width = local_events(events, roi)
    keep = (local[:, 2] >= start_us) & (local[:, 2] < start_us + delta_t_us)
    local = local[keep]
    output = np.zeros((4, height, width), dtype=np.float32)
    if len(local):
        channel = 1 - local[:, 3]
        np.add.at(output, (channel, local[:, 1], local[:, 0]), 1)
        normalized = (local[:, 2] - start_us) / delta_t_us
        np.maximum.at(output, (channel + 2, local[:, 1], local[:, 0]), normalized)
    return output


def frame_sequence(
    events,
    frames=20,
    representation="time_surface",
    roi=DEFAULT_ROI,
    partition="count",
    shift_fraction=0.0,
    sample_fraction=1.0,
    voxel_bins=5,
):
    events = validate_events(events)
    if frames < 1 or not 0 <= shift_fraction < 1 or not 0 < sample_fraction <= 1:
        raise ValueError("Invalid frame count or event sampling fractions")
    channels = voxel_bins if representation == "voxel_grid" else 2
    output = np.zeros(
        (frames, channels, roi[3] - roi[1], roi[2] - roi[0]), dtype=np.float32
    )
    operators = {
        "time_surface": time_surface,
        "event_frame": event_frame,
        "voxel_grid": voxel_grid,
    }
    if representation not in operators:
        raise ValueError(f"Unknown event representation: {representation}")
    if not len(events):
        return output
    if partition == "count":
        gap = len(events) // frames
        boundaries = [
            (
                i * gap + int(shift_fraction * gap),
                i * gap + int(shift_fraction * gap) + int(gap * sample_fraction),
            )
            for i in range(frames)
        ]
    elif partition == "time":
        edges = np.linspace(int(events[0, 2]), int(events[-1, 2]) + 1, frames + 1)
        positions = np.searchsorted(events[:, 2], edges)
        boundaries = list(zip(positions[:-1], positions[1:]))
    else:
        raise ValueError("partition must be count or time")
    for i, (begin, end) in enumerate(boundaries):
        kwargs = {"bins": voxel_bins} if representation == "voxel_grid" else {}
        output[i] = operators[representation](events[begin:end], roi=roi, **kwargs)
    return output
